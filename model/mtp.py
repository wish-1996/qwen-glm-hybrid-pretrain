"""
MTP (Multi-Token Prediction) 模块

包含：
- SharedMTPHead: 共享参数的 MTP 预测头
- MTPModel: 包装主干 LM，添加多步预测能力
- mtp_loss_from_hidden: 从隐藏状态计算 MTP loss

设计目标：
1. 可独立阅读/测试/复用
2. 支持训练时的 MTP loss 计算和推理时的 draft 生成
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


class SharedMTPHead(nn.Module):
    """
    共享参数的 MTP 预测头
    同一套参数预测 t+1, t+2, t+3... 多个未来 token
    避免传统实现中参数量随预测步数线性增长的问题
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class MTPModel(nn.Module):
    """
    MTP 模型：包装主干 LM，添加多步预测能力
    支持训练时的 MTP loss 计算和推理时的 draft 生成
    """
    def __init__(self, backbone: nn.Module, hidden_size: int, vocab_size: int, mtp_k: int = 3):
        super().__init__()
        self.backbone = backbone
        self.mtp_head = SharedMTPHead(hidden_size, vocab_size)
        self.mtp_k = mtp_k
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                attention_mask: torch.Tensor = None, labels: torch.Tensor = None,
                output_hidden_states: bool = True):
        """
        前向传播
        Args:
            input_ids: [B, T] 输入 token IDs
            positions: [B, T] 位置编码
            attention_mask: [B, T] 注意力掩码
            labels: [B, T] 标签（用于计算 loss）
            output_hidden_states: 是否输出隐藏状态
        Returns:
            logits_main: 主干 LM 的 logits
            logits_mtp_list: MTP 多步预测的 logits 列表
            loss: 总 loss（如果提供了 labels）
            hidden_states: 最后一层隐藏状态
        """
        out = self.backbone(input_ids=input_ids, positions=positions,
                           attention_mask=attention_mask, output_hidden_states=output_hidden_states)

        if isinstance(out, tuple):
            hidden = out[0]
            past_states = out[1] if len(out) > 1 else None
            aux_loss = out[2] if len(out) > 2 else None
        else:
            hidden = out.hidden_states[-1] if output_hidden_states else out.last_hidden_state
            past_states = None
            aux_loss = None

        logits_main = self.mtp_head(hidden)

        logits_mtp_list = []
        loss_mtp_total = 0.0

        if labels is not None and self.training:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

            # main loss（shift=1）
            shift_logits = logits_main[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_main = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

            # mtp loss（shift=2..K）
            loss_mtp_total = 0.0
            logits_mtp_list = []
            for step in range(2, self.mtp_k + 1):
                logits_step = self.mtp_head(hidden[:, :-step, :].contiguous())  # [B, T-step, V]
                labels_step = labels[:, step:].contiguous()                     # [B, T-step]

                loss_step = loss_fct(
                    logits_step.view(-1, self.vocab_size),
                    labels_step.view(-1),
                )
                loss_mtp_total = loss_mtp_total + loss_step
                logits_mtp_list.append(logits_step)

            loss_mtp = loss_mtp_total / max(1, (self.mtp_k - 1))
            mtp_weight = 0.3
            total_loss = loss_main + mtp_weight * loss_mtp

            if aux_loss is not None:
                total_loss = total_loss + 0.01 * aux_loss

            return {
                'logits_main': logits_main,
                'logits_mtp_list': logits_mtp_list,
                'loss': total_loss,
                'loss_main': float(loss_main.detach().cpu()),
                'loss_mtp': float(loss_mtp.detach().cpu()),
                'hidden_states': hidden,
                'past_states': past_states,
            }

        return {
            'logits_main': logits_main,
            'logits_mtp_list': logits_mtp_list,
            'hidden_states': hidden,
            'past_states': past_states
        }

    @torch.no_grad()
    def draft_generate(
        self,
        input_ids: torch.Tensor,          # [1, T]
        positions: torch.Tensor,          # [1, T, 3]
        spec_k: int = 4,
        eos_token_id: int | None = None,
        temperature: float = 0.0,         # 0=greedy
        top_k: int = 0,
    ):
        self.eval()
        out_ids = input_ids
        out_pos = positions
        draft_tokens = []

        for _ in range(spec_k):
            logits, _, _ = self.backbone(
                input_ids=out_ids,
                positions=out_pos,
                pixel_values=None,
                attention_mask=None,
                use_cache=False,
                output_hidden_states=False,
            )
            next_logits = logits[:, -1, :]  # [1, V]

            if temperature and temperature > 0:
                l = next_logits / temperature
                if top_k and top_k > 0:
                    v, _ = torch.topk(l, top_k, dim=-1)
                    l = torch.where(l < v[:, [-1]], torch.tensor(float("-inf"), device=l.device), l)
                probs = torch.softmax(l, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # [1,1]

            tok = int(next_id.item())
            draft_tokens.append(tok)

            out_ids = torch.cat([out_ids, next_id], dim=1)
            t_next = out_pos[:, -1, 0] + 1
            next_pos = torch.stack([t_next, torch.zeros_like(t_next), torch.zeros_like(t_next)], dim=-1)  # [1,3]
            out_pos = torch.cat([out_pos, next_pos.unsqueeze(1)], dim=1)  # [1, T+1, 3]

            if eos_token_id is not None and tok == int(eos_token_id):
                break

        return draft_tokens


@dataclass
class MTPOutput:
    """
    MTP loss 输出
    """
    loss_mtp: torch.Tensor
    loss_mtp_list: list[torch.Tensor]


def mtp_loss_from_hidden(
    hidden_states: torch.Tensor,      # [B, T, H]
    labels: torch.Tensor,             # [B, T]
    lm_head: nn.Linear,               # 语言模型头（可复用主干的 lm_head）
    mtp_k: int = 3,
    ignore_index: int = -100,
) -> MTPOutput:
    """
    从隐藏状态计算 MTP loss
    
    Args:
        hidden_states: 主干模型的隐藏状态，形状 [B, T, H]
        labels: 标签，形状 [B, T]
        lm_head: 语言模型头，用于生成 logits
        mtp_k: MTP 预测步数
        ignore_index: 忽略索引
    
    Returns:
        MTPOutput: 包含 MTP loss 和各步 loss 的列表
    """
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss_mtp_total = 0.0
    loss_mtp_list = []
    
    # mtp loss（shift=2..K）
    for step in range(2, mtp_k + 1):
        # 计算 logits_step: [B, T-step, V]
        logits_step = lm_head(hidden_states[:, :-step, :].contiguous())
        # 计算 labels_step: [B, T-step]
        labels_step = labels[:, step:].contiguous()
        
        # 计算 loss_step
        loss_step = loss_fct(
            logits_step.view(-1, logits_step.size(-1)),
            labels_step.view(-1),
        )
        
        loss_mtp_total += loss_step
        loss_mtp_list.append(loss_step)
    
    # 计算平均 MTP loss
    loss_mtp = loss_mtp_total / max(1, (mtp_k - 1))
    
    return MTPOutput(
        loss_mtp=loss_mtp,
        loss_mtp_list=loss_mtp_list,
    )



