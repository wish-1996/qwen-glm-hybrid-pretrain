"""
Speculative Decoding 模块

投机解码是一种加速 LLM 推理的技术：
- 使用一个轻量的 draft model 快速生成若干个 token
- 使用 target model 验证这些 token
- 只接受验证通过的 token，大幅减少推理步数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .mtp import MTPModel


@dataclass
class SpecDecodeOutput:
    generated_ids: torch.Tensor
    accepted_lengths: List[int]
    total_tokens: int
    acceptance_rate: float


class SpeculativeDecoder:
    """
    投机解码器

    原理：
    1. Draft model (MTPModel) 快速生成 spec_k 个候选 token
    2. Target model 逐个验证这些候选 token
    3. 如果验证通过，接受 token；如果验证失败，使用 target model 的预测

    优势：
    - 当 acceptance rate 高时，可以一次生成多个 token
    - 大幅加速自回归生成
    """

    def __init__(
        self,
        draft_model: MTPModel,
        target_model: nn.Module,
        device: str = "cuda",
    ):
        self.draft_model = draft_model.to(device)
        self.target_model = target_model.to(device)
        self.device = device
        self.draft_model.eval()
        self.target_model.eval()

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        max_new_tokens: int = 64,
        spec_k: int = 4,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> SpecDecodeOutput:
        """
        投机解码

        Args:
            input_ids: [1, T] 输入 token IDs
            positions: [1, T, 3] 位置编码 (M-RoPE 3D)
            max_new_tokens: 最大生成 token 数
            spec_k: 每次 draft 生成的 token 数
            eos_token_id: 结束符 token ID
            temperature: 采样温度，0 表示贪心
            top_k: Top-K 采样参数

        Returns:
            SpecDecodeOutput: 包含生成结果和统计信息
        """
        out_ids = input_ids.to(self.device)
        out_pos = positions.to(self.device)
        accepted_lengths: List[int] = []
        total_accepted = 0

        for _ in range(max_new_tokens):
            draft_tokens = self._draft_generate(
                out_ids, out_pos, spec_k=spec_k,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_k=top_k,
            )

            if not draft_tokens:
                break

            accepted = 0
            for tok in draft_tokens:
                logits, _, _ = self.target_model(
                    input_ids=out_ids,
                    positions=out_pos,
                    pixel_values=None,
                    attention_mask=None,
                    use_cache=False,
                    output_hidden_states=False,
                )
                target_next = int(torch.argmax(logits[:, -1, :], dim=-1).item())

                if target_next == int(tok):
                    next_id = torch.tensor([[tok]], device=self.device, dtype=out_ids.dtype)
                    out_ids = torch.cat([out_ids, next_id], dim=1)
                    t_next = out_pos[:, -1, 0] + 1
                    next_pos = torch.stack([
                        t_next,
                        torch.zeros_like(t_next),
                        torch.zeros_like(t_next)
                    ], dim=-1)
                    out_pos = torch.cat([out_pos, next_pos.unsqueeze(1)], dim=1)
                    accepted += 1

                    if eos_token_id is not None and tok == int(eos_token_id):
                        accepted_lengths.append(accepted)
                        total_accepted += accepted
                        return SpecDecodeOutput(
                            generated_ids=out_ids,
                            accepted_lengths=accepted_lengths,
                            total_tokens=out_ids.size(1) - input_ids.size(1),
                            acceptance_rate=total_accepted / max(1, sum(accepted_lengths)),
                        )
                else:
                    next_id = torch.tensor([[target_next]], device=self.device, dtype=out_ids.dtype)
                    out_ids = torch.cat([out_ids, next_id], dim=1)
                    t_next = out_pos[:, -1, 0] + 1
                    next_pos = torch.stack([
                        t_next,
                        torch.zeros_like(t_next),
                        torch.zeros_like(t_next)
                    ], dim=-1)
                    out_pos = torch.cat([out_pos, next_pos.unsqueeze(1)], dim=1)
                    accepted_lengths.append(accepted)
                    total_accepted += accepted
                    if eos_token_id is not None and target_next == int(eos_token_id):
                        return SpecDecodeOutput(
                            generated_ids=out_ids,
                            accepted_lengths=accepted_lengths,
                            total_tokens=out_ids.size(1) - input_ids.size(1),
                            acceptance_rate=total_accepted / max(1, sum(accepted_lengths)),
                        )
                    break
            else:
                accepted_lengths.append(accepted)
                total_accepted += accepted

            if eos_token_id is not None and int(out_ids[0, -1].item()) == int(eos_token_id):
                break

        return SpecDecodeOutput(
            generated_ids=out_ids,
            accepted_lengths=accepted_lengths,
            total_tokens=out_ids.size(1) - input_ids.size(1),
            acceptance_rate=total_accepted / max(1, sum(accepted_lengths)) if accepted_lengths else 0.0,
        )

    def _draft_generate(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        spec_k: int,
        eos_token_id: Optional[int],
        temperature: float,
        top_k: int,
    ) -> List[int]:
        """
        使用 draft model 生成候选 token

        Args:
            input_ids: 当前输入
            positions: 位置编码
            spec_k: 生成 token 数
            eos_token_id: 结束符
            temperature: 温度
            top_k: Top-K

        Returns:
            生成的 token 列表
        """
        if hasattr(self.draft_model, 'draft_generate'):
            return self.draft_model.draft_generate(
                input_ids=input_ids,
                positions=positions,
                spec_k=spec_k,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_k=top_k,
            )

        draft_tokens = []
        current_ids = input_ids
        current_pos = positions

        for _ in range(spec_k):
            if eos_token_id is not None and int(current_ids[0, -1].item()) == eos_token_id:
                break

            result = self.draft_model(
                input_ids=current_ids,
                positions=current_pos,
                pixel_values=None,
                attention_mask=None,
                use_cache=False,
                output_hidden_states=False,
            )

            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result

            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    top_k_vals, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.where(
                        torch.isin(torch.arange(logits.size(-1), device=logits.device), top_k_indices),
                        next_token_logits,
                        float('-inf')
                    )
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_tok = int(next_token.item())
            draft_tokens.append(next_tok)

            next_id = torch.tensor([[next_tok]], device=self.device, dtype=current_ids.dtype)
            current_ids = torch.cat([current_ids, next_id], dim=1)
            t_next = current_pos[:, -1, 0] + 1
            next_pos = torch.stack([
                t_next,
                torch.zeros_like(t_next),
                torch.zeros_like(t_next)
            ], dim=-1)
            current_pos = torch.cat([current_pos, next_pos.unsqueeze(1)], dim=1)

        return draft_tokens


def create_spec_decoder(
    draft_model: MTPModel,
    target_model: nn.Module,
    device: str = "cuda",
) -> SpeculativeDecoder:
    """
    创建投机解码器

    Args:
        draft_model: Draft model (MTPModel)
        target_model: Target model
        device: 设备

    Returns:
        SpeculativeDecoder 实例
    """
    return SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        device=device,
    )


__all__ = [
    "SpeculativeDecoder",
    "SpecDecodeOutput",
    "create_spec_decoder",
]
