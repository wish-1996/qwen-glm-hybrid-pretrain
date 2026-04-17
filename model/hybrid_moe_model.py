import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# MoE 组件拆分到独立模块，便于复用/测试/后续生产化（capacity / dropless / EP）
from .moe import SharedExpertMoE

# ==========================================
# 1. M-RoPE (Multimodal RoPE / 3D-RoPE)
# ==========================================
class MROPE(nn.Module):
    """
    简化可用版 M-RoPE（对齐 Qwen3.5 的 mrope_section 思路）
    - head_dim=128 时：half_dim=64，mrope_section=[11,11,10] 的和=32，对应 64 维（每个"旋转对"2维）
    """
    def __init__(self, head_dim: int, mrope_section=None, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        assert head_dim % 2 == 0, "head_dim must be even"
        
        # 每个 "旋转对" 占用 2 维，所以 mrope_section 的总和应该是 half_dim // 2
        required_sum = self.half_dim // 2
        
        # 根据 head_dim 自动计算 mrope_section
        if mrope_section is None:
            # 对于 head_dim=128，half_dim=64，required_sum=32
            # 分配为 (11, 11, 10) 总和为 32
            if head_dim == 128:
                mrope_section = (11, 11, 10)
            # 对于其他 head_dim 值，可以根据需要调整
            else:
                # 简单分配：将 required_sum 分成三部分
                part = required_sum // 3
                mrope_section = (part, part, required_sum - 2 * part)
        
        assert sum(mrope_section) == required_sum, f"sum(mrope_section) must equal {required_sum}, got {mrope_section}"

        self.mrope_section = tuple(int(x) for x in mrope_section)

        inv_freq = 1.0 / (theta ** (torch.arange(0, self.half_dim, dtype=torch.float32) / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, positions_3d):
        """
        q: [B, Hq, T, Dh]
        k: [B, Hk, T, Dh]
        positions_3d: [B, T, 3] -> [t, h, w]
        """
        device = q.device
        pos_t = positions_3d[:, :, 0].to(device=device, dtype=torch.float32)  # [B,T]
        pos_h = positions_3d[:, :, 1].to(device=device, dtype=torch.float32)
        pos_w = positions_3d[:, :, 2].to(device=device, dtype=torch.float32)

        nt, nh, nw = self.mrope_section
        inv_t = self.inv_freq[:nt]
        inv_h = self.inv_freq[nt:nt + nh]
        inv_w = self.inv_freq[nt + nh:nt + nh + nw]

        ang_t = torch.einsum("bt,d->btd", pos_t, inv_t)
        ang_h = torch.einsum("bt,d->btd", pos_h, inv_h)
        ang_w = torch.einsum("bt,d->btd", pos_w, inv_w)
        angles = torch.cat([ang_t, ang_h, ang_w], dim=-1)  # [B,T,half_dim]

        cos = torch.cos(angles).unsqueeze(1)  # [B,1,T,half_dim]
        sin = torch.sin(angles).unsqueeze(1)
        cos = torch.cat([cos, cos], dim=-1)  # [B,1,T,Dh]
        sin = torch.cat([sin, sin], dim=-1)

        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

# ==========================================
# 2. Gated Delta Networks (线性注意力 + GQA)
# ==========================================
class GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads # GQA 核心
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        
        # 投影层 (K/V 维度小于 Q，体现 GQA)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 初始化 MROPE
        self.mrope = MROPE(self.head_dim)

    def forward(self, x, positions, past_state=None, use_cache=False):
        B, N, _ = x.shape
        
        # 1. 投影
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        g = torch.sigmoid(self.gate_proj(x)).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 2. 应用 M-RoPE
        q, k = self.mrope(q, k, positions)
        
        # 3. 归一化 (稳定训练关键)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # 4. Gated Delta Update (推理模式：递归)
        if use_cache and N == 1:
            if past_state is None:
                # State shape: (B, KV_Heads, Head_Dim, Head_Dim)
                state = torch.zeros(B, self.num_kv_heads, self.head_dim, self.head_dim, device=x.device)
            else:
                state = past_state
            
            # Delta 更新: S = S + g * (k^T @ v)
            kv_outer = torch.matmul(k.transpose(-2, -1), v.unsqueeze(-2)) # (B, H_kv, d, d)
            # 广播 g 以匹配 state 维度
            new_state = state + g.unsqueeze(-1) * kv_outer
            
            # 输出: Q @ State
            # 需要将 State (KV_Heads) 广播给 Q (Num_Heads)
            # GQA 逻辑：每 (num_heads // num_kv_heads) 个 Q 共享一个 State
            groups = self.num_heads // self.num_kv_heads
            state_expanded = state.repeat_interleave(groups, dim=1) # (B, Num_Heads, d, d)
            
            output = torch.matmul(q.unsqueeze(-2), state_expanded).squeeze(-2)
            output = output.transpose(1, 2).contiguous().view(B, N, self.hidden_size)
            
            return self.out_proj(output), new_state

        else:
            # 训练模式：用"顺序递推"实现（正确但慢），先保证数学正确
            state = torch.zeros(B, self.num_kv_heads, self.head_dim, self.head_dim, device=x.device, dtype=q.dtype)
            outputs = []
            groups = self.num_heads // self.num_kv_heads

            for t in range(N):
                k_t = k[:, :, t, :]  # [B, H_kv, d]
                v_t = v[:, :, t, :]  # [B, H_kv, d]
                g_t = g[:, :, t, :]  # [B, H_kv, d]

                outer = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)          # [B,H_kv,d,d]
                state = state + g_t.unsqueeze(-1) * outer              # gated update

                state_expanded = state.repeat_interleave(groups, dim=1) # [B,H_q,d,d]
                q_t = q[:, :, t, :]                                     # [B,H_q,d]
                out_t = torch.matmul(q_t.unsqueeze(-2), state_expanded).squeeze(-2)  # [B,H_q,d]
                outputs.append(out_t)

            output = torch.stack(outputs, dim=2)  # [B, H_q, N, d]
            output = output.transpose(1, 2).contiguous().view(B, N, self.hidden_size)
            return self.out_proj(output), None

# ==========================================
# 3. Standard Attention (带 GQA + M-RoPE)
# ==========================================
class StandardAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # 初始化 MROPE
        self.mrope = MROPE(self.head_dim)

    def forward(self, x, positions, mask=None):
        #falsh_attention
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # M-RoPE
        q, k = self.mrope(q, k, positions)
        
        # GQA 扩展 K/V 以匹配 Q
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        
        # Softmax Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)

# ==========================================
# 4.1 MTP (Multi-Token Prediction) 模块
# ==========================================
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

# ==========================================
# 5. Qwen3.5 Block & Model
# ==========================================
from configs.qwen35_config import Qwen35Config

class Qwen35Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.total_layers = config.num_layers
        
        # 混合策略：最后 25% 层使用标准 Attention，其余使用 Gated DeltaNet
        threshold = int(self.total_layers * 0.75)
        
        if layer_idx >= threshold:
            print(f"Layer {layer_idx}: Standard Attention (Global Reasoning)")
            self.attn = StandardAttention(config)
            self.is_linear = False
        else:
            print(f"Layer {layer_idx}: Gated DeltaNet (Linear Speed)")
            self.attn = GatedDeltaNet(config, layer_idx)
            self.is_linear = True
            
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        
        # 共享专家 MoE
        self.moe = SharedExpertMoE(config)

    def forward(self, x, positions, past_state=None, use_cache=False):
        # Attention
        residual = x
        x = self.norm1(x)
        if self.is_linear:
            x, new_state = self.attn(x, positions, past_state=past_state, use_cache=use_cache)
            return self.moe(self.norm2(x + residual)), new_state
        else:
            x = self.attn(x, positions)
            return self.moe(self.norm2(x + residual)), None

class VisionPatchEmbed(nn.Module):
    """
    视觉补丁嵌入
    将图像分割成补丁并进行线性投影
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        
        # 计算补丁数量
        self.num_patches = (config.image_size // self.patch_size) ** 2
        
        # 线性投影层
        self.proj = nn.Conv2d(
            3, 
            self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, pixel_values):
        # pixel_values: [B, 3, 224, 224]
        x = self.proj(pixel_values)  # [B, hidden_size, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, hidden_size]
        x = self.norm(x)  # [B, 196, hidden_size]
        return x

class VisionBlock(nn.Module):
    """
    视觉编码器块
    包含注意力和 MLP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # 注意力层
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=0.1
        )
        
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

    def forward(self, hidden_states):
        # hidden_states: [B, seq_len, hidden_size]
        # 注意力层
        attn_output, _ = self.attn(
            self.norm1(hidden_states),
            self.norm1(hidden_states),
            self.norm1(hidden_states)
        )
        hidden_states = hidden_states + attn_output
        
        # MLP 层
        mlp_output = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp_output
        
        return hidden_states

class VisionEncoder(nn.Module):
    """
    视觉编码器
    将图像像素转换为视觉特征
    参考官方 Qwen3_5MoeVisionModel 实现
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 补丁嵌入
        self.patch_embed = VisionPatchEmbed(config)
        
        # 位置嵌入
        self.pos_embed = nn.Embedding(self.patch_embed.num_patches, config.hidden_size)
        
        # 视觉块
        self.blocks = nn.ModuleList([
            VisionBlock(config) for _ in range(4)  # 使用 4 个视觉块
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, pixel_values):
        # pixel_values: [B, 3, 224, 224]
        
        # 补丁嵌入
        x = self.patch_embed(pixel_values)  # [B, 196, hidden_size]
        
        # 添加位置嵌入
        # 1. 获取输入特征的维度信息
        batch_size, seq_len, _ = x.shape  # x: [B, seq_len, hidden_size]
        # 2. 生成位置ID序列，范围从0到seq_len-1
        position_ids = torch.arange(seq_len, device=x.device)  # position_ids: [seq_len]
        # 3. 通过嵌入层获取位置嵌入向量，并扩展到整个批次
        # self.pos_embed(position_ids): [seq_len, hidden_size]
        # unsqueeze(0): [1, seq_len, hidden_size]
        # expand(batch_size, -1, -1): [B, seq_len, hidden_size]
        pos_embeddings = self.pos_embed(position_ids).unsqueeze(0).expand(batch_size, -1, -1)
        # 4. 将位置嵌入与输入特征相加，为每个补丁添加位置信息
        x = x + pos_embeddings  # x: [B, seq_len, hidden_size]
        
        # 视觉块
        for block in self.blocks:
            x = block(x)
        
        # 层归一化
        x = self.norm(x)
        
        return x

class HybridMMMoEModel(nn.Module):
    def __init__(self, config, use_multimodal=False):
        super().__init__()
        self.config = config
        self.use_multimodal = use_multimodal
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.layers = nn.ModuleList([Qwen35Block(config, i) for i in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size)
        
        # 视觉编码器
        if self.use_multimodal:
            self.vision_encoder = VisionEncoder(config)
            # 多模态融合层
            self.multimodal_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(
        self,
        input_ids=None,
        positions=None,
        pixel_values=None,
        attention_mask=None,
        use_cache=False,
        output_hidden_states=False,
        *,
        image_pad_token_id: int | None = None,
    ):
        """
        多模态 forward（更像生产的写法）：

        支持两种输入格式：

        A) 旧格式（“旁路 pixel_values + concat embeddings”）
           - input_ids: [B, T_text]
           - pixel_values: [B, 3, H, W]
           - 模型内部把 image_embeds 与 text_embeds 直接 concat

        B) 新格式（“token 序列里显式占位 <image_pad>”）
           - input_ids: [B, T_total]，其中前 T_img 个位置是 image_pad_token_id
           - pixel_values: [B, 3, H, W]
           - 模型内部用 image_embeds 替换这段占位符 embedding
           - 好处：token 序列上可见图片位置，更容易对齐指令数据/packing/多图等生产需求
        """

        # 1) token embeddings（如果有 input_ids）
        if input_ids is not None:
            x = self.embed(input_ids)  # [B, T, H]
        else:
            x = None

        # 2) image embeddings（如果开启多模态）
        if self.use_multimodal and pixel_values is not None:
            image_embeds = self.vision_encoder(pixel_values)  # [B, T_img, H]
        else:
            image_embeds = None

        # 3) 融合
        if x is not None and image_embeds is not None:
            T_img = image_embeds.size(1)

            # --- 新格式：input_ids 里已有 image 占位符，直接替换 prefix 的 embedding ---
            if image_pad_token_id is not None and x.size(1) >= T_img:
                # 强校验：前 T_img 个 token 必须都是 image_pad_token_id，否则很可能是误用
                if not torch.all(input_ids[:, :T_img] == int(image_pad_token_id)):
                    raise ValueError(
                        "Multimodal expects input_ids prefix to be image_pad_token_id when image_pad_token_id is provided. "
                        f"got mismatch in first {T_img} tokens."
                    )
                x[:, :T_img, :] = image_embeds

            # --- 旧格式：input_ids 只有文本，走 concat ---
            else:
                x = torch.cat([image_embeds, x], dim=1)  # [B, T_img + T_text, H]

            # positions 对齐：
            # - 如果调用方传的是 text_positions（长度=T_text），这里补上 image_positions；
            # - 如果调用方已经传了 total positions（长度=T_total），这里不再重复拼接。
            if positions is not None:
                if positions.size(1) == x.size(1):
                    # 已经是 total positions，直接用
                    pass
                else:
                    # 认为传的是 text_positions：需要拼上 image_positions
                    batch_size = x.size(0)
                    grid = self.config.image_size // self.config.patch_size  # e.g. 14
                    image_positions = torch.zeros(batch_size, T_img, 3, device=positions.device, dtype=positions.dtype)
                    idx = 0
                    for h in range(grid):
                        for w in range(grid):
                            image_positions[:, idx, 0] = 0
                            image_positions[:, idx, 1] = h
                            image_positions[:, idx, 2] = w
                            idx += 1
                    positions = torch.cat([image_positions, positions], dim=1)

        elif x is not None:
            # 纯文本
            pass
        elif image_embeds is not None:
            # 纯图像（暂不作为主路径）
            x = image_embeds
        else:
            raise ValueError("Either input_ids or pixel_values must be provided")

        past_states = []
        aux_loss = 0.0

        for layer in self.layers:
            x, state = layer(x, positions, use_cache=use_cache)
            if state is not None:
                past_states.append(state)
            if hasattr(layer.moe, 'aux_loss'):
                aux_loss += layer.moe.aux_loss

        x = self.norm(x)            # [B, T, H]
        logits = self.lm_head(x)    # [B, T, V]

        if output_hidden_states:
            return logits, x, past_states, aux_loss  # logits + hidden_states
        return logits, past_states, aux_loss

# ==========================================
# 测试运行
# ==========================================
def calculate_theoretical_parameters(config):
    """理论计算模型参数量"""
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    num_attention_heads = config.num_attention_heads
    num_kv_heads = config.num_kv_heads
    num_experts = config.num_experts
    top_k = config.top_k
    intermediate_size = config.intermediate_size
    vocab_size = 32000
    
    # 1. 注意力层参数（共享部分）
    # 按照用户示例：单层注意力参数 ≈ 4 * d_model²
    attn_params_per_layer = 4 * hidden_size * hidden_size
    total_attn_params = attn_params_per_layer * num_layers
    
    # 2. 专家部分参数
    # 每个专家: 2 * d_model * d_ff
    expert_params_per_expert = 2 * hidden_size * intermediate_size
    total_expert_params = expert_params_per_expert * num_experts
    
    # 3. 嵌入层和输出层参数
    embed_params = vocab_size * hidden_size
    output_params = hidden_size * vocab_size
    
    # 4. 门控网络参数
    gate_params = hidden_size * num_experts
    
    # 总参数量（按照用户示例公式）
    # 总参数量 ≈ (专家数量 × 单个专家的大小) + (共享的注意力层等)
    total_params = total_expert_params + total_attn_params + embed_params + output_params + gate_params
    
    # 激活参数量（按照用户示例公式）
    # 激活参数量 = 共享部分 + (激活专家数 × 单个专家的大小)
    shared_params = total_attn_params
    activation_expert_params = expert_params_per_expert * top_k
    total_activation_params = shared_params + activation_expert_params
    
    return total_params, total_activation_params

if __name__ == "__main__":
    config = Qwen35Config()
    
    # 理论计算参数量
    total_params, total_activation_params = calculate_theoretical_parameters(config)
    
    print(f"\n--- 模型参数量统计 ---")
    print(f"总参数量: {total_params / 1e9:.2f}B")
    print(f"激活参数量: {total_activation_params / 1e9:.2f}B")
    print(f"目标总参数量: 7.00B")
    print(f"目标激活参数量: 0.60B")
    
    # 调整参数以达到目标
    print("\n--- 参数调整建议 ---")
    if total_params < 7e9:
        print(f"总参数量低于目标，建议增加专家数量或中间层维度")
    elif total_params > 7e9:
        print(f"总参数量高于目标，建议减少专家数量或中间层维度")
    else:
        print("总参数量符合目标")
    
    if total_activation_params < 0.6e9:
        print(f"激活参数量低于目标，建议增加激活专家数或单个专家大小")
    elif total_activation_params > 0.6e9:
        print(f"激活参数量高于目标，建议减少激活专家数或单个专家大小")
    else:
        print("激活参数量符合目标")


# ==========================================
# 6. Speculative Decoding 推理模块
# ==========================================
class SpeculativeDecoder:
    """
    可用版投机解码（先保证正确性，不做 KV cache 复用）
    - draft: 用 draft_model 生成 spec_k 个 token
    - verify: target_model 逐 token 验证（teacher forcing 一致性判定）
    - 统计每轮 accept length
    """

    def __init__(self, draft_model: MTPModel, target_model: MTPModel, device: str = "cuda"):
        self.draft_model = draft_model.to(device)
        self.target_model = target_model.to(device)
        self.device = device

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,      # [1,T]
        positions: torch.Tensor,      # [1,T,3]
        max_new_tokens: int = 64,
        spec_k: int = 4,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
    ):
        self.draft_model.eval()
        self.target_model.eval()

        out_ids = input_ids.to(self.device)
        out_pos = positions.to(self.device)
        accepted_lengths = []

        for _ in range(max_new_tokens):
            draft_tokens = self.draft_model.draft_generate(
                out_ids, out_pos, spec_k=spec_k,
                eos_token_id=eos_token_id, temperature=temperature, top_k=top_k
            )
            if not draft_tokens:
                break

            accepted = 0
            for tok in draft_tokens:
                logits, _, _ = self.target_model.backbone(
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
                    next_pos = torch.stack([t_next, torch.zeros_like(t_next), torch.zeros_like(t_next)], dim=-1)
                    out_pos = torch.cat([out_pos, next_pos.unsqueeze(1)], dim=1)
                    accepted += 1

                    if eos_token_id is not None and tok == int(eos_token_id):
                        accepted_lengths.append(accepted)
                        return out_ids, accepted_lengths
                else:
                    next_id = torch.tensor([[target_next]], device=self.device, dtype=out_ids.dtype)
                    out_ids = torch.cat([out_ids, next_id], dim=1)
                    t_next = out_pos[:, -1, 0] + 1
                    next_pos = torch.stack([t_next, torch.zeros_like(t_next), torch.zeros_like(t_next)], dim=-1)
                    out_pos = torch.cat([out_pos, next_pos.unsqueeze(1)], dim=1)
                    accepted_lengths.append(accepted)
                    if eos_token_id is not None and target_next == int(eos_token_id):
                        return out_ids, accepted_lengths
                    break
            else:
                accepted_lengths.append(accepted)

            if eos_token_id is not None and int(out_ids[0, -1].item()) == int(eos_token_id):
                break

        return out_ids, accepted_lengths


def mtp_training_step(model: MTPModel, input_ids: torch.Tensor,
                      positions: torch.Tensor, attention_mask: torch.Tensor,
                      labels: torch.Tensor, mtp_weight: float = 0.3):
    """
    MTP 训练步骤示例

    Args:
        model: MTP 模型
        input_ids: [B, T] 输入 token IDs
        positions: [B, T] 位置编码
        attention_mask: [B, T] 注意力掩码
        labels: [B, T] 标签
        mtp_weight: MTP loss 权重

    Returns:
        loss: 总 loss
        metrics: 训练指标
    """
    model.train()
    result = model(input_ids=input_ids, positions=positions,
                  attention_mask=attention_mask, labels=labels)

    loss = result['loss']
    metrics = {
        'loss_main': result['loss_main'],
        'loss_mtp': result['loss_mtp'],
        'total_loss': loss.item()
    }

    return loss, metrics


def create_mtp_model(config: Qwen35Config, mtp_k: int = 3) -> MTPModel:
    """
    创建带 MTP 的模型

    Args:
        config: 模型配置
        mtp_k: MTP 预测步数

    Returns:
        MTPModel 实例
    """
    backbone = HybridMMMoEModel(config, use_multimodal=False)
    model = MTPModel(
        backbone=backbone,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        mtp_k=mtp_k
    )
    return model

# -------------------------
# Backward compatibility
# -------------------------
# 保留历史类名，避免旧代码/旧checkpoint加载路径立刻失效
Qwen35Model = HybridMMMoEModel

__all__ = ["HybridMMMoEModel", "Qwen35Model"]
