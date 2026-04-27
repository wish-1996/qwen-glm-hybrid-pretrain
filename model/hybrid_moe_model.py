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
    def __init__(
        self,
        head_dim: int,
        mrope_section=None,
        theta=10000.0,
        *,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
        rope_scaling_base_len: int = 4096,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        assert head_dim % 2 == 0, "head_dim must be even"
        
        # mrope_section 的总和应该是 half_dim
        required_sum = self.half_dim
        
        # 根据 head_dim 自动计算 mrope_section
        if mrope_section is None:
            # 对于 head_dim=128，half_dim=64，required_sum=64
            # 分配为 (22, 22, 20) 总和为 64
            if head_dim == 128:
                mrope_section = (22, 22, 20)
            # 对于其他 head_dim 值，可以根据需要调整
            else:
                # 简单分配：将 required_sum 分成三部分
                part = required_sum // 3
                mrope_section = (part, part, required_sum - 2 * part)
        
        assert sum(mrope_section) == required_sum, f"sum(mrope_section) must equal {required_sum}, got {mrope_section}"

        self.mrope_section = tuple(int(x) for x in mrope_section)

        self.theta_base = float(theta)
        inv_freq = 1.0 / (self.theta_base ** (torch.arange(0, self.half_dim, dtype=torch.float32) / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # base inv_freq（不缩放）

        # RoPE scaling（长上下文策略）
        self.rope_scaling_type = str(rope_scaling_type or "none")
        self.rope_scaling_factor = float(rope_scaling_factor or 1.0)
        self.rope_scaling_base_len = int(rope_scaling_base_len or 4096)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _dynamic_ntk_inv_freq_t(self, *, device: torch.device, dtype: torch.dtype, seq_len: int) -> torch.Tensor:
        """
        dynamic NTK（只用于 t 轴）：
        - 根据当前序列长度 seq_len 与训练基准长度 base_len 动态计算 alpha
        - 使用 NTK-aware 的 base scaling：theta' = theta * alpha^(d/(d-2))
          （常见近似写法来自 NTK-aware / dynamic NTK 讨论）
        - 然后用 theta' 重新生成 inv_freq，并切出 t 轴对应的 nt 段
        """
        base_len = max(1, int(self.rope_scaling_base_len))
        alpha = max(1.0, float(seq_len) / float(base_len))

        # d 用 head_dim（RoPE 的旋转维度口径通常按 head_dim）
        d = float(self.head_dim)
        if d <= 2:
            power = 1.0
        else:
            power = d / (d - 2.0)

        theta_eff = self.theta_base * (alpha ** power)

        # 重新生成 inv_freq_eff（全 half_dim），再切出 t 段
        ar = torch.arange(0, self.half_dim, device=device, dtype=torch.float32)
        inv_freq_eff = 1.0 / (theta_eff ** (ar / float(self.half_dim)))  # [half_dim]
        nt, _, _ = self.mrope_section
        return inv_freq_eff[:nt].to(device=device, dtype=dtype)

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

        # -------------------------
        # RoPE scaling：只对 t 轴做缩放（更符合多模态 3D RoPE 的语义）
        # -------------------------
        nt, nh, nw = self.mrope_section

        # 默认 inv_freq（不缩放）
        inv_t = self.inv_freq[:nt]
        inv_h = self.inv_freq[nt:nt + nh]
        inv_w = self.inv_freq[nt + nh:nt + nh + nw]

        if self.rope_scaling_type == "linear":
            if self.rope_scaling_factor and self.rope_scaling_factor != 1.0:
                pos_t = pos_t / self.rope_scaling_factor
        elif self.rope_scaling_type in ("none", "", None):
            pass
        elif self.rope_scaling_type == "ntk":
            # 固定 NTK：用 rope_scaling_factor 作为 alpha（目标/训练 的比例）
            # theta' = theta * alpha^(d/(d-2))
            alpha = float(self.rope_scaling_factor)
            d = float(self.head_dim)
            power = d / (d - 2.0) if d > 2 else 1.0
            theta_eff = self.theta_base * (alpha ** power)
            ar = torch.arange(0, self.half_dim, device=device, dtype=torch.float32)
            inv_freq_eff = 1.0 / (theta_eff ** (ar / float(self.half_dim)))
            inv_t = inv_freq_eff[:nt].to(device=device, dtype=inv_t.dtype)
        elif self.rope_scaling_type == "dynamic_ntk":
            # 动态 NTK：alpha 根据当前 seq_len 动态计算
            seq_len = int(pos_t.max().item()) + 1 if pos_t.numel() else 0
            inv_t = self._dynamic_ntk_inv_freq_t(device=device, dtype=inv_t.dtype, seq_len=seq_len)
        else:
            raise ValueError(f"Unknown rope_scaling_type: {self.rope_scaling_type}")

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
        self.deltanet_chunk_size = int(getattr(config, "deltanet_chunk_size", 256))
        
        # 投影层 (K/V 维度小于 Q，体现 GQA)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # 初始化 MROPE
        self.mrope = MROPE(
            self.head_dim,
            rope_scaling_type=getattr(config, "rope_scaling_type", "none"),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
            rope_scaling_base_len=getattr(config, "rope_scaling_base_len", 4096),
        )

    def forward(self, x, positions, past_state=None, use_cache=False):
        # x: 输入 hidden states，形状 [B, N, H]，示例：[2, 228, 2048]
        # B=批次大小, N=序列长度, H=hidden_size
        B, N, _ = x.shape
        
        # 1. 投影：将 hidden_size 投影到 Q/K/V 维度
        # Q: [B, N, H] -> [B, N, num_heads*head_dim] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        # 示例：q_proj: [2, 228, 2048] -> [2, 228, 16*128=2048] -> [2, 16, 228, 128]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, N, head_dim]
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, N, head_dim]
        g = torch.sigmoid(self.gate_proj(x)).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, N, head_dim]，门控值
        
        # 2. 应用 M-RoPE：使用 3D 位置编码旋转 Q/K
        # positions: [B, T, 3] -> [t, h, w]
        # 输出形状不变：q, k 仍然是 [B, num_heads, N, head_dim] 和 [B, num_kv_heads, N, head_dim]
        q, k = self.mrope(q, k, positions)
        
        # 3. 归一化 (稳定训练关键)
        q = F.normalize(q, p=2, dim=-1)  # L2 归一化
        k = F.normalize(k, p=2, dim=-1)
        
        # 4. Gated Delta Update (推理模式：递归)
        # use_cache=True 且 N==1 时启用增量推理
        if use_cache and N == 1:
            # 初始化或获取 KV state
            # state 形状：(B, num_kv_heads, head_dim, head_dim)
            # 示例：(2, 4, 128, 128)，4 个 KV 头，每个 128x128
            if past_state is None:
                state = torch.zeros(B, self.num_kv_heads, self.head_dim, self.head_dim, device=x.device)
            else:
                state = past_state
            
            # Delta 更新: S = S + g * (k^T @ v)
            # k^T: [B, num_kv_heads, head_dim, 1]
            # v: [B, num_kv_heads, 1, head_dim]
            # kv_outer: [B, num_kv_heads, head_dim, head_dim]
            kv_outer = torch.matmul(k.transpose(-2, -1), v.unsqueeze(-2)) # (B, H_kv, d, d)
            # 广播 g 以匹配 state 维度
            # g: [B, num_kv_heads, 1, head_dim]
            new_state = state + g.unsqueeze(-1) * kv_outer
            
            # 输出: Q @ State
            # 需要将 State (KV_Heads) 广播给 Q (Num_Heads)
            # GQA 逻辑：每 (num_heads // num_kv_heads) 个 Q 共享一个 State
            groups = self.num_heads // self.num_kv_heads  # 16/4=4，每4个Q头共享1个KV头
            state_expanded = state.repeat_interleave(groups, dim=1) # (B, Num_Heads, d, d)，示例：(2, 16, 128, 128)
            
            # Q @ State: [B, num_heads, 1, head_dim] @ [B, num_heads, head_dim, head_dim] -> [B, num_heads, 1, head_dim]
            output = torch.matmul(q.unsqueeze(-2), state_expanded).squeeze(-2)
            output = output.transpose(1, 2).contiguous().view(B, N, self.hidden_size)
            
            return self.out_proj(output), new_state

        else:
            # 训练模式：chunk-wise 向量化（P0-3）
            #
            # 原实现是：
            #   state_t = state_{t-1} + update_t
            #   out_t   = q_t @ state_t
            # 其中 update_t = g_t * outer(k_t, v_t)，outer(i,j)=k_i*v_j（再按 g_i 做行缩放）
            #
            # 这里按 chunk 计算，避免 O(N) Python 循环，同时控制显存峰值。
            state = torch.zeros(B, self.num_kv_heads, self.head_dim, self.head_dim, device=x.device, dtype=q.dtype)
            groups = self.num_heads // self.num_kv_heads  # 16/4=4
            outputs = []

            chunk = max(1, int(self.deltanet_chunk_size))
            for start in range(0, N, chunk):
                end = min(N, start + chunk)
                T = end - start

                # [B, H_kv, T, d]
                k_c = k[:, :, start:end, :]
                v_c = v[:, :, start:end, :]
                g_c = g[:, :, start:end, :]
                q_c = q[:, :, start:end, :]  # [B, H_q, T, d]

                # update_t(i,j) = (g_i*k_i) * v_j
                # updates: [B, H_kv, T, d, d]
                updates = (g_c * k_c).unsqueeze(-1) * v_c.unsqueeze(-2)

                # state_seq: [B, H_kv, T, d, d]，每个时间步的"更新后 state"
                state_seq = updates.cumsum(dim=2) + state.unsqueeze(2)

                # 不做 state.repeat_interleave，直接按 GQA 分组计算：
                # q_rg: [B, H_kv, groups, T, d]
                q_c = q_c.contiguous()
                q_rg = q_c.view(B, self.num_kv_heads, groups, T, self.head_dim)

                # out_rg: [B, H_kv, groups, T, d]
                out_rg = torch.einsum("bhgtd,bhtde->bhgte", q_rg, state_seq)

                # [B, H_q, T, d]
                out_c = out_rg.reshape(B, self.num_heads, T, self.head_dim)
                outputs.append(out_c)

                # 更新 state 到 chunk 末尾
                state = state_seq[:, :, -1, :, :].contiguous()

            # [B, H_q, N, d] -> [B, N, H]
            output = torch.cat(outputs, dim=2)
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
        self.attention_backend = getattr(config, "attention_backend", "torch")
        self.attention_causal = bool(getattr(config, "attention_causal", True))
        self.use_flash_attn = bool(getattr(config, "use_flash_attn", False))
        self.flash_dropout_p = float(getattr(config, "flash_attn_dropout", 0.0))
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # 初始化 MROPE
        self.mrope = MROPE(
            self.head_dim,
            rope_scaling_type=getattr(config, "rope_scaling_type", "none"),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
            rope_scaling_base_len=getattr(config, "rope_scaling_base_len", 4096),
        )

    def forward(self, x, positions, attention_mask: torch.Tensor | None = None):
        # x: 输入 hidden states，形状 [B, N, H]，示例：[2, 228, 2048]
        # positions: 3D 位置编码，形状 [B, N, 3]，示例：[2, 228, 3]
        B, N, _ = x.shape
        
        # 投影：Q/K/V 计算
        # q: [B, N, H] -> [B, N, num_heads*head_dim] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        # 示例：[2, 228, 2048] -> [2, 16, 228, 128]
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, N, head_dim]
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, num_kv_heads, N, head_dim]
        
        # M-RoPE：应用 3D 位置编码旋转
        # 输出形状不变
        q, k = self.mrope(q, k, positions)

        # -------------------------
        # P0-2：Flash Attention 路径（优先）
        # - 仅支持 causal（自回归）+ 不带显式 mask（后续做 varlen/packing 再上 varlen 接口）
        # - GQA/MQA：flash-attn 允许 K/V head 数 < Q head 数（避免 repeat_interleave 的显存/带宽浪费）
        # -------------------------
        if self.use_flash_attn and attention_mask is None:
            try:
                from flash_attn import flash_attn_func  # type: ignore
            except Exception:
                flash_attn_func = None

            if flash_attn_func is not None:
                # flash-attn 期望 shape: [B, T, H, Dh]
                q_bt = q.transpose(1, 2).contiguous()
                k_bt = k.transpose(1, 2).contiguous()
                v_bt = v.transpose(1, 2).contiguous()

                out_bt = flash_attn_func(
                    q_bt,
                    k_bt,
                    v_bt,
                    dropout_p=self.flash_dropout_p,
                    causal=True,
                )  # [B, T, Hq, Dh]

                out = out_bt.reshape(B, N, -1)
                return self.out_proj(out)

        # GQA 扩展 K/V 以匹配 Q
        # groups = num_heads / num_kv_heads = 16/4 = 4
        # 扩展后：k, v 从 [B, 4, N, 128] -> [B, 16, N, 128]
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)  # [B, num_heads, N, head_dim]
        v = v.repeat_interleave(groups, dim=1)  # [B, num_heads, N, head_dim]
        
        # ------------------------------------------
        # Attention backend（torch / flash），并保证 causal
        # ------------------------------------------
        use_flash = False
        flash_attn_func = None
        if (
            self.attention_backend == "flash"
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            try:
                # flash-attn 2.x
                from flash_attn import flash_attn_func as _flash_attn_func
                flash_attn_func = _flash_attn_func
                use_flash = True
            except Exception:
                use_flash = False

        # flash-attn 这里先支持“无 padding（全 1）”的情况；否则回退 torch
        if use_flash and attention_mask is not None:
            if not torch.all(attention_mask == 1):
                use_flash = False

        if use_flash and flash_attn_func is not None:
            # flash_attn_func 期望 [B, N, H, D]
            q_ = q.transpose(1, 2).contiguous()  # [B,N,H,D]
            k_ = k.transpose(1, 2).contiguous()
            v_ = v.transpose(1, 2).contiguous()
            out_ = flash_attn_func(q_, k_, v_, dropout_p=0.0, causal=self.attention_causal)  # [B,N,H,D]
            out = out_.contiguous().view(B, N, -1)
            return self.out_proj(out)

        # ---- torch fallback（支持 causal + padding mask）----
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,N,N]

        if self.attention_causal:
            causal = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))

        if attention_mask is not None:
            # attention_mask: [B,N], 1=valid, 0=pad
            key_mask = (attention_mask[:, None, None, :] == 0)  # [B,1,1,N]
            scores = scores.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        
        # 输出: attn @ v -> [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, H]
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)

# ==========================================
# 4.1 MTP (Multi-Token Prediction) 模块
# ==========================================
# 说明：
# - MTP 的实现/训练目标已抽离到 `model/mtp.py`
# - speculative decoding 已抽离到 `model/spec_decode.py`
# - 参数量估算脚本已抽离到 `tools/param_count.py`

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

    def forward(self, x, positions, attention_mask: torch.Tensor | None = None, past_state=None, use_cache=False):
        # Attention
        residual = x
        x = self.norm1(x)
        if self.is_linear:
            x, new_state = self.attn(x, positions, past_state=past_state, use_cache=use_cache)
            return self.moe(self.norm2(x + residual)), new_state
        else:
            x = self.attn(x, positions, attention_mask=attention_mask)
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

        A) 旧格式（"旁路 pixel_values + concat embeddings"）
           - input_ids: [B, T_text]
           - pixel_values: [B, 3, H, W]
           - 模型内部把 image_embeds 与 text_embeds 直接 concat

        B) 新格式（"token 序列里显式占位 <image_pad>"）
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
            x, state = layer(x, positions, attention_mask=attention_mask, use_cache=use_cache)
            if state is not None:
                past_states.append(state)
            if hasattr(layer.moe, 'aux_loss'):
                aux_loss += layer.moe.aux_loss

        x = self.norm(x)            # [B, T, H]
        logits = self.lm_head(x)    # [B, T, V]

        if output_hidden_states:
            return logits, x, past_states, aux_loss  # logits + hidden_states
        return logits, past_states, aux_loss

# -------------------------
# Backward compatibility
# -------------------------
# 保留历史类名，避免旧代码/旧checkpoint加载路径立刻失效
Qwen35Model = HybridMMMoEModel

__all__ = ["HybridMMMoEModel", "Qwen35Model"]
