# 08 注意力实现与 3D RoPE（M-RoPE）

## 注意力实现

项目实现了两种注意力机制：
- **Gated DeltaNet**：线性注意力，用于低层，速度快
- **Standard Attention**：标准注意力，用于顶层，全局推理能力强

### 标准注意力（StandardAttention）

目前 StandardAttention 支持两条路径：
- **torch**：纯 PyTorch matmul + softmax（最稳，任何环境都能跑）
- **flash**：可选 flash-attn（满足条件时启用，否则自动 fallback 到 torch）
- **flash_varlen**：基于 attention_mask 做 unpad/pad，走 flash-attn varlen（推荐配合 dynamic padding / packing）

```python
class StandardAttention(nn.Module):
    def forward(self, x, positions, attention_mask=None):
        # x: [B, N, H]
        # 投影 Q/K/V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 M-RoPE
        q, k = self.mrope(q, k, positions)

        # flash-attn 路径（GQA/MQA）：避免 repeat_interleave 扩 KV（节省显存/带宽）
        if self.use_flash_attn and attention_mask is None:
            # 这里省略具体 flash_attn_func 调用细节
            return self.out_proj(out_from_flash_attn)

        # torch 路径：如需兼容 GQA，会在这里 repeat_interleave（后续可继续优化）
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # 输出
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)
```

### 门控 DeltaNet（GatedDeltaNet）

线性注意力的一种变体，通过门控机制提高性能：

```python
class GatedDeltaNet(nn.Module):
    def forward(self, x, positions, past_state=None, use_cache=False):
        # 投影 Q/K/V/G
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        g = torch.sigmoid(self.gate_proj(x)).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 M-RoPE
        q, k = self.mrope(q, k, positions)

        # 归一化
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # 增量推理或训练模式
        if use_cache and N == 1:
            # 增量更新 state
            kv_outer = torch.matmul(k.transpose(-2, -1), v.unsqueeze(-2))
            new_state = state + g.unsqueeze(-1) * kv_outer
            # 输出
            output = torch.matmul(q.unsqueeze(-2), state_expanded).squeeze(-2)
            return self.out_proj(output), new_state
        else:
            # 训练模式：chunk-wise 向量化（P0-3）
            # - 去掉 token 级 for-loop（长序列下会非常慢）
            # - chunk size 通过 config.deltanet_chunk_size 控制（默认 256）
            # ...
```

#### 训练性能参数（P0-3）

在 `configs/model_config.py` 中配置：

```python
deltanet_chunk_size: int = 256
```

你可以用基准脚本快速检查 chunk-wise 是否生效：

```bash
python tools/bench_deltanet_chunk.py --seq 4096 --chunk 256 --dtype bf16 --batch 2
```

#### P1：varlen / packing（从 1024 起步更划算）

1) 多模态：dynamic padding（batch 内按最大长度 pad）
2) text-only：sample packing（EOS 拼接成近似满的 block）

当 `attention_backend=flash_varlen` 且 `USE_FLASH_ATTN=1` 时，StandardAttention 会根据 attention_mask 做 unpad/pad，
在长序列上可明显减少 padding 带来的无效计算。

## 3D RoPE（M-RoPE）

### 什么是 M-RoPE？

M-RoPE（Multimodal RoPE）是一种为多模态模型设计的 3D 位置编码方案：

- **文本位置**：`[t, 0, 0]` - 只在时间维度有位置信息
- **图像位置**：`[0, h, w]` - 只在空间维度有位置信息

### 实现原理

```python
class MROPE(nn.Module):
    def forward(self, q, k, positions_3d):
        device = q.device
        pos_t = positions_3d[:, :, 0].to(device=device, dtype=torch.float32)
        pos_h = positions_3d[:, :, 1].to(device=device, dtype=torch.float32)
        pos_w = positions_3d[:, :, 2].to(device=device, dtype=torch.float32)

        # RoPE scaling：只对 t 轴做缩放
        if self.rope_scaling_type == "linear":
            if self.rope_scaling_factor and self.rope_scaling_factor != 1.0:
                pos_t = pos_t / self.rope_scaling_factor

        # 计算旋转角度
        nt, nh, nw = self.mrope_section
        inv_t = self.inv_freq[:nt]
        inv_h = self.inv_freq[nt:nt + nh]
        inv_w = self.inv_freq[nt + nh:nt + nh + nw]

        ang_t = torch.einsum("bt,d->btd", pos_t, inv_t)
        ang_h = torch.einsum("bt,d->btd", pos_h, inv_h)
        ang_w = torch.einsum("bt,d->btd", pos_w, inv_w)
        angles = torch.cat([ang_t, ang_h, ang_w], dim=-1)

        # 应用旋转
        cos = torch.cos(angles).unsqueeze(1)
        sin = torch.sin(angles).unsqueeze(1)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k
```

### 长上下文扩展（RoPE Scaling）

为了支持更长的上下文，实现了 RoPE 缩放策略：

- **Linear Scaling**：将位置值除以缩放因子（默认 2.0）
- **NTK Scaling**：固定 NTK-aware base scaling（适合"已知目标长度/倍率"的场景）
- **Dynamic NTK**：动态 NTK-aware base scaling（根据当前 seq_len 动态计算 alpha，短序列≈不缩放，长序列自动增强）

**配置**：在 `configs/model_config.py` 中设置：
```python
rope_scaling_type: str = "linear"
rope_scaling_factor: float = 2.0
rope_scaling_base_len: int = 4096
```

dynamic NTK 的核心是：
- `alpha = max(1, L_current / rope_scaling_base_len)`
- `theta' = theta * alpha^(d/(d-2))`
- 用 `theta'` 重新生成 t 轴的 `inv_freq`（h/w 不变）

### dynamic NTK vs linear scaling：公式层面到底差在哪？

先记住 RoPE 的角度公式（对某个频率分量 i）：

`angle = t * inv_freq[i]`

#### A) linear scaling（线性缩放 / 线性插值）

做法：把位置缩小：

- `t' = t / s`（`s=rope_scaling_factor`）
- `angle' = (t/s) * inv_freq`

特点：**所有频率分量都被同样缩小 1/s**（均匀缩放），实现简单、但可能会牺牲一点短程精度。

#### B) (dynamic) NTK scaling

做法：不直接改 t，而是改 "base/theta"，重新生成 `inv_freq`：

- `alpha = max(1, L_current / base_len)`（dynamic 的关键：随长度变化）
- `theta' = theta * alpha^(d/(d-2))`
- `inv_freq' = 1 / theta'^(i/d)`
- `angle' = t * inv_freq'`

特点：**不同频率分量的变化幅度不同**（相当于"非均匀缩放频率分布"）：
- 低频/长程相关的分量会更"保守"（更稳地外推到长距离）
- 高频/短程相关的分量相对更接近原始，从而更容易保留短程能力

这也是为什么工程上常见的节奏是：
- 先用 linear 跑通 8k
- 再用 dynamic_ntk 冲 16k/32k（更稳）

### 为什么 3D RoPE 只对 t 轴 scaling，h/w 不动？

因为：
- `t` 是文本时间轴：会从 4k→8k→32k 持续增长，长上下文问题主要发生在这条轴上
- `h/w` 是图像空间轴：范围固定（例如 14×14），不存在"上下文外推"

把 `h/w` 也缩放会改变图像空间几何尺度，反而可能损伤视觉空间结构建模，所以保持不动更符合语义与工程直觉。

## 注意力与 M-RoPE 的结合

在 `Qwen35Block` 中，注意力计算会使用 M-RoPE：

```python
def forward(self, x, positions, past_state=None, use_cache=False):
    # 层归一化
    x = self.norm1(x)
    # 应用 M-RoPE 到 Q 和 K
    # 计算注意力
    if self.is_linear:
        x, new_state = self.attn(x, positions, past_state=past_state, use_cache=use_cache)
    else:
        x = self.attn(x, positions)
    # 残差连接和 MoE 处理
    x = self.moe(self.norm2(x + residual))
    return x, new_state
```

## 性能优化

1. **KV Cache**：避免重复计算，加速自回归生成
2. **线性注意力**：低层使用 Gated DeltaNet，降低计算复杂度
3. **批处理**：高效处理批量输入
4. **内存优化**：合理管理中间激活值

## 代码位置

- **注意力实现**：`model/hybrid_moe_model.py` 中的 `StandardAttention` 和 `GatedDeltaNet` 类
- **M-RoPE 实现**：`model/hybrid_moe_model.py` 中的 `MROPE` 类
- **注意力块**：`model/hybrid_moe_model.py` 中的 `Qwen35Block` 类
- **配置**：`configs/model_config.py` 中的 RoPE 相关参数
