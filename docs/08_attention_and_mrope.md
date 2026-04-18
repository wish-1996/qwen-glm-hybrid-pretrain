# 08 注意力实现与 3D RoPE（M-RoPE）

## 注意力实现

项目实现了两种注意力机制：
- **Gated DeltaNet**：线性注意力，用于低层，速度快
- **Standard Attention**：标准注意力，用于顶层，全局推理能力强

### 标准注意力（StandardAttention）

```python
class StandardAttention(nn.Module):
    def forward(self, x, positions, mask=None):
        # x: [B, N, H]
        # 投影 Q/K/V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 应用 M-RoPE
        q, k = self.mrope(q, k, positions)
        
        # GQA 扩展
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
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
            # 训练模式：顺序递推
            # ...
```

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
- **NTK Scaling**：预留，后续实现
- **Dynamic NTK**：预留，后续实现

**配置**：在 `configs/model_config.py` 中设置：
```python
rope_scaling_type: str = "linear"
rope_scaling_factor: float = 2.0
```

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
