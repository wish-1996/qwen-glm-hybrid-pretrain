# 02 模型侧：原生多模态 + MoE

## 模型架构

项目采用融合 Qwen 工程风格与 GLM-5 MoE 思路的多模态预训练模型架构。

### 核心组件

1. **`HybridMMMoEModel`**：项目级模型入口，包含以下组件：
   - **文本编码器**：基于 Transformer 架构
   - **视觉编码器**：包含补丁嵌入和视觉块
   - **多模态融合层**：将图像和文本特征融合
   - **MoE 层**：包含稀疏专家和共享专家

2. **`VisionEncoder`**：视觉编码器，将图像像素转换为视觉特征
   - **补丁嵌入**：将图像分割成补丁并进行线性投影
   - **位置嵌入**：为每个补丁添加位置信息
   - **视觉块**：包含注意力和 MLP 层

3. **`SharedExpertMoE`**：共享专家 MoE 实现（已抽离到 `model/moe.py`，便于独立阅读/测试/复用）
   - **稀疏专家**：动态路由，使用 SwiGLU 激活函数
   - **共享专家**：所有 token 都会经过
   - **门控网络**：负责路由决策
   - **负载均衡损失**：确保专家负载均衡

4. **`Qwen35Block`**：模型块，支持混合注意力策略
   - **Gated DeltaNet**：线性注意力，用于大部分层
   - **Standard Attention**：标准注意力，用于顶层

## 多模态融合

采用 early-fusion 方案：
1. 将图像通过视觉编码器转换为特征序列
2. 将文本通过嵌入层转换为特征序列
3. 拼接图像和文本特征序列
4. 输入到 Transformer 编码器进行处理

## MoE 实现

1. **路由机制**：通过门控网络为每个 token 选择 top-k 个专家
2. **专家网络**：使用 SwiGLU 激活函数的前馈网络
3. **负载均衡**：计算辅助损失确保专家负载均衡
4. **共享专家**：所有 token 都会经过共享专家，提高模型鲁棒性

## 位置编码

使用 M-RoPE（Multimodal RoPE / 3D-RoPE）：
- 支持 3D 位置编码 [t, h, w]
- 文本位置：[t, 0, 0]
- 图像补丁位置：[0, h, w]

## 前向传播流程

1. 处理文本输入：通过嵌入层获取文本特征
2. 处理图像输入：通过视觉编码器获取图像特征
3. 融合文本和图像特征：拼接特征序列
4. 构造位置编码：为文本和图像部分创建相应的位置编码
5. 通过 Transformer 层：包含注意力和 MoE 处理
6. 输出 logits：通过语言模型头生成预测

---

## 端到端 Shape Walkthrough（用一个真实 batch 把 seq_len / hidden_size 讲清楚）

这一节的目标：用**一个 batch**把“数据张量如何一步步变成模型 logits”讲明白。

我们用你在 `test/check_multimodal_alignment.py` 里常用的参数作为示例：

- batch_size：`B = 2`
- max_length：`T_text = 32`
- image_size：`224`
- patch_size：`16`  → `T_img = (224/16)^2 = 14^2 = 196`
- 所以 `T_total = T_img + T_text = 196 + 32 = 228`

再结合 `configs/model_config.py` 的默认模型超参：

- hidden_size：`H = 2048`
- num_attention_heads：`n_head = 16` → `head_dim = 2048/16 = 128`
- num_kv_heads：`n_kv = 4`（GQA：16 个 Q 头共享 4 个 KV 头）
- vocab_size：`V = 151936`
- num_layers：`L = 28`
- MoE：`num_experts=192, top_k=4`

> 代码位置提示：
> - 数据对齐（Step1）：`data/multimodal_sequence_alignment.py`
> - 模型实现：`model/hybrid_moe_model.py`

### Step 0：DataLoader 原始 batch（还没做 Step1 对齐）

来自 `data/multimodal_data_loader.py` 的 batch 结构：

| key | dtype | shape | 含义 |
|---|---:|---:|---|
| `input_ids` | int64 | `[B, T_text] = [2, 32]` | 文本 token ids（已 padding/truncation 到 max_length） |
| `attention_mask` | int64 | `[2, 32]` | 文本 mask：1=真实 token，0=padding |
| `pixel_values` | float32 | `[B, 3, 224, 224]` | 图片像素（RGB，resize 到 224×224，归一化到 [0,1]） |

此时你能明确看到：**文本序列长度是 32，但图像 token 还没进入序列**（图像目前还是旁路的 `pixel_values`）。

### Step 1：多模态对齐（把图片占位符 token 显式拼进 input_ids）

对齐函数：`build_aligned_masks_and_labels(...)`（`data/multimodal_sequence_alignment.py`）

它会构造：

1) `input_ids_total`：显式图片占位符（更像生产）

```text
input_ids_total[b] =
[<|image_pad|> * 196] + [text tokens * 32]
len = 228
```

也就是：

- `input_ids_total.shape == [2, 228]`
- `input_ids_total[:, :196]` 全是 `image_pad_token_id`（默认 151655）
- `input_ids_total[:, 196:]` 是文本 token ids

2) `attention_mask_total`：图像区全 1 + 文本区沿用原 attention_mask

- `attention_mask_total.shape == [2, 228]`
- `attention_mask_total[:, :196] == 1`

3) `labels_total`：图像区全 -100 + 文本区用 input_ids（但 padding 位置必须是 -100）

- `labels_total.shape == [2, 228]`
- `labels_total[:, :196] == -100`（图像 token 不参与 CE loss）
- 文本 padding 位置 `attention_mask==0` 的 label 也会被置为 -100（不参与 loss）

4) （可选）`positions_total`：3D 位置信息（长度也对齐到 228）

```text
positions_total[b, 0:196]   = [0, h, w]    # 图像 patch：h,w 是 14×14 网格坐标
positions_total[b, 196:228] = [t, 0, 0]    # 文本 token：t=0..31
```

### Step 2：进入模型 forward（token embedding + vision embedding 替换）

模型入口：`HybridMMMoEModel.forward(...)`（`model/hybrid_moe_model.py`）

#### 2.1 token embedding

```python
x = embed(input_ids_total)
```

- `x.shape == [B, T_total, H] = [2, 228, 2048]`

注意：此时 `x[:, :196, :]` 还是“图片占位符 `<|image_pad|>`”查表得到的 embedding（理论上会相同）。

#### 2.2 vision encoder（把 pixel_values 变成 patch embeddings）

视觉编码器：`VisionEncoder`（同文件）

1) patch embed（conv 切 patch）

```python
x_img = Conv2d(pixel_values) -> flatten -> LayerNorm
```

- 输入：`pixel_values` `[2, 3, 224, 224]`
- `Conv2d(kernel=16,stride=16)`：`[2, 2048, 14, 14]`
- flatten：`[2, 196, 2048]`

2) 视觉位置 embedding（learned）

位置嵌入的作用是为每个图像补丁添加位置信息，使模型能够理解补丁在图像中的空间位置。具体实现：

```python
# 获取输入特征的维度信息
batch_size, seq_len, _ = x.shape  # x: [B, seq_len, hidden_size]
# 生成位置ID序列，范围从0到seq_len-1
position_ids = torch.arange(seq_len, device=x.device)  # position_ids: [seq_len]
# 通过嵌入层获取位置嵌入向量，并扩展到整个批次
# self.pos_embed(position_ids): [seq_len, hidden_size]
# unsqueeze(0): [1, seq_len, hidden_size]
# expand(batch_size, -1, -1): [B, seq_len, hidden_size]
pos_embeddings = self.pos_embed(position_ids).unsqueeze(0).expand(batch_size, -1, -1)
# 将位置嵌入与输入特征相加，为每个补丁添加位置信息
x = x + pos_embeddings  # x: [B, seq_len, hidden_size]
```

- `pos_embed: Embedding(196, 2048)`：可学习的位置嵌入层
- 为什么相加就能成为image_embed？
  - 补丁嵌入（patch embed）捕获了每个局部区域的视觉特征
  - 位置嵌入（position embed）提供了每个补丁的空间位置信息
  - 两者相加后，每个补丁的特征向量既包含了视觉信息，又包含了位置信息
  - 这样模型就能理解图像中不同位置的内容及其相互关系

3) 视觉 blocks（默认 4 层）

每个 `VisionBlock` 包含注意力机制和 MLP 层：

```python
class VisionBlock(nn.Module):
    def forward(self, hidden_states):
        # 注意力层：学习补丁之间的空间关系
        attn_output, _ = self.attn(
            self.norm1(hidden_states),
            self.norm1(hidden_states),
            self.norm1(hidden_states)
        )
        hidden_states = hidden_states + attn_output  # 残差连接
        
        # MLP 层：增强特征表达
        mlp_output = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp_output  # 残差连接
        
        return hidden_states
```

- 注意力机制：捕获补丁之间的全局依赖关系
- MLP 层：进一步处理和增强特征表示
- 残差连接：帮助梯度流动，提高模型训练稳定性

4) 输出：

- `image_embeds.shape == [2, 196, 2048]`：包含位置信息的视觉特征

#### 2.3 替换占位符 embedding（关键：占位符只是“占坑”，最终会被视觉特征覆盖）

当你传入了 `image_pad_token_id` 且 input_ids 的前 196 个位置确实是 `<|image_pad|>` 时：

```python
x[:, :T_img, :] = image_embeds
```

替换之后：

- `x.shape` 仍然是 `[2, 228, 2048]`
- 但 `x[:, :196, :]` 现在已经是**真实视觉 patch 特征**（不再是同一个占位符 embedding）

### Step 3：Transformer 主干（注意力 + MoE）

主干层数：`L=28`，每层是 `Qwen35Block`：

1) Attention
- 低层：`GatedDeltaNet`（线性注意力路线）
- 高层：`StandardAttention`（softmax 注意力路线）

两者都会用到 `MROPE(q,k,positions_total)`：把 3D positions 通过 RoPE 的“旋转”注入到 Q/K 中。

2) MoE FFN
- `SharedExpertMoE`（已抽离到 `model/moe.py`）
- top-k routing（K=4），并在训练模式下产出 aux_loss（负载均衡）

每一层的输入/输出 hidden states shape 都保持：

- `x: [2, 228, 2048] -> [2, 228, 2048]`

#### KV Cache（past_states）详解

在推理加速中，模型会使用 KV Cache 来避免重复计算：

```python
past_states = []  # 存储每层的 KV cache

for layer in self.layers:
    x, state = layer(x, positions, use_cache=use_cache)
    if state is not None:
        past_states.append(state)  # 缓存这一层的 K, V
```

- **past_state**：之前计算过的 Key-Value 缓存
- **new_state**：当前层更新后的 KV cache
- **past_states**：所有层的 KV cache 列表

**作用**：在自回归生成时，避免重复计算已处理 token 的注意力，大幅加速推理！

### Step 4：输出 logits（语言模型头）

最后：

```python
x = RMSNorm(x)
logits = lm_head(x)
```

- `logits.shape == [B, T_total, V] = [2, 228, 151936]`

### Step 5：loss 只在文本区计算（图像区 labels=-100 被忽略）

训练里常见的 next-token CE：

```python
shift_logits = logits[:, :-1, :]      # [2, 227, V]
shift_labels = labels_total[:, 1:]    # [2, 227]
loss = CrossEntropy(ignore_index=-100)(shift_logits.reshape(-1, V), shift_labels.reshape(-1))
```

因为图像区的 labels 是 `-100`，所以图像 token 对 loss **没有直接监督**；
但图像 token 通过 self-attention 作为上下文参与文本预测，仍然会通过文本 loss 反向传播影响视觉/主干参数。

---

## MTP（Multi-Token Prediction）：为什么会出现 `T_total-2 / T_total-3` 这些 shape？

这一段专门解释你问的这个：

- `logits_main:  [B, T_total,   V]`
- `logits_step2: [B, T_total-2, V]`
- `logits_step3: [B, T_total-3, V]`

### 先把“预测目标”说清楚

标准 next-token LM（主任务）是：
- 用位置 `t` 的上下文去预测 `t+1`

MTP 做的是：**同一个位置 `t` 额外预测更远的未来**
- step=2：用位置 `t` 去预测 `t+2`
- step=3：用位置 `t` 去预测 `t+3`

### 为什么序列长度会变短？

因为序列末尾没有足够的“未来 token”可以监督。

举例：如果序列长度是 `T_total=228`：

#### 主任务（shift=1）
- 可用的 `t` 是 `0..226`（共 227 个位置）
- 目标是 `1..227`
- 所以参与 loss 的对齐张量是：
  - `shift_logits = logits_main[:, :-1, :]` → `[B, 227, V]`
  - `shift_labels = labels_total[:, 1:]` → `[B, 227]`

#### step=2（预测 t+2）
- 可用的 `t` 是 `0..225`（共 226 个位置）
- 目标是 `2..227`
- 所以对齐张量是：
  - `logits_step2 = lm_head(hidden[:, :-2, :])` → `[B, 226, V]`
  - `labels_step2 = labels_total[:, 2:]` → `[B, 226]`

#### step=3（预测 t+3）
- 可用的 `t` 是 `0..224`（共 225 个位置）
- 目标是 `3..227`
- 所以对齐张量是：
  - `logits_step3 = lm_head(hidden[:, :-3, :])` → `[B, 225, V]`
  - `labels_step3 = labels_total[:, 3:]` → `[B, 225]`

所以你看到的 `T_total-2 / T_total-3`，本质就是：
> **"为了预测更远的未来，序列末尾可用监督位置减少了 step 个。"**

### MTP loss 在训练里怎么合并？

常见做法：
- `loss = loss_main + mtp_weight * mean(loss_step2 .. loss_stepK) + aux_weight * aux_loss`

其中：
- `loss_main` 是 shift=1（标准 next-token）
- `loss_step2..K` 是多步预测的附加监督
- `labels_total` 的图像区/文本 padding 区本来就是 `-100`，因此 MTP 不会对这些区域产生监督污染
