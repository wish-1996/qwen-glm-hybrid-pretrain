# 05 多模态序列对齐（Step 1）

## 核心概念

多模态序列对齐是多模态预训练中的关键步骤，确保模型能够正确处理图像和文本的组合输入。

### 序列长度

- **文本序列长度**：`T_text`，由输入的 token 数量决定
- **图像序列长度**：`T_img`，由图像补丁数量决定，计算公式为 `(image_size // patch_size)²`
- **总序列长度**：`T_total = T_img + T_text`，模型输出的 logits 长度

### 对齐目标（更像生产：显式图片占位符）

1. **input_ids 对齐（新增）**：显式在 token 序列中插入图片占位符（如 `<|image_pad|>`），占用 `T_img` 个位置  
   - `input_ids_total = [<|image_pad|> * T_img] + [text tokens]`
2. **注意力掩码对齐**：确保图像部分的注意力掩码为 1（有效），文本部分保持原掩码
3. **标签对齐**：确保图像部分的标签为 -100（忽略），文本部分保持原标签（padding 位置也必须是 -100）
4. **位置编码对齐**：为图像和文本部分创建相应的位置编码

## 实现方案

### 注意力掩码处理

- **图像部分**：创建全 1 的注意力掩码，因为所有图像补丁都是有效的
- **文本部分**：保持原有的注意力掩码，0 表示填充
- **拼接**：将图像和文本的注意力掩码拼接成总掩码

### 标签处理

- **图像部分**：创建全 -100 的标签，因为图像部分不需要计算损失
- **文本部分**：使用输入的 token IDs 作为标签，因为我们要预测下一个 token
- **拼接**：将图像和文本的标签拼接成总标签

### 位置编码处理

- **文本位置**：使用 3D 位置编码 `[t, 0, 0]`，其中 t 是文本 token 的位置
- **图像位置**：使用 3D 位置编码 `[0, h, w]`，其中 h 和 w 是图像补丁的空间位置
- **拼接**：将图像和文本的位置编码拼接成总位置编码

## 代码实现

### 核心函数

```python
def build_aligned_masks_and_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_size: int,
    patch_size: int,
    pad_ignore_index: int = -100,
) -> AlignedOutput:
    # 计算图像补丁数量
    num_patches = (image_size // patch_size) ** 2
    T_img = num_patches
    
    # 创建图像部分的注意力掩码
    attention_mask_img = torch.ones(B, T_img, dtype=attention_mask.dtype, device=attention_mask.device)
    
    # 创建图像部分的 labels
    labels_img = torch.full((B, T_img), pad_ignore_index, dtype=input_ids.dtype, device=input_ids.device)
    
    # 创建文本部分的 labels
    labels_text = input_ids.clone()
    
    # 拼接注意力掩码
    attention_mask_total = torch.cat([attention_mask_img, attention_mask], dim=1)
    
    # 拼接 labels
    labels_total = torch.cat([labels_img, labels_text], dim=1)
    
    return AlignedOutput(
        attention_mask_total=attention_mask_total,
        labels_total=labels_total,
    )
```

### 在训练中的应用

```python
# Step 1：多模态序列对齐
aligned = build_aligned_masks_and_labels(
    input_ids=input_ids,
    attention_mask=attention_mask,
    image_size=config.image_size,
    patch_size=config.patch_size,
    pad_ignore_index=-100,
)
attention_mask_total = aligned.attention_mask_total
labels_total = aligned.labels_total

# 前向传播
logits, past_states, aux_loss = model(
    input_ids=input_ids,
    positions=text_positions,
    pixel_values=pixel_values,
    attention_mask=attention_mask_total,
    use_cache=False,
    output_hidden_states=False
)

# 计算主任务损失
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels_total[:, 1:].contiguous()
```

## 真实数据逐步举例（一步步对齐到模型 forward 所需的 T_total）

这里用 `docs/06_data_pipeline.md` 中同一条真实样本继续往下走，展示：
`(input_ids, attention_mask, pixel_values)` 如何变成模型 forward 需要的
`attention_mask_total / labels_total`（长度变为 `T_total=T_img+T_text`）。

### Step 0：这条真实样本的“已知事实”

来自 `tools/dump_real_sample_trace.py` 的真实输出（节选）：

```json
{
  "image_size": 224,
  "patch_size": 16,
  "T_img": 196,
  "T_text": 512,
  "T_total": 708,
  "num_text_tokens_after_trunc": 175,
  "pad_tokens": 337
}
```

解释：
- `T_img=(224//16)^2=14*14=196`（图像 patch tokens 数）
- `T_text=512`（文本被 padding 到 max_length）
- 这条样本真实文本 token 数是 `175`，所以 padding token 数是 `512-175=337`

### Step 1：attention_mask_total 怎么拼出来？

规则：
- 图像部分：全 1（`[B,T_img]`）
- 文本部分：沿用 DataLoader 给你的 `attention_mask`（`[B,T_text]`）
- 拼接：`attention_mask_total = cat([img_ones, attention_mask_text], dim=1)`

对这条样本（B=1）来说，`attention_mask_total` 的结构就是：

```text
len = 708
[0 : 196)         -> 1（图像 patch tokens）
[196 : 196+175)   -> 1（真实文本 tokens）
[196+175 : 708)   -> 0（文本 padding tokens）
```

### Step 2：labels_total 怎么拼出来？

规则（Step1 的关键点）：
- 图像部分：全 `-100`（不计算 loss）
- 文本部分：用 `input_ids`，但 **padding 位置也必须置为 -100**（不计算 loss）
- 拼接：`labels_total = cat([labels_img, labels_text], dim=1)`

对这条样本（B=1）来说：

```text
len = 708
[0 : 196)         -> -100（图像 patch tokens）
[196 : 196+175)   -> input_ids（真实文本 tokens）
[196+175 : 708)   -> -100（文本 padding tokens）
```

举一个“头部片段”的真实例子（同一条样本的 `input_ids_head_24`）：

```text
labels_total[0:10]          = [-100, -100, ...]  # 图像区
labels_total[196:196+10]    = [13608, 9752, 61705, 1210, 364, 43288, 99639, 86341, 101987, 100169]
labels_total[196+175:196+185] = [-100, -100, ...]  # 文本 padding 区
```

### Step 3：loss 的 shift 为什么不会“污染”图像区？

训练里一般是：
```python
shift_logits = logits[:, :-1, :]
shift_labels = labels_total[:, 1:]
loss = CE(shift_logits, shift_labels, ignore_index=-100)
```

因为图像区的 labels 是 `-100`，shift 之后依然是 `-100`，因此图像 token 对 loss **完全不产生贡献**；
同理，文本 padding 区也因为 labels=-100 而被忽略。

## 注意事项

1. **序列长度计算**：确保 `T_total = T_img + T_text` 与模型输出的 logits 长度一致
2. **设备一致性**：确保所有张量在相同的设备上
3. **数据类型一致性**：确保注意力掩码和标签的数据类型正确
4. **填充处理**：确保文本部分的填充位置标签正确设置为 -100

## 常见问题

1. **损失计算错误**：检查标签是否正确对齐，特别是图像部分的标签是否为 -100
2. **维度不匹配**：检查注意力掩码和 logits 的维度是否一致
3. **位置编码错误**：确保位置编码的构造正确，特别是图像部分的空间位置

## 下一步优化
1. **动态序列长度**：支持不同大小的图像和文本输入
2. **序列打包**：实现序列打包以提高训练效率
3. **跨样本边界掩码**：处理序列打包时的跨样本边界

---

## 快速自检（强烈建议合并后先跑）

如果你想确认“第 1 步对齐”是不是真的生效（尤其是 image 部分 labels 是否全为 -100），运行：

```bash
python test/check_multimodal_alignment.py \
  --data_dir ./data \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --batch_size 2 \
  --max_length 32
```

预期输出要点：
- `T_img` 打印为 **196**（224/16=14，14^2=196）
- `labels_total` shape 为 `[B, 196 + T_text]`
- `image labels all -100: True`
