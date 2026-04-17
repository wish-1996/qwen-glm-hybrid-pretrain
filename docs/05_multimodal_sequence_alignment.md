# 05 多模态序列对齐（Step 1）

## 核心概念

多模态序列对齐是多模态预训练中的关键步骤，确保模型能够正确处理图像和文本的组合输入。

### 序列长度

- **文本序列长度**：`T_text`，由输入的 token 数量决定
- **图像序列长度**：`T_img`，由图像补丁数量决定，计算公式为 `(image_size // patch_size)²`
- **总序列长度**：`T_total = T_img + T_text`，模型输出的 logits 长度

### 对齐目标

1. **注意力掩码对齐**：确保图像部分的注意力掩码为 1（有效），文本部分保持原掩码
2. **标签对齐**：确保图像部分的标签为 -100（忽略），文本部分保持原标签
3. **位置编码对齐**：为图像和文本部分创建相应的位置编码

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
