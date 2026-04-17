"""
多模态序列对齐（Step 1）

核心目标：当模型将 image embeddings 与 text embeddings 拼接时，
确保 labels 和 attention_mask 也扩展到相同长度，并且 image 部分的 labels 全为 -100。

输入：
- input_ids: [B, T_text]
- attention_mask: [B, T_text]
- image_size: 图像尺寸（如 224）
- patch_size: 补丁尺寸（如 16）
- pad_ignore_index: 填充忽略索引（默认为 -100）

输出：
- attention_mask_total: [B, T_img + T_text]
- labels_total: [B, T_img + T_text]
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class AlignedOutput:
    """
    对齐后的输出
    """
    attention_mask_total: torch.Tensor
    labels_total: torch.Tensor


def build_aligned_masks_and_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_size: int,
    patch_size: int,
    pad_ignore_index: int = -100,
) -> AlignedOutput:
    """
    构建对齐的 masks 和 labels
    
    Args:
        input_ids: 输入的 token IDs，形状为 [B, T_text]
        attention_mask: 注意力掩码，形状为 [B, T_text]
        image_size: 图像尺寸
        patch_size: 补丁尺寸
        pad_ignore_index: 填充忽略索引
    
    Returns:
        AlignedOutput: 包含对齐后的 attention_mask_total 和 labels_total
    """
    B, T_text = input_ids.shape
    
    # 计算图像补丁数量
    num_patches = (image_size // patch_size) ** 2
    T_img = num_patches
    
    # 创建图像部分的注意力掩码（全 1，因为图像补丁都是有效的）
    attention_mask_img = torch.ones(B, T_img, dtype=attention_mask.dtype, device=attention_mask.device)
    
    # 创建图像部分的 labels（全 -100，因为图像部分不需要计算损失）
    labels_img = torch.full((B, T_img), pad_ignore_index, dtype=input_ids.dtype, device=input_ids.device)
    
    # 创建文本部分的 labels（使用 input_ids，因为我们要预测下一个 token）
    labels_text = input_ids.clone()
    # 关键：文本 padding 位置不应该参与 loss
    # - attention_mask == 0 的位置是 padding token
    # - labels 设为 -100（ignore_index）即可在 CrossEntropyLoss 中被忽略
    labels_text = labels_text.masked_fill(attention_mask == 0, pad_ignore_index)
    
    # 拼接注意力掩码
    attention_mask_total = torch.cat([attention_mask_img, attention_mask], dim=1)
    
    # 拼接 labels
    labels_total = torch.cat([labels_img, labels_text], dim=1)
    
    return AlignedOutput(
        attention_mask_total=attention_mask_total,
        labels_total=labels_total,
    )
