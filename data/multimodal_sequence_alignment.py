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
    # 新增：显式把“图片占位符 tokens”拼进 input_ids（更像生产）
    # - shape: [B, T_total]
    input_ids_total: torch.Tensor
    attention_mask_total: torch.Tensor
    labels_total: torch.Tensor
    # 可选：对齐后的 3D positions（如果你希望不在训练脚本里拼 positions）
    positions_total: torch.Tensor | None = None
    # 方便调试/打印
    t_img: int = 0


def build_aligned_masks_and_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_size: int,
    patch_size: int,
    pad_ignore_index: int = -100,
    *,
    image_pad_token_id: int | None = None,
    build_positions: bool = False,
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

    # ----------------------------
    # (可选) 把“图片占位符 tokens”拼进 input_ids
    # ----------------------------
    # 生产多模态通常会在 token 序列里显式放一个 <image> / <image_pad> 占位符，
    # 而不是只靠 pixel_values 这条“旁路”输入。
    #
    # 我们这里采用“patch 级占位”：
    #   input_ids_total = [<image_pad> * T_img] + [text tokens]
    #
    # 然后在模型 forward 里用 vision_encoder(image) 的输出去替换这段占位符 embedding，
    # 达到：token 序列上可见、多模态融合逻辑仍保持 early-fusion。
    if image_pad_token_id is None:
        # 保持兼容：不提供 image_pad_token_id 时，仍然返回原始 input_ids（仅用于老流程/对比）
        input_ids_img = None
        input_ids_total = input_ids
    else:
        input_ids_img = torch.full(
            (B, T_img),
            int(image_pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids_total = torch.cat([input_ids_img, input_ids], dim=1)
    
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

    positions_total = None
    if build_positions:
        # 3D positions:
        # - image: [0, h, w]  (h,w in patch grid)
        # - text : [t, 0, 0]  (t=0..T_text-1)
        #
        # 注意：这里我们默认图片放在序列开头，因此 positions_total 也是 image 在前、text 在后。
        grid = image_size // patch_size  # 例如 224//16=14
        image_positions = torch.zeros(B, T_img, 3, dtype=torch.long, device=input_ids.device)
        # 为每个 patch 分配 (0,h,w)
        idx = 0
        for h in range(grid):
            for w in range(grid):
                image_positions[:, idx, 0] = 0
                image_positions[:, idx, 1] = h
                image_positions[:, idx, 2] = w
                idx += 1

        t = torch.arange(T_text, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        text_positions = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)  # [B,T_text,3]
        positions_total = torch.cat([image_positions, text_positions], dim=1)
    
    return AlignedOutput(
        input_ids_total=input_ids_total,
        attention_mask_total=attention_mask_total,
        labels_total=labels_total,
        positions_total=positions_total,
        t_img=T_img,
    )
