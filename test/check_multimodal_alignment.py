"""
Step 1 对齐检查脚本（可单独运行）

目标：验证“多模态拼接后”的 labels 与 attention_mask 是否与模型输出序列长度一致：
- tokens_total = [image_tokens] + [text_tokens]
- labels_total[:T_img] 必须全为 -100（图像 token 不参与 LM loss）

运行示例：
  python test/check_multimodal_alignment.py ^
    --data_dir ./data ^
    --tokenizer_path ./tokenizers/qwen3-0.6b ^
    --batch_size 2 ^
    --max_length 32
"""

import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
from transformers import AutoTokenizer

from configs.model_config import ModelConfig
from data.multimodal_data_loader import get_data_loader
from data.multimodal_sequence_alignment import build_aligned_masks_and_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizers/qwen3-0.6b")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    loader = get_data_loader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        image_size=cfg.image_size,
        num_workers=0,
        pin_memory=False,
        distributed=False,
        seed=42,
    )

    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)

    print("=== Raw batch ===")
    print("input_ids:", tuple(input_ids.shape), input_ids.dtype)
    print("attention_mask:", tuple(attention_mask.shape), attention_mask.dtype)
    print("pixel_values:", tuple(pixel_values.shape), pixel_values.dtype)

    # 由配置推导 T_img（224/16=14 => 196）
    t_img = (cfg.image_size // cfg.patch_size) ** 2
    print("\n=== Config-derived image tokens ===")
    print("image_size:", cfg.image_size, "patch_size:", cfg.patch_size, "=> T_img:", t_img)

    aligned = build_aligned_masks_and_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        pad_ignore_index=-100,
    )

    am_total = aligned.attention_mask_total
    labels_total = aligned.labels_total

    print("\n=== Aligned tensors ===")
    print("attention_mask_total:", tuple(am_total.shape), am_total.dtype)
    print("labels_total:", tuple(labels_total.shape), labels_total.dtype)

    bsz, t_text = input_ids.shape
    assert am_total.shape == (bsz, t_img + t_text), "attention_mask_total shape mismatch"
    assert labels_total.shape == (bsz, t_img + t_text), "labels_total shape mismatch"

    img_region = labels_total[:, :t_img]
    img_ok = (img_region == -100).all().item()
    print("\n=== Checks ===")
    print("image labels all -100:", bool(img_ok))
    if not img_ok:
        print("image labels sample:", img_region[0, :16].tolist())
        raise SystemExit("FAIL: image labels are not all -100")

    # 文本 pad 位置应为 -100（数量与 attention_mask==0 对齐）
    text_region = labels_total[:, t_img:]
    pad_count = int((attention_mask == 0).sum().item())
    neg_count = int((text_region == -100).sum().item())
    print("text pad tokens:", pad_count, "| text labels == -100:", neg_count)
    print("\nPASS: multimodal alignment looks correct.")


if __name__ == "__main__":
    main()
