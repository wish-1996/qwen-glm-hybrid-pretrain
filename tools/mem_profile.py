#!/usr/bin/env python3
"""
显存占用分析工具

功能：
- 基于配置的模型显存估算
- 单卡/多卡显存使用情况分析
- 模型不同组件的显存占用统计

使用方法：
python tools/mem_profile.py [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH] [--image_size IMAGE_SIZE] [--output_json OUTPUT_JSON]

示例：
python tools/mem_profile.py
python tools/mem_profile.py --batch_size 2 --max_length 1024 --image_size 448
python tools/mem_profile.py --batch_size 4 --max_length 2048 --output_json results.json
"""

import argparse
import json
import sys
import os
from datetime import datetime

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="显存占用分析工具")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--output_json", type=str, default="", help="输出 JSON 结果文件")
    parser.add_argument("--config_preset", type=str, default="default", choices=["default", "local", "prod7b"],
                        help="使用哪套模型配置：default(仓库默认) / local(本地小模型) / prod7b(7B 目标配置)")
    return parser.parse_args()


def get_gpu_memory():
    """获取所有 GPU 的内存使用情况"""
    if not torch.cuda.is_available():
        return {}

    memory_info = {}
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        max_mem = torch.cuda.max_memory_allocated(i) / 1024**3
        memory_info[i] = {
            "allocated": mem,
            "max_allocated": max_mem,
            "total": torch.cuda.get_device_properties(i).total_memory / 1024**3
        }
    return memory_info


def print_memory_info(info, prefix=""):
    """打印内存信息"""
    for gpu_id, mem in info.items():
        print(f"{prefix}GPU {gpu_id}:")
        print(f"{prefix}  已分配: {mem['allocated']:.2f} GB")
        print(f"{prefix}  最大分配: {mem['max_allocated']:.2f} GB")
        print(f"{prefix}  总内存: {mem['total']:.2f} GB")
        print(f"{prefix}  使用率: {mem['allocated'] / mem['total'] * 100:.1f}%")


def profile_model_memory(args):
    """基于配置估算模型显存占用"""
    print("=== 显存占用估算 ===")
    print(f"时间: {datetime.utcnow().isoformat()}")
    print(f"参数: {args}")

    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "args": vars(args),
        "memory": {},
        "model_info": {}
    }

    torch.cuda.empty_cache()
    initial_mem = get_gpu_memory()
    stats["memory"]["initial"] = initial_mem
    print("\n初始内存:")
    print_memory_info(initial_mem)

    if args.config_preset == "local":
        from configs.model_config_local import LocalModelConfig
        config = LocalModelConfig()
        print("[config] Using LocalModelConfig")
    elif args.config_preset == "prod7b":
        from configs.model_config_prod_7b import Prod7BModelConfig
        config = Prod7BModelConfig()
        print("[config] Using Prod7BModelConfig")
    else:
        config = ModelConfig()
        print("[config] Using default ModelConfig")

    embed_params = config.vocab_size * config.hidden_size
    layer_params = 0
    attn_params = config.hidden_size * config.hidden_size * 3
    # MoE 参数（与当前仓库实现对齐）
    # - experts 是 SwiGLU：gate_proj/up_proj/down_proj -> 3 * H * FF
    # - per-layer experts：每一层都有一套 experts（不是跨层共享）
    # - shared expert：每层额外 1 个 shared expert
    moe_params_per_expert = 3 * config.hidden_size * config.intermediate_size
    moe_params_per_layer = moe_params_per_expert * (config.num_experts + 1)  # sparse experts + shared expert
    layer_params = attn_params + moe_params_per_layer
    layers_params = layer_params * config.num_layers
    lm_head_params = config.hidden_size * config.vocab_size

    vision_params = 0
    vision_params += 3 * config.hidden_size * (config.patch_size ** 2)
    num_patches = (config.image_size // config.patch_size) ** 2
    vision_params += num_patches * config.hidden_size
    for _ in range(4):
        vision_params += config.hidden_size * config.hidden_size * 3
        vision_params += config.hidden_size * config.intermediate_size * 2

    total_params = embed_params + layers_params + lm_head_params + vision_params
    trainable_params = total_params

    stats["model_info"]["total_params"] = total_params
    stats["model_info"]["trainable_params"] = trainable_params

    print(f"\n模型参数量:")
    print(f"  总参数: {total_params / 1e9:.2f}B")
    print(f"  可训练参数: {trainable_params / 1e9:.2f}B")

    param_mem_gb = total_params * 2 / (1024**3)
    batch_size = args.batch_size
    text_seq_len = args.max_length
    num_patches = (args.image_size // config.patch_size) ** 2
    seq_len = text_seq_len + num_patches
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    input_embed_mem_gb = batch_size * seq_len * hidden_size * 2 / (1024**3)
    output_logits_mem_gb = batch_size * seq_len * vocab_size * 2 / (1024**3)
    activation_mem_gb = batch_size * seq_len * hidden_size * 4 / (1024**3)
    backward_mem_gb = (input_embed_mem_gb + output_logits_mem_gb + activation_mem_gb) * 2.5

    total_estimated_mem_gb = param_mem_gb + input_embed_mem_gb + output_logits_mem_gb + activation_mem_gb + backward_mem_gb

    print(f"\n估算显存占用:")
    print(f"  模型参数: {param_mem_gb:.2f} GB")
    print(f"  输入嵌入: {input_embed_mem_gb:.2f} GB")
    print(f"  输出 logits: {output_logits_mem_gb:.2f} GB")
    print(f"  中间激活: {activation_mem_gb:.2f} GB")
    print(f"  反向传播: {backward_mem_gb:.2f} GB")
    print(f"  总估算: {total_estimated_mem_gb:.2f} GB")

    stats["model_info"]["estimated_memory"] = {
        "param_mem_gb": param_mem_gb,
        "input_embed_mem_gb": input_embed_mem_gb,
        "output_logits_mem_gb": output_logits_mem_gb,
        "activation_mem_gb": activation_mem_gb,
        "backward_mem_gb": backward_mem_gb,
        "total_estimated_mem_gb": total_estimated_mem_gb
    }

    for gpu_id, mem in initial_mem.items():
        gpu_total = mem["total"]
        if total_estimated_mem_gb > gpu_total * 0.9:
            print(f"\n警告: GPU {gpu_id} 可能会 OOM！")
            print(f"  GPU 总内存: {gpu_total:.2f} GB")
            print(f"  估算需要: {total_estimated_mem_gb:.2f} GB")
        else:
            print(f"\nGPU {gpu_id} 内存充足")
            print(f"  GPU 总内存: {gpu_total:.2f} GB")
            print(f"  估算需要: {total_estimated_mem_gb:.2f} GB")
            print(f"  剩余空间: {gpu_total - total_estimated_mem_gb:.2f} GB")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n分析结果已保存到: {args.output_json}")

    return stats


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("错误: 没有可用的 GPU，无法进行显存分析")
        sys.exit(1)

    try:
        profile_model_memory(args)
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
