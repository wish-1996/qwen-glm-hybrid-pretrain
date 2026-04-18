#!/usr/bin/env python3
"""
显存占用分析工具

功能：
- 单卡显存使用情况分析
- 多卡 DDP 显存使用情况分析
- 模型不同组件的显存占用统计
- 前向/反向传播显存变化分析

使用方法：
python tools/mem_profile.py --help
"""

import argparse
import json
import sys
import os
from datetime import datetime

import torch
import torch.nn as nn

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.hybrid_moe_model import HybridMMMoEModel
from configs.model_config import ModelConfig
from data.multimodal_data_loader import get_data_loader
from data.multimodal_sequence_alignment import build_aligned_masks_and_labels
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="显存占用分析工具")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizers/qwen3-0.6b", help="分词器路径")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--distributed", action="store_true", help="是否使用分布式模式")
    parser.add_argument("--profile_forward", action="store_true", help="分析前向传播显存变化")
    parser.add_argument("--profile_backward", action="store_true", help="分析反向传播显存变化")
    parser.add_argument("--output_json", type=str, default="", help="输出 JSON 结果文件")
    return parser.parse_args()


def get_gpu_memory():
    """获取所有 GPU 的内存使用情况"""
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3  # GB
        max_mem = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
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
    """分析模型显存占用"""
    print("=== 显存占用分析 ===")
    print(f"时间: {datetime.utcnow().isoformat()}")
    print(f"参数: {args}")
    
    # 初始化统计
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "args": vars(args),
        "memory": {},
        "model_info": {}
    }
    
    # 初始内存
    torch.cuda.empty_cache()
    initial_mem = get_gpu_memory()
    stats["memory"]["initial"] = initial_mem
    print("\n初始内存:")
    print_memory_info(initial_mem)
    
    # 加载分词器
    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 加载数据
    print("加载数据...")
    train_loader = get_data_loader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        image_size=args.image_size,
        num_workers=0,
        pin_memory=False,
        distributed=args.distributed,
        rank=0,
        world_size=1 if not args.distributed else 2,
        seed=42,
    )
    
    # 获取一个 batch
    batch = next(iter(train_loader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    pixel_values = batch['pixel_values']
    
    print(f"\nBatch 信息:")
    print(f"  input_ids: {input_ids.shape}, dtype={input_ids.dtype}")
    print(f"  attention_mask: {attention_mask.shape}, dtype={attention_mask.dtype}")
    print(f"  pixel_values: {pixel_values.shape}, dtype={pixel_values.dtype}")
    
    # 构建对齐后的张量
    aligned = build_aligned_masks_and_labels(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_size=args.image_size,
        patch_size=16,
        pad_ignore_index=-100,
        image_pad_token_id=None,
        build_positions=True,
    )
    input_ids_total = aligned.input_ids_total
    attention_mask_total = aligned.attention_mask_total
    labels_total = aligned.labels_total
    positions_total = aligned.positions_total
    
    print(f"\n对齐后张量:")
    print(f"  input_ids_total: {input_ids_total.shape}")
    print(f"  attention_mask_total: {attention_mask_total.shape}")
    print(f"  labels_total: {labels_total.shape}")
    print(f"  positions_total: {positions_total.shape}")
    
    # 初始化模型
    print("\n初始化模型...")
    config = ModelConfig()
    model = HybridMMMoEModel(config, use_multimodal=True)
    
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats["model_info"]["total_params"] = total_params
    stats["model_info"]["trainable_params"] = trainable_params
    
    print(f"模型参数量:")
    print(f"  总参数: {total_params / 1e9:.2f}B")
    print(f"  可训练参数: {trainable_params / 1e9:.2f}B")
    
    # 移至 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids_total = input_ids_total.to(device)
    attention_mask_total = attention_mask_total.to(device)
    labels_total = labels_total.to(device)
    positions_total = positions_total.to(device)
    pixel_values = pixel_values.to(device)
    
    # 模型加载后的内存
    after_model_mem = get_gpu_memory()
    stats["memory"]["after_model_loaded"] = after_model_mem
    print("\n模型加载后内存:")
    print_memory_info(after_model_mem)
    
    # 前向传播分析
    if args.profile_forward:
        print("\n=== 前向传播分析 ===")
        torch.cuda.empty_cache()
        pre_forward_mem = get_gpu_memory()
        
        with torch.no_grad():
            logits, past_states, aux_loss = model(
                input_ids=input_ids_total,
                positions=positions_total,
                pixel_values=pixel_values,
                attention_mask=attention_mask_total,
                use_cache=False,
                output_hidden_states=False,
            )
        
        post_forward_mem = get_gpu_memory()
        stats["memory"]["pre_forward"] = pre_forward_mem
        stats["memory"]["post_forward"] = post_forward_mem
        
        print("前向传播前内存:")
        print_memory_info(pre_forward_mem)
        print("前向传播后内存:")
        print_memory_info(post_forward_mem)
        
        # 计算前向传播内存增加
        for gpu_id in pre_forward_mem:
            delta = post_forward_mem[gpu_id]["allocated"] - pre_forward_mem[gpu_id]["allocated"]
            print(f"GPU {gpu_id} 前向传播内存增加: {delta:.2f} GB")
    
    # 反向传播分析
    if args.profile_backward:
        print("\n=== 反向传播分析 ===")
        torch.cuda.empty_cache()
        
        # 前向传播
        logits, past_states, aux_loss = model(
            input_ids=input_ids_total,
            positions=positions_total,
            pixel_values=pixel_values,
            attention_mask=attention_mask_total,
            use_cache=False,
            output_hidden_states=False,
        )
        
        # 计算损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels_total[:, 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        pre_backward_mem = get_gpu_memory()
        
        # 反向传播
        loss.backward()
        
        post_backward_mem = get_gpu_memory()
        stats["memory"]["pre_backward"] = pre_backward_mem
        stats["memory"]["post_backward"] = post_backward_mem
        
        print("反向传播前内存:")
        print_memory_info(pre_backward_mem)
        print("反向传播后内存:")
        print_memory_info(post_backward_mem)
        
        # 计算反向传播内存增加
        for gpu_id in pre_backward_mem:
            delta = post_backward_mem[gpu_id]["allocated"] - pre_backward_mem[gpu_id]["allocated"]
            print(f"GPU {gpu_id} 反向传播内存增加: {delta:.2f} GB")
    
    # 清理
    del model
    del input_ids_total, attention_mask_total, labels_total, positions_total, pixel_values
    torch.cuda.empty_cache()
    
    final_mem = get_gpu_memory()
    stats["memory"]["final"] = final_mem
    print("\n清理后内存:")
    print_memory_info(final_mem)
    
    # 输出 JSON
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
