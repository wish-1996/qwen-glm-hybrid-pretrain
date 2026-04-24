#!/usr/bin/env python3
"""
训练计划估算工具

功能：
- 统计数据量（CSV 图文、JSONL 文本、Parquet 文本）
- 用 tokenizer 抽样估算平均有效 tokens（attention_mask.sum()，padding 不计入 loss）
- 根据 world_size / batch_size / grad_accum 计算 tokens/optimizer_step 与需要的总 steps
- 读取 metrics_rank0.jsonl 里的 tokens_per_sec，估算训练时长

使用方法：
python tools/estimate_training_plan.py --help
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import torch
from transformers import AutoTokenizer

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="训练计划估算工具")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizers/qwen3-0.6b", help="分词器路径")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--parquet_glob", type=str, default="", help="Parquet 文件路径模式")
    parser.add_argument("--metrics_jsonl", type=str, default="", help="训练日志文件路径")
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--world_size", type=int, default=8, help="并行度")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--target_tokens", type=float, default=3e11, help="目标训练 tokens 数")
    parser.add_argument("--output_json", type=str, default="", help="输出 JSON 结果文件")
    parser.add_argument("--sample_size", type=int, default=1000, help="抽样数量")
    return parser.parse_args()


def count_csv_data(data_dir: str) -> Dict[str, Any]:
    """统计 CSV 图文数据"""
    csv_path = os.path.join(data_dir, "mm_pairs.csv")
    if not os.path.exists(csv_path):
        return {"csv_rows": 0, "csv_image_cache_hits": 0}
    
    df = pd.read_csv(csv_path)
    rows = len(df)
    
    # 统计 image_cache 命中
    image_cache_dir = os.path.join(data_dir, "image_cache")
    if os.path.exists(image_cache_dir):
        image_files = set(f for f in os.listdir(image_cache_dir) if f.endswith(".jpg") or f.endswith(".png"))
        image_cache_hits = sum(1 for _, row in df.iterrows() if row.get("image") in image_files)
    else:
        image_cache_hits = 0
    
    return {"csv_rows": rows, "csv_image_cache_hits": image_cache_hits}


def count_jsonl_data(data_dir: str) -> Dict[str, Any]:
    """统计 JSONL 文本数据"""
    jsonl_path = os.path.join(data_dir, "text_only.jsonl")
    if not os.path.exists(jsonl_path):
        return {"jsonl_rows": 0}
    
    rows = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            rows += 1
    
    return {"jsonl_rows": rows}


def count_parquet_data(parquet_glob: str) -> Dict[str, Any]:
    """统计 Parquet 文本数据"""
    if not parquet_glob:
        return {"parquet_rows": 0}
    
    import glob
    parquet_files = glob.glob(parquet_glob)
    if not parquet_files:
        return {"parquet_rows": 0}
    
    rows = 0
    for file in parquet_files:
        df = pd.read_parquet(file)
        rows += len(df)
    
    return {"parquet_rows": rows}


def estimate_avg_tokens(tokenizer, data_dir: str, max_length: int, sample_size: int) -> float:
    """估算平均有效 tokens 数"""
    # 读取 CSV 数据
    csv_path = os.path.join(data_dir, "mm_pairs.csv")
    jsonl_path = os.path.join(data_dir, "text_only.jsonl")
    
    samples = []
    
    # 从 CSV 中采样
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        sample_size_csv = min(sample_size // 2, len(df))
        sample_df = df.sample(sample_size_csv, random_state=42)
        for _, row in sample_df.iterrows():
            text = row.get("text", "")
            if text:
                samples.append(text)
    
    # 从 JSONL 中采样
    if os.path.exists(jsonl_path):
        jsonl_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        jsonl_samples.append(text)
                except:
                    pass
        
        sample_size_jsonl = min(sample_size // 2, len(jsonl_samples))
        samples.extend(random.sample(jsonl_samples, sample_size_jsonl))
    
    if not samples:
        return max_length * 0.5  # 默认估算
    
    # 计算平均有效 tokens
    total_tokens = 0
    for text in samples:
        tokens = tokenizer(text, truncation=True, max_length=max_length)
        attention_mask = tokens.get("attention_mask", [])
        total_tokens += sum(attention_mask)
    
    return total_tokens / len(samples)


def estimate_tokens_per_step(world_size: int, batch_size: int, grad_accum: int, avg_tokens: float) -> float:
    """计算每步训练的 tokens 数"""
    return world_size * batch_size * grad_accum * avg_tokens


def estimate_total_steps(target_tokens: float, tokens_per_step: float) -> int:
    """计算总训练步数"""
    return int(target_tokens / tokens_per_step)


def estimate_training_time(metrics_jsonl: str, total_steps: int, world_size: int, batch_size: int, grad_accum: int, avg_tokens: float) -> Dict[str, Any]:
    """根据日志估算训练时长"""
    if not os.path.exists(metrics_jsonl):
        return {"estimated_hours": 0, "tokens_per_sec": 0}
    
    tokens_per_sec_list = []
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("event") == "train_step":
                    tps = data.get("tokens_per_sec", 0)
                    if tps > 0:
                        tokens_per_sec_list.append(tps)
            except:
                pass
    
    if not tokens_per_sec_list:
        return {"estimated_hours": 0, "tokens_per_sec": 0}
    
    avg_tokens_per_sec = sum(tokens_per_sec_list) / len(tokens_per_sec_list)
    estimated_seconds = total_steps / (avg_tokens_per_sec / (world_size * batch_size * grad_accum * avg_tokens))
    estimated_hours = estimated_seconds / 3600
    
    return {"estimated_hours": estimated_hours, "tokens_per_sec": avg_tokens_per_sec}


def main():
    args = parse_args()
    
    print("=== 训练计划估算 ===")
    print(f"时间: {datetime.utcnow().isoformat()}")
    print(f"参数: {args}")
    
    # 加载分词器
    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 统计数据量
    print("\n统计数据量...")
    csv_stats = count_csv_data(args.data_dir)
    jsonl_stats = count_jsonl_data(args.data_dir)
    parquet_stats = count_parquet_data(args.parquet_glob)
    
    print(f"CSV 数据: {csv_stats['csv_rows']} 行, 图像缓存命中: {csv_stats['csv_image_cache_hits']}")
    print(f"JSONL 数据: {jsonl_stats['jsonl_rows']} 行")
    print(f"Parquet 数据: {parquet_stats['parquet_rows']} 行")
    
    # 估算平均有效 tokens
    print("\n估算平均有效 tokens...")
    avg_tokens = estimate_avg_tokens(tokenizer, args.data_dir, args.max_length, args.sample_size)
    print(f"平均有效 tokens: {avg_tokens:.2f}")
    
    # 计算每步训练的 tokens 数
    tokens_per_step = estimate_tokens_per_step(args.world_size, args.batch_size, args.grad_accum, avg_tokens)
    print(f"每步训练 tokens: {tokens_per_step:.2f}")
    
    # 计算总训练步数
    total_steps = estimate_total_steps(args.target_tokens, tokens_per_step)
    print(f"总训练步数: {total_steps:,}")
    
    # 估算训练时长
    time_stats = estimate_training_time(args.metrics_jsonl, total_steps, args.world_size, args.batch_size, args.grad_accum, avg_tokens)
    if time_stats["tokens_per_sec"] > 0:
        print(f"估算训练时长: {time_stats['estimated_hours']:.2f} 小时")
        print(f"估算吞吐: {time_stats['tokens_per_sec']:.2f} tokens/sec")
    else:
        print("无法估算训练时长：未找到有效的训练日志")
    
    # 生成结果
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "args": vars(args),
        "data_stats": {
            **csv_stats,
            **jsonl_stats,
            **parquet_stats
        },
        "estimation": {
            "avg_tokens": avg_tokens,
            "tokens_per_step": tokens_per_step,
            "total_steps": total_steps,
            **time_stats
        }
    }
    
    # 输出 JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n分析结果已保存到: {args.output_json}")
    
    return results


if __name__ == "__main__":
    main()