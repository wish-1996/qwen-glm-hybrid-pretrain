"""
参数量估算脚本

用于对齐目标模型规模：
- 总参数量：7B (用于 MoE total params)
- 激活参数量：0.6B (用于 MoE active params)

使用方法：
    python tools/param_count.py
    python tools/param_count.py --config configs/qwen35_config.py
    python tools/param_count.py --target_total 7e9 --target_active 0.6e9
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


@dataclass
class ParamCountResult:
    total_params: int
    active_params: int
    embedding_params: int
    output_params: int
    attention_params_per_layer: int
    total_attention_params: int
    moe_params_per_expert: int
    total_moe_params: int
    gate_params: int


def calculate_theoretical_parameters(
    hidden_size: int = 2048,
    num_layers: int = 28,
    num_attention_heads: int = 16,
    num_kv_heads: int = 4,
    num_experts: int = 192,
    top_k: int = 4,
    intermediate_size: int = 5734,
    vocab_size: int = 151936,
) -> ParamCountResult:
    """
    理论计算模型参数量

    Args:
        hidden_size: 隐藏层维度
        num_layers: 模型层数
        num_attention_heads: Q 的注意力头数
        num_kv_heads: KV 的注意力头数 (GQA)
        num_experts: MoE 专家数量
        top_k: 每个 token 激活的专家数量
        intermediate_size: FFN 中间层维度
        vocab_size: 词表大小

    Returns:
        ParamCountResult: 参数量统计结果
    """
    embedding_params = vocab_size * hidden_size

    output_params = hidden_size * vocab_size

    head_dim = hidden_size // num_attention_heads

    q_params_per_layer = hidden_size * hidden_size
    k_params_per_layer = hidden_size * (head_dim * num_kv_heads)
    v_params_per_layer = hidden_size * (head_dim * num_kv_heads)
    o_params_per_layer = (head_dim * num_kv_heads) * hidden_size
    attention_params_per_layer = q_params_per_layer + k_params_per_layer + v_params_per_layer + o_params_per_layer
    total_attention_params = attention_params_per_layer * num_layers

    moe_params_per_expert = 2 * hidden_size * intermediate_size
    total_moe_params = moe_params_per_expert * num_experts

    gate_params = hidden_size * num_experts

    total_params = (
        embedding_params
        + output_params
        + total_attention_params
        + total_moe_params
        + gate_params
    )

    active_params = (
        total_attention_params
        + (moe_params_per_expert * top_k)
    )

    return ParamCountResult(
        total_params=total_params,
        active_params=active_params,
        embedding_params=embedding_params,
        output_params=output_params,
        attention_params_per_layer=attention_params_per_layer,
        total_attention_params=total_attention_params,
        moe_params_per_expert=moe_params_per_expert,
        total_moe_params=total_moe_params,
        gate_params=gate_params,
    )


def print_param_report(
    result: ParamCountResult,
    target_total: Optional[float] = None,
    target_active: Optional[float] = None,
) -> None:
    """打印参数量报告"""
    print("\n" + "=" * 60)
    print("模型参数量统计报告")
    print("=" * 60)

    print(f"\n[参数量汇总]")
    print(f"  总参数量 (Total):     {result.total_params / 1e9:.4f} B ({result.total_params:,})")
    print(f"  激活参数量 (Active):  {result.active_params / 1e9:.4f} B ({result.active_params:,})")

    if target_total is not None:
        diff = result.total_params - target_total
        status = "✓" if abs(diff) < target_total * 0.01 else "✗"
        print(f"  目标总参数量:         {target_total / 1e9:.2f} B ({status})")
        print(f"  差距:                 {diff / 1e9:.4f} B ({diff / target_total * 100:+.2f}%)")

    if target_active is not None:
        diff = result.active_params - target_active
        status = "✓" if abs(diff) < target_active * 0.01 else "✗"
        print(f"  目标激活参数量:       {target_active / 1e9:.2f} B ({status})")
        print(f"  差距:                 {diff / 1e9:.4f} B ({diff / target_active * 100:+.2f}%)")

    print(f"\n[详细分解]")
    print(f"  Embedding 层:         {result.embedding_params / 1e9:.4f} B")
    print(f"  Output 层:            {result.output_params / 1e9:.4f} B")
    print(f"  单层 Attention:       {result.attention_params_per_layer / 1e6:.2f} M")
    print(f"  总 Attention:         {result.total_attention_params / 1e9:.4f} B")
    print(f"  单个专家 FFN:         {result.moe_params_per_expert / 1e6:.2f} M")
    print(f"  总 MoE 参数量:        {result.total_moe_params / 1e9:.4f} B ({192} experts)")
    print(f"  Gate 参数量:         {result.gate_params / 1e6:.2f} M")

    print(f"\n[比例分析]")
    print(f"  MoE 占比 (总):       {result.total_moe_params / result.total_params * 100:.2f}%")
    print(f"  MoE 占比 (激活):     {result.moe_params_per_expert * 4 / result.active_params * 100:.2f}%")
    print(f"  Attention 占比:      {result.total_attention_params / result.total_params * 100:.2f}%")


def suggest_adjustments(
    result: ParamCountResult,
    target_total: float,
    target_active: float,
) -> None:
    """给出参数调整建议"""
    print(f"\n[参数调整建议]")

    if result.total_params < target_total:
        gap = target_total - result.total_params
        print(f"  总参数量偏低 {gap / 1e9:.2f}B，建议:")
        print(f"    - 增加 num_experts (当前 {192})")
        print(f"    - 或增加 intermediate_size (当前 5734)")
    elif result.total_params > target_total:
        gap = result.total_params - target_total
        print(f"  总参数量偏高 {gap / 1e9:.2f}B，建议:")
        print(f"    - 减少 num_experts (当前 {192})")
        print(f"    - 或减少 intermediate_size (当前 5734)")

    if result.active_params < target_active:
        gap = target_active - result.active_params
        print(f"\n  激活参数量偏低 {gap / 1e9:.2f}B，建议:")
        print(f"    - 增加 top_k (当前 4)")
        print(f"    - 或增加 intermediate_size (当前 5734)")
    elif result.active_params > target_active:
        gap = result.active_params - target_active
        print(f"\n  激活参数量偏高 {gap / 1e9:.2f}B，建议:")
        print(f"    - 减少 top_k (当前 4)")
        print(f"    - 或减少 intermediate_size (当前 5734)")


def main():
    parser = argparse.ArgumentParser(
        description="模型参数量估算工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/param_count.py
  python tools/param_count.py --target_total 7e9 --target_active 0.6e9
  python tools/param_count.py --num_experts 256 --top_k 6
        """
    )

    parser.add_argument(
        "--hidden_size", type=int, default=2048,
        help="隐藏层维度 (default: 2048)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=28,
        help="模型层数 (default: 28)"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=16,
        help="注意力头数 (default: 16)"
    )
    parser.add_argument(
        "--num_kv_heads", type=int, default=4,
        help="KV 头数，用于 GQA (default: 4)"
    )
    parser.add_argument(
        "--num_experts", type=int, default=192,
        help="MoE 专家数量 (default: 192)"
    )
    parser.add_argument(
        "--top_k", type=int, default=4,
        help="每个 token 激活的专家数 (default: 4)"
    )
    parser.add_argument(
        "--intermediate_size", type=int, default=5734,
        help="FFN 中间层维度 (default: 5734)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=151936,
        help="词表大小 (default: 151936)"
    )
    parser.add_argument(
        "--target_total", type=float, default=7e9,
        help="目标总参数量 (default: 7e9)"
    )
    parser.add_argument(
        "--target_active", type=float, default=0.6e9,
        help="目标激活参数量 (default: 0.6e9)"
    )

    args = parser.parse_args()

    result = calculate_theoretical_parameters(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
    )

    print_param_report(result, args.target_total, args.target_active)
    suggest_adjustments(result, args.target_total, args.target_active)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
