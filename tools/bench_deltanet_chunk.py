"""
简单基准：比较 GatedDeltaNet（训练分支）在长序列下的吞吐。

用法示例：
  python tools/bench_deltanet_chunk.py --seq 4096 --chunk 256

说明：
- 该脚本只做 forward（不做 backward），用于快速验证 chunk-wise 是否避免 Python token 循环瓶颈。
- 真实训练吞吐还受 data loader、MoE、backward 等影响，但这里能快速看出"线性注意力训练分支"的量级变化。
"""

from __future__ import annotations

import argparse
import time

import torch

from configs.model_config import ModelConfig
from model.hybrid_moe_model import GatedDeltaNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    cfg = ModelConfig()
    cfg.deltanet_chunk_size = int(args.chunk)

    attn = GatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=dtype)
    attn.eval()

    B = int(args.batch)
    N = int(args.seq)
    H = int(cfg.hidden_size)

    x = torch.randn(B, N, H, device=device, dtype=dtype)
    # positions: [t,0,0]
    t = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    positions = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)

    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = attn(x, positions, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(int(args.iters)):
            _ = attn(x, positions, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    sec = (t1 - t0) / max(1, int(args.iters))
    tok_s = (B * N) / max(1e-9, sec)
    print(f"seq={N} batch={B} chunk={args.chunk} dtype={args.dtype} time/iter={sec:.4f}s tok/s={tok_s:.1f}")


if __name__ == "__main__":
    main()
