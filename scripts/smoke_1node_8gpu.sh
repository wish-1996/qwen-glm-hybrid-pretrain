#!/bin/bash
set -e

# 1-node 8-GPU smoke test（torchrun + DDP）
#
# 用途：
# - 验证 DDP 初始化/通信是否正常
# - 验证 P0-1（MOE_BACKEND=deepspeed）在多卡下能跑通
# - 验证 P0-2（USE_FLASH_ATTN=1）开关在多卡下不影响主路径
#
# 默认只跑少量 optimizer steps（max_steps），用于 CI / 自测 / 快速回归。

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29501}
SMOKE_STEPS=${SMOKE_STEPS:-30}
OUT_BASE=${OUT_BASE:-./checkpoints_smoke}

echo "=============================="
echo "[smoke][8gpu] torchrun"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  MOE_BACKEND=${MOE_BACKEND:-}"
echo "  USE_FLASH_ATTN=${USE_FLASH_ATTN:-}"
echo "  SMOKE_STEPS=${SMOKE_STEPS}"
echo "=============================="

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train/train_multimodal.py \
    --data_dir ./data \
    --tokenizer_path ./tokenizers/qwen3-0.6b \
    --max_length 512 \
    --image_size 224 \
    --batch_size 8 \
    --epochs 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 10 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --distributed \
    --log_interval 1 \
    --save_interval 999999 \
    --save_steps 0 \
    --max_steps "${SMOKE_STEPS}" \
    --output_dir "${OUT_BASE}/ddp_8gpu"
