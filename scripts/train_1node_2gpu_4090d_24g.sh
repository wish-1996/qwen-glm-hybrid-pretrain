#!/bin/bash
set -e

# 1-node 2-GPU（2x4090D 24G）训练启动脚本
#
# 目标：
# - 在单机 2 卡上，优先跑通更大配置（prod7b），同时保证显存可控
# - 推荐配合：MOE_BACKEND=deepspeed（MoE kernel）、flash-attn（attention 加速）、DeepSpeed ZeRO-3（省显存）
#
# 可通过环境变量覆盖关键参数（方便你本地调参）：
#   MAX_LEN=2048 BATCH=1 GRAD_ACCUM=8 STEPS=2000 bash scripts/train_1node_2gpu_4090d_24g.sh
#

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-29502}

MAX_LEN=${MAX_LEN:-2048}          # 先从 2k 稳定起步，再逐步加到 4k/8k
BATCH=${BATCH:-1}                # micro-batch per GPU（24G 建议从 1 起）
GRAD_ACCUM=${GRAD_ACCUM:-8}      # global batch = BATCH * NPROC * GRAD_ACCUM
STEPS=${STEPS:-0}                # 0 表示跑满 epochs；建议用 max_steps 做阶段训练
EPOCHS=${EPOCHS:-1}

OUT_DIR=${OUT_DIR:-./outputs_2x4090d_prod7b}
NUM_WORKERS=${NUM_WORKERS:-4}

# P0 开关（推荐）
export MOE_BACKEND=${MOE_BACKEND:-deepspeed}
export USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}

echo "=============================="
echo "[train][1node-2gpu-4090d]"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  MAX_LEN=${MAX_LEN}"
echo "  BATCH=${BATCH}"
echo "  GRAD_ACCUM=${GRAD_ACCUM}"
echo "  global_batch=$((BATCH * NPROC_PER_NODE * GRAD_ACCUM))"
echo "  MOE_BACKEND=${MOE_BACKEND}"
echo "  USE_FLASH_ATTN=${USE_FLASH_ATTN}"
echo "  OUT_DIR=${OUT_DIR}"
echo "=============================="

# 说明：
# - --deepspeed + --zero_stage 3：显存最省，适合 2x24G 撑更大模型
# - --attention_backend flash：优先用 flash-attn（不可用会 fallback）
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train/train_multimodal.py \
  --data_dir ./data \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --config_preset prod7b \
  --attention_backend flash \
  --max_length "${MAX_LEN}" \
  --image_size 224 \
  --batch_size "${BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --epochs "${EPOCHS}" \
  --bf16 \
  --distributed \
  --deepspeed \
  --zero_stage 3 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --max_grad_norm 1.0 \
  --log_interval 10 \
  --save_steps 200 \
  --max_steps "${STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --pin_memory \
  --output_dir "${OUT_DIR}"
