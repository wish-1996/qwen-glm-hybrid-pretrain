#!/bin/bash
set -e

# 本地 RTX 4060 8G 友好 smoke
# - 使用 --config_preset local（configs/model_config_local.py）
# - batch_size 默认更小，max_steps 默认更短
# - 默认不启用 flash-attn / deepspeed，避免本地环境依赖问题

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

SMOKE_STEPS=${SMOKE_STEPS:-30}
OUT_DIR=${OUT_DIR:-./checkpoints_smoke/local_4060_8g}

# 显存紧张时建议：
# - BATCH=1 或 2
# - GRAD_ACCUM 增大以模拟更大 batch
BATCH=${BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_LEN=${MAX_LEN:-512}

echo "=============================="
echo "[smoke][local_4060_8g]"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  SMOKE_STEPS=${SMOKE_STEPS}"
echo "  BATCH=${BATCH}"
echo "  GRAD_ACCUM=${GRAD_ACCUM}"
echo "  MAX_LEN=${MAX_LEN}"
echo "=============================="

# 可选：如果你本机 bf16 不可用，把 --bf16 换成 --fp16
python train/train_multimodal.py \
  --data_dir ./data \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --config_preset local \
  --attention_backend torch \
  --max_length "${MAX_LEN}" \
  --image_size 224 \
  --batch_size "${BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --epochs 1 \
  --bf16 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 10 \
  --max_grad_norm 1.0 \
  --log_interval 1 \
  --save_interval 999999 \
  --save_steps 0 \
  --max_steps "${SMOKE_STEPS}" \
  --output_dir "${OUT_DIR}" \
  --num_workers 0 \
  --seed 42
