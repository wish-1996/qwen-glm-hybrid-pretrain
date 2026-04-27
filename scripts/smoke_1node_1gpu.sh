#!/bin/bash

# 1-node 1-GPU smoke test 脚本
# 用于验证多模态预训练全链路是否正常工作

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 可配置：smoke 只跑少量 optimizer steps（默认 30）
SMOKE_STEPS=${SMOKE_STEPS:-30}
OUT_BASE=${OUT_BASE:-./checkpoints_smoke}
SKIP_EVAL=${SKIP_EVAL:-1}  # 默认跳过 eval，加速 smoke；需要 eval 时设为 0

set -e
 
run_train() {
  local name=$1
  shift
 
  echo "=============================="
  echo "[smoke][1gpu] case=${name}"
  echo "  MOE_BACKEND=${MOE_BACKEND:-}"
  echo "  USE_FLASH_ATTN=${USE_FLASH_ATTN:-}"
  echo "  SMOKE_STEPS=${SMOKE_STEPS}"
  echo "=============================="
 
  python train/train_multimodal.py \
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
      --log_interval 1 \
      --save_interval 999999 \
      --save_steps 0 \
      --max_steps "${SMOKE_STEPS}" \
      --output_dir "${OUT_BASE}/${name}" \
      --num_workers 2 \
      --pin_memory \
      --seed 42 \
      "$@"
}
 
# Case A: native MoE（对照组）
unset MOE_BACKEND
run_train "native"
 
# Case B: DeepSpeed-MoE（P0-1）
export MOE_BACKEND=deepspeed
run_train "deepspeed_moe"
 
if [ "${SKIP_EVAL}" != "1" ]; then
  echo "[smoke][1gpu] running eval..."
  python eval/evaluate_multimodal.py \
      --data_dir ./data \
      --tokenizer_path ./tokenizers/qwen3-0.6b \
      --max_length 512 \
      --image_size 224 \
      --batch_size 8 \
      --num_workers 2 \
      --pin_memory \
      --seed 42
fi
