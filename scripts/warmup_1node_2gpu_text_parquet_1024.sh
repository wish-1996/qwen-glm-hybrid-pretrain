#!/bin/bash
set -e

# 1-node 2-GPU warmup（text-only + parquet + packing + 1024）
#
# 适用场景：
# - 先用 text-only 做"多卡部署 + 吞吐"热身（最稳、最容易定位问题）
# - 数据来自 data/ultrafineweb_zh/*.parquet（列名默认自动探测：text/content/passage）
#
# 你可以通过环境变量覆盖参数：
#   MAX_LEN=1024 BATCH=1 GRAD_ACCUM=8 STEPS=200 bash scripts/warmup_1node_2gpu_text_parquet_1024.sh

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-29503}

DATA_DIR=${DATA_DIR:-./data}
PARQUET_GLOB=${PARQUET_GLOB:-"ultrafineweb_zh/*.parquet"}
PARQUET_TEXT_COLUMN=${PARQUET_TEXT_COLUMN:-content}

MAX_LEN=${MAX_LEN:-1024}
BATCH=${BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
STEPS=${STEPS:-200}

OUT_DIR=${OUT_DIR:-./outputs_warmup_text_parquet_1024}

# 推荐开关
export MOE_BACKEND=${MOE_BACKEND:-deepspeed}
export USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}

echo "=============================="
echo "[warmup][1node-2gpu][text+parquet+packing]"
echo "  DATA_DIR=${DATA_DIR}"
echo "  PARQUET_GLOB=${PARQUET_GLOB}"
echo "  PARQUET_TEXT_COLUMN=${PARQUET_TEXT_COLUMN}"
echo "  MAX_LEN=${MAX_LEN}"
echo "  BATCH=${BATCH}"
echo "  GRAD_ACCUM=${GRAD_ACCUM}"
echo "  global_batch=$((BATCH * NPROC_PER_NODE * GRAD_ACCUM))"
echo "  STEPS=${STEPS}"
echo "  MOE_BACKEND=${MOE_BACKEND}"
echo "  USE_FLASH_ATTN=${USE_FLASH_ATTN}"
echo "  OUT_DIR=${OUT_DIR}"
echo "=============================="

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train/train_multimodal.py \
  --data_dir "${DATA_DIR}" \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --dataset_mode text \
  --text_format parquet \
  --parquet_glob "${PARQUET_GLOB}" \
  --parquet_text_column "${PARQUET_TEXT_COLUMN}" \
  --packing \
  --config_preset prod7b \
  --max_length "${MAX_LEN}" \
  --attention_backend flash_varlen \
  --batch_size "${BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --bf16 \
  --distributed \
  --deepspeed --zero_stage 3 \
  --max_steps "${STEPS}" \
  --log_interval 10 \
  --save_steps 200 \
  --output_dir "${OUT_DIR}"
