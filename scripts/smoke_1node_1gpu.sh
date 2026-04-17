#!/bin/bash

# 1-node 1-GPU smoke test 脚本
# 用于验证多模态预训练全链路是否正常工作

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行训练脚本，执行少量步骤验证全链路
python train/train_multimodal.py \
    --data_dir ./data \
    --tokenizer_path ./tokenizers/qwen3-0.6b \
    --max_length 512 \
    --image_size 224 \
    --batch_size 8 \
    --epochs 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --max_grad_norm 1.0 \
    --log_interval 10 \
    --save_interval 1 \
    --output_dir ./checkpoints \
    --num_workers 4 \
    --pin_memory \
    --seed 42

# 运行评估脚本验证模型
python eval/evaluate_multimodal.py \
    --data_dir ./data \
    --tokenizer_path ./tokenizers/qwen3-0.6b \
    --max_length 512 \
    --image_size 224 \
    --batch_size 8 \
    --num_workers 4 \
    --pin_memory \
    --seed 42
