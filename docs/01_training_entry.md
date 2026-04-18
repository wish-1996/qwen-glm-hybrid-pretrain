# 01 训练入口与配置

## 训练入口

项目提供两个训练入口：

1. **`train/train_multimodal.py`**：当前主要的训练脚本，支持多模态预训练
2. **`train/pretrain.py`**：统一训练入口，提供更一致的训练接口

## 配置体系

### 模型配置

- **`configs/model_config.py`**：项目级模型配置，不绑定特定模型命名
- **`configs/qwen35_config.py`**：兼容层，从model_config.py导入

### 命令行参数

主要参数包括：

- **数据参数**：`--data_dir`、`--tokenizer_path`
- **模型参数**：`--max_length`、`--image_size`
- **训练参数**：`--batch_size`、`--epochs`、`--learning_rate`、`--weight_decay`、`--warmup_steps`、`--max_grad_norm`
- **分布式训练**：`--distributed`
- **DataLoader参数**：`--num_workers`、`--pin_memory`、`--seed`
- **其他参数**：`--output_dir`、`--log_interval`、`--save_interval`

## 运行示例

### 1-node 1-GPU Smoke Test

```bash
bash scripts/smoke_1node_1gpu.sh
```

### 手动运行

```bash
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
```

---

## torchrun 多卡启动（DDP，生产口径）

`train/train_multimodal.py` 支持 torchrun 标准环境变量（`LOCAL_RANK/RANK/WORLD_SIZE`），设备绑定使用 `LOCAL_RANK`。

### 1-node 8-GPU 示例

```bash
torchrun --standalone --nproc_per_node=8 train/train_multimodal.py \
  --distributed \
  --data_dir ./data \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --max_length 4096 \
  --image_size 224 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --output_dir ./checkpoints_4k \
  --log_interval 10 \
  --save_steps 100
```

### 断点恢复（resume）

```bash
torchrun --standalone --nproc_per_node=8 train/train_multimodal.py \
  --distributed \
  --resume_from ./checkpoints_4k/checkpoint_step_1000.pt \
  --data_dir ./data \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --max_length 4096 \
  --image_size 224 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --output_dir ./checkpoints_4k \
  --log_interval 10
```

### 训练指标输出

rank0 会在 `output_dir` 下写入：
- `metrics_rank0.jsonl`：结构化训练日志（loss/lr/grad_norm/tokens_per_sec 等）
