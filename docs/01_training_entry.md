# 01 训练入口与配置

## 训练入口

项目提供两个训练入口：

1. **`train/train_multimodal.py`**：当前主要的训练脚本，支持多模态预训练
2. **`train/pretrain.py`**：统一训练入口，提供更一致的训练接口

## 配置体系

### 模型配置

- **`configs/model_config.py`**：项目级模型配置，不绑定特定模型命名
- **`configs/qwen35_config.py`**：兼容层，从model_config.py导入

## 环境配置（推荐）

推荐环境：**PyTorch 2.5.1 + Python 3.11 + CUDA 12.4**。

### 1) 环境自检

进入环境后先确认 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda)"
```

### 2) 安装项目依赖（不包含 PyTorch）

仓库根目录提供：
- `requirements.txt`（运行依赖，不包含 PyTorch；要求 PyTorch >= 2.0）
- `env/environment.yml`（conda 入口，内部仍通过 requirements.txt 装依赖）

推荐 pip 安装方式：

```bash
pip install -r requirements.txt
```

### 3) 安装 PyTorch（按你的平台选择）

如果你使用的基础镜像/环境里已经带了 PyTorch，可以跳过本步骤。

若需要自行安装，可参考官方 CUDA 12.4 wheel（示例）：

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  # PyTorch>=2.0
```

> Windows 下部分可选依赖（如 flash-attn）可能不易安装；建议先跑通训练链路，再在 Linux 服务器上启用 flash-attn/大规模并行。

## 环境配置（推荐）

你截图里的环境 **PyTorch 2.5.1 + Python 3.11 + CUDA 12.4** 是适配的，建议保持一致。

### 1) 环境自检

进入环境后先确认 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda)"
```

### 2) 安装项目依赖（不包含 PyTorch）

仓库根目录提供：
- `requirements.txt`（运行依赖）
- `env/environment.yml`（conda 入口，内部仍通过 requirements.txt 装依赖）

推荐 pip 安装方式：

```bash
pip install -r requirements.txt
```

### 3) 安装 PyTorch（按你的平台选择）

如果你使用的基础镜像/环境里已经带了 PyTorch（比如你截图的 PyTorch 2.5.1 + CUDA 12.4），可以跳过本步骤。

若需要自行安装，可参考官方 CUDA 12.4 wheel（示例）：

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

> Windows 下部分可选依赖（如 flash-attn）可能不易安装；建议先跑通训练链路，再在 Linux 服务器上启用 flash-attn/大规模并行。

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

---

## 训练步数（steps）/ epoch / 训练时间：怎么估算？（生产口径）

生产预训练一般按 **目标 tokens**（而不是固定 epochs）规划训练量。仓库提供工具：

`tools/estimate_training_plan.py`

它会：
1) 统计数据量（CSV 图文、JSONL 文本、Parquet 文本）
2) 用 tokenizer 抽样估算 **平均有效 tokens**（`attention_mask.sum()`，padding 不计入 loss）
3) 根据 `world_size / batch_size / grad_accum` 计算 **tokens/optimizer_step** 与需要的总 steps
4) 读取 `metrics_rank0.jsonl` 里的 `tokens_per_sec`，估算训练时长

### 示例：按目标 tokens 规划训练

```bash
python tools/estimate_training_plan.py \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --data_dir ./data \
  --max_length 4096 \
  --world_size 8 --batch_size 1 --grad_accum 8 \
  --target_tokens 3e11
```

### 示例：生产口径 Parquet + 用日志推算吞吐

```bash
python tools/estimate_training_plan.py \
  --tokenizer_path ./tokenizers/qwen3-0.6b \
  --data_dir ./data \
  --parquet_glob "/path/to/ultrafineweb-zh-part-*-of-256.parquet" \
  --metrics_jsonl ./outputs/metrics_rank0.jsonl \
  --max_length 4096 \
  --world_size 8 --batch_size 1 --grad_accum 8 \
  --target_tokens 3e11 \
  --output_json training_plan.json
```

> 注意：当前 dataloader 的 `Dataset.__len__()` 以 `mm_pairs`（CSV+image_cache 命中）为基准；文本-only 是按概率注入（mix_ratio），因此“按 epoch”只是一种辅助口径。

---

## DeepSpeed（ZeRO-3）显存优化（推荐在服务器上启用）

当模型规模较大（例如 7B 目标配置）时，纯 DDP 会在每张 GPU 上复制一整份参数/梯度/优化器状态，显存压力很大。
推荐使用 DeepSpeed ZeRO-3 做参数/梯度/优化器状态分片。

### 安装（建议 Linux）

```bash
pip install deepspeed
```

### 运行示例（torchrun + DeepSpeed engine）

```bash
torchrun --standalone --nproc_per_node=8 train/train_multimodal.py \
  --deepspeed --zero_stage 3 \
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
  --output_dir ./outputs_ds_zero3
```

> 说明：你也可以用 `--deepspeed_config configs/deepspeed_zero3_bf16.json` 传入配置文件；不传则脚本会按命令行参数生成一份。
