# qwen-glm-hybrid-pretrain

基于 **Qwen 系列（工程风格）** + **GLM-5（MoE/长上下文/异步训练思路）** 的“从零预训练”工程化项目，用于展示我们作为算法工程师具备：

- 从零搭建 **原生多模态（early-fusion）** Transformer 训练栈的能力
- 从零实现/集成 **MoE（Mixture-of-Experts）** 的关键训练要素（router、aux loss、负载均衡）
- 面向生产的训练工程能力：数据管线、分布式训练、稳定性、checkpoint、日志与评测

> 目标：**接近生产级预训练**（跑通全链路 + 原理齐全），并不追求在这个仓库里把模型完整训练到最终效果。

---

## 快速开始（Smoke Test）

先跑一个 1-node 1-GPU 的 smoke test（几十步验证全链路）：

```bash
bash scripts/smoke_1node_1gpu.sh
```

如果你是本地 **RTX 4060 8G**（显存紧张），建议用"本地小模型 preset"跑 smoke（更稳，不易 OOM）：

```bash
bash scripts/smoke_local_4060_8g.sh
```

如果你在单机有 8 张 GPU，可以跑一个 1-node 8-GPU 的 DDP smoke（用于回归验证）：

```bash
bash scripts/smoke_1node_8gpu.sh
```

## 单机 2x4090D（24G）推荐启动方式

你只有两张卡（每卡 24G），**总显存不是 48G 共享**，而是 **每张卡各 24G**。  
想跑更大配置（如 `prod7b`）建议直接用 **DeepSpeed ZeRO-3**（否则 optimizer states/activation 很容易 OOM）。

推荐脚本：

```bash
bash scripts/train_1node_2gpu_4090d_24g.sh
```

常用调参（从稳到激进）：

```bash
# 先从 2k 上下文稳定起步
MAX_LEN=2048 BATCH=1 GRAD_ACCUM=8 STEPS=200 bash scripts/train_1node_2gpu_4090d_24g.sh

# 若显存足够，再尝试 4k（必要时把 BATCH 降到 1 并增大 GRAD_ACCUM）
MAX_LEN=4096 BATCH=1 GRAD_ACCUM=16 STEPS=200 bash scripts/train_1node_2gpu_4090d_24g.sh
```

## P1：从 1024 开始训练（packing + varlen）

你说"从 1024 开始训练"，推荐先把 **padding 浪费**砍掉，再逐步上更长上下文。

### 方案 A（推荐先用）：text-only + packing（吞吐最稳）

适合先热身/对齐训练链路，不引入图像 prefix，packing 能显著减少 padding：

```bash
# 2x4090D：text-only + packing + flash-attn varlen
MOE_BACKEND=deepspeed USE_FLASH_ATTN=1 \
torchrun --nproc_per_node=2 train/train_multimodal.py \
  --dataset_mode text \
  --packing \
  --config_preset prod7b \
  --max_length 1024 \
  --attention_backend flash_varlen \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --distributed \
  --deepspeed --zero_stage 3 \
  --max_steps 200 \
  --output_dir ./outputs_text_pack_1024
```

如果你用的是 `data/ultrafineweb_zh/*.parquet`（字段列名为 `content`），可直接跑 warmup 脚本：

```bash
bash scripts/warmup_1node_2gpu_text_parquet_1024.sh
```

### 方案 B：multimodal + dynamic padding + flash-varlen

多模态训练仍然保留 image prefix（T_img=196），text 部分采用 dynamic padding，使 attention_mask 具有 0/1，
从而启用 flash-attn varlen。

```bash
MOE_BACKEND=deepspeed USE_FLASH_ATTN=1 \
torchrun --nproc_per_node=2 train/train_multimodal.py \
  --dataset_mode multimodal \
  --padding_mode dynamic \
  --config_preset prod7b \
  --max_length 1024 \
  --attention_backend flash_varlen \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --distributed \
  --deepspeed --zero_stage 3 \
  --max_steps 200 \
  --output_dir ./outputs_mm_varlen_1024
```

> 运行前请确保 `./data` 下存在 `image_cache/` 和对应的 csv/jsonl（仓库已带少量示例文件）。

---

## 代码结构

```
.
├── configs/                  # 模型/训练配置
│   ├── model_config.py        # 推荐：项目级配置（不绑定 qwen35 命名）
│   └── qwen35_config.py       # 兼容层（历史文件名）
├── data/                      # 数据管线（图文 + 文本混入）
│   └── multimodal_data_loader.py
├── model/                     # 模型实现
│   ├── hybrid_model.py         # 推荐：项目级模型入口（不绑定 qwen35 命名）
│   └── hybrid_moe_model.py     # 主干可运行实现（后续可逐步拆分重构）
├── train/
│   ├── pretrain.py             # 推荐：统一训练入口
│   └── train_multimodal.py     # 训练脚本（当前主要实现）
├── eval/
│   └── evaluate_multimodal.py  # 最小评估脚本
├── scripts/
│   └── smoke_1node_1gpu.sh
└── docs/                      # 生产级要素文档（强烈建议按章维护）
    └── README.md
```

---

## 文档（生产级要素对齐）

所有“生产级要素”都拆成独立文档，见：

- [docs/README.md](docs/README.md)

---

## 环境配置（快速复现）

推荐环境：**PyTorch 2.5.1 + Python 3.11 + CUDA 12.4**（与你截图一致）。

```bash
# 1) 安装项目依赖（不包含 PyTorch）
pip install -r requirements.txt

# 2) 自检
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

> 如果你的环境里还没装 PyTorch（或版本太旧），请先安装 **PyTorch >= 2.0**（示例：CUDA 12.4 wheel）：
>
> ```bash
> pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
> ```

### 可选加速依赖（P0 相关）

#### 1) DeepSpeed（用于 MoE 内核：MOE_BACKEND=deepspeed）

```bash
# 已包含在 requirements.txt；若你未安装或需要升级，可单独执行
pip install -U deepspeed
```

启用方式（不改变启动方式，仍 torchrun+DDP）：

```bash
MOE_BACKEND=deepspeed bash scripts/smoke_1node_1gpu.sh
```

#### 2) flash-attn（用于 Attention 加速：USE_FLASH_ATTN=1 / attention_backend=flash）

> 若未安装/不兼容，会自动回退到 torch 原生实现。

```bash
pip install flash-attn --no-build-isolation
```

启用方式：

```bash
USE_FLASH_ATTN=1 bash scripts/smoke_1node_1gpu.sh
```

---

## Tools（工程脚本）

- 参数量估算（按当前 per-layer MoE 实现口径）：`python tools/param_count.py --help`
- 显存占用分析（单卡/多卡 DDP）：`python tools/mem_profile.py --help`
- 训练计划估算（数据量/avg tokens/steps/时间）：`python tools/estimate_training_plan.py --help`
- DeltaNet chunk 基准（验证 P0-3 chunk-wise 是否生效）：`python tools/bench_deltanet_chunk.py --help`

## 配置预设（本地/生产）

- 本地调试：`configs/model_config_local.py`（目标：4060-8G 也能跑通训练链路）
- 生产级 7B：`configs/model_config_prod_7b.py`（按当前 per-layer MoE 实现口径对齐 7B/0.6B）

---

## Roadmap（建议的下一步）

更完整的"生产级落地计划 + 勾选清单"请直接看：

- [docs/09_production_plan.md](docs/09_production_plan.md)

接下来优先级最高的几项：
1) MoE 路由/负载监控（expert load、topk 分布、aux_loss 曲线）
2) 长上下文阶段训练：4k→8k（linear）→16k/32k（dynamic_ntk + 稀疏 attention）
3) StandardAttention 接入 flash-attn（varlen/packing mask 适配）；线性注意力训练路径继续优化（chunk → scan kernel）
