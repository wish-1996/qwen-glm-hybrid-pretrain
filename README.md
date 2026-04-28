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

跑一个 1-node 8-GPU 的 DDP smoke（默认只跑少量 steps，用于回归）：

```bash
bash scripts/smoke_1node_8gpu.sh
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
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 可选加速依赖（P0 相关）

#### 1) DeepSpeed（用于 MoE 内核：MOE_BACKEND=deepspeed）

```bash
pip install deepspeed
```

启用方式（不改变启动方式，仍 torchrun+DDP）：

```bash
MOE_BACKEND=deepspeed bash scripts/smoke_1node_1gpu.sh
```

#### 2) flash-attn（用于 Attention 加速：USE_FLASH_ATTN=1）

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
3) StandardAttention 接入 flash-attn；线性注意力训练路径去掉 Python for-loop
