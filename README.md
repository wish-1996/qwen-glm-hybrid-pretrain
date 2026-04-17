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
│   └── qwen35_tiny_model.py    # 历史实现（后续可逐步重构）
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

## Roadmap（建议的下一步）

1) 训练入口升级为 YAML/JSON 配置体系（CLI 可覆盖）  
2) 增加 `torchrun` 的 1node/8gpu smoke 脚本与 DDP 训练说明  
3) 增加 MoE 负载监控（每个 expert token 直方图）与通信耗时统计  
4) 增加 packing 与跨样本边界 mask（生产预训练关键点）  
5) 增加 checkpoint resume（包含 RNG states）与更完善的 eval（PPL + probe）
