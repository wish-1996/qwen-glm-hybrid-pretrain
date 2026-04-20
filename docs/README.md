# 文档索引（docs/）

目标：每一项"生产级要素"都对应一份可阅读、可复现、可扩展的说明文档。

1. [01 训练入口与配置](./01_training_entry.md)
2. [02 模型侧：原生多模态 + MoE](./02_model_architecture.md)（含端到端 shape walkthrough：从 batch 到 logits）
3. [03 并行与通信：DP/TP/PP/EP 与 all-to-all](./03_parallelism_and_communication.md)
4. [04 训练稳定性与内存：BF16、重计算、梯度裁剪、ZeRO](./04_stability_and_memory.md)
5. [05 多模态序列对齐（Step 1）](./05_multimodal_sequence_alignment.md)
6. [06 数据管线：格式、混合采样、packing 与 mask](./06_data_pipeline.md)
7. [07 Checkpoint / 日志 / 评测](./07_checkpoint_logging_eval.md)
8. [08 注意力实现与 3D RoPE（M-RoPE）](./08_attention_and_mrope.md)
9. [09 生产级训练落地计划（Roadmap + Checklist）](./09_production_plan.md)

## Tools（可直接运行的工程脚本）

- 参数量估算（用于对齐 7B total / 0.6B active）：`python tools/param_count.py --help`
- 参数量估算（用于对齐 7B total / 0.6B active）：`python tools/param_count.py --help`
- Speculative Decoding demo：`python tools/run_spec_decode_demo.py --help`
- 显存占用分析（按你们真实实现口径估算）：`python tools/mem_profile.py --help`

## 配置预设（本地/生产）

- 本地调试小模型：`configs/model_config_local.py`（目标：4060-8G 也能跑通训练链路）
- 生产级 7B 目标：`configs/model_config_prod_7b.py`（按当前 per-layer MoE 实现口径对齐 7B/0.6B）
