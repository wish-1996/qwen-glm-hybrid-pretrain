# 09 生产级训练落地计划（Roadmap + Checklist）

本文档描述将项目从"实验阶段"推进到"生产级训练"所需的工程化检查清单和路线图。

## 目标

将当前代码库从一个"可运行实验"转变为"可上线生产"的预训练系统，确保：

- 训练稳定性（BF16、梯度裁剪、容错）
- 训练效率（并行、显存优化、吞吐）
- 可复现性（seed、checkpoint、log）
- 可观测性（metrics、profiling）
- 可维护性（代码结构、文档）

## 阶段一：核心功能验证（当前阶段）

### 1.1 模型架构

- [x] MoE 模块实现（SharedExpertMoE）
- [x] 混合注意力机制（GatedDeltaNet + StandardAttention）
- [x] M-RoPE（3D 位置编码）
- [x] 多模态融合（图像 + 文本）
- [x] MTP（Multi-Token Prediction）
- [x] RoPE Scaling（linear / ntk / dynamic\_ntk）

### 1.2 数据管线

- [x] CSV 图文数据加载
- [x] ultrafineweb\_zh 文本数据混合
- [x] 多模态序列对齐
- [x] 图像缓存机制
- [x] 容错处理（坏图像、格式错误）

### 1.3 训练流程

- [x] 基础训练循环
- [x] 分布式训练（DDP）
- [x] 学习率调度
- [x] 梯度累积
- [x] 梯度裁剪
- [x] 模型保存与加载

## 阶段二：生产级特性

### 2.1 混合精度训练

- [x] BF16 autocast 支持
- [x] FP16 autocast + GradScaler 支持
- [x] AMP 训练与评估逻辑分离

### 2.2 检查点管理

- [x] epoch-based 检查点保存
- [x] step-based 检查点保存
- [x] 完整检查点内容（model / optimizer / scheduler / scaler / rng\_state）
- [x] 检查点恢复（resume）
- [x] 分布式检查点处理

### 2.3 日志与监控

- [x] 结构化日志（JSONL 格式）
- [x] rank0 专用日志文件
- [x] 训练指标记录（loss、grad\_norm、tokens\_per\_sec）
- [x] 时间统计（data\_time、step\_time）

### 2.4 分布式训练

- [x] torchrun 兼容初始化
- [x] LOCAL\_RANK / RANK / WORLD\_SIZE 环境变量处理
- [x] DDP GPU 绑定
- [x] 分布式同步（all\_reduce for tokens）

## 阶段三：性能优化

### 3.1 显存优化

- [ ] 激活重计算（gradient checkpointing）
- [ ] ZeRO Stage 1/2/3 集成
- [ ] 混合并行（TP / PP / EP）集成
- [ ] CPU offload（ZeRO-Offload）

### 3.2 计算优化

- [ ] Flash Attention 集成
- [ ] 序列 packing（多个样本打包成一个序列）
- [ ] 动态批处理（根据序列长度调整批次大小）
- [ ] 数据预取与异步加载

### 3.3 IO 优化

- [ ] 更高效的图像解码（opencv / turbojpeg）
- [ ] WebDataset / Parquet 流式读取
- [ ] 图像缓存预热

## 阶段四：可靠性

### 4.1 容错训练

- [ ] 检查点定期保存（防崩溃丢失）
- [ ] 训练中断恢复（自动 resume）
- [ ] 数据加载失败重试
- [ ] NaN / Inf 检测与处理

### 4.2 验证与测试

- [ ] 单元测试（关键模块）
- [ ] 集成测试（训练流程）
- [ ] 端到端测试（真实数据）
- [ ] 性能基准测试

### 4.3 评测

- [ ] 困惑度（PPL）评测
- [ ] 多模态评测（图像描述、VQA）
- [ ] 下游任务评测

## 阶段五：工程化

### 5.1 配置管理

- [ ] YAML / TOML 配置文件支持
- [ ] 配置验证与合并
- [ ] 超参数搜索支持

### 5.2 监控与告警

- [ ] GPU 利用率监控
- [ ] 显存使用告警
- [ ] Loss 发散告警
- [ ] 训练进度可视化

### 5.3 自动化

- [ ] 训练脚本化（run\_train.sh）
- [ ] 多节点训练脚本
- [ ] 数据准备流水线

## 快速开始

### 单机训练

```bash
python train/train_multimodal.py \
    --data_dir ./data \
    --tokenizer_path ./tokenizers/qwen3-0.6b \
    --batch_size 8 \
    --epochs 10 \
    --bf16 \
    --output_dir ./outputs
```

### 分布式训练

```bash
torchrun --nproc_per_node=8 train/train_multimodal.py \
    --data_dir ./data \
    --tokenizer_path ./tokenizers/qwen3-0.6b \
    --batch_size 8 \
    --epochs 10 \
    --bf16 \
    --distributed \
    --output_dir ./outputs
```

### 带 MTP 的训练

```bash
python train/train_multimodal.py \
    --data_dir ./data \
    --tokenizer_path ./tokenizers/qwen3-0.6b \
    --batch_size 8 \
    --epochs 10 \
    --bf16 \
    --enable_mtp \
    --mtp_k 3 \
    --mtp_weight 0.3 \
    --output_dir ./outputs
```

## 常见问题

### Q: 如何选择 BF16 还是 FP16？

A: BF16 优先，原因：

- 动态范围更大，训练更稳定
- 不需要 GradScaler（FP16 需要）
- 硬件支持更好（H100 / A100）

### Q: 如何调整 RoPE Scaling 类型？

A: 在 `configs/model_config.py` 中设置：

```python
rope_scaling_type: str = "linear"  # 或 "ntk" / "dynamic_ntk"
rope_scaling_factor: float = 2.0
rope_scaling_base_len: int = 4096  # dynamic_ntk 专用
```

### Q: 如何恢复中断的训练？

A: 使用 `--resume_from` 参数：

```bash
python train/train_multimodal.py \
    --resume_from ./outputs/checkpoint_step_1000.pt \
    ...
```

## 下一步

1. 完善阶段三的性能优化（Flash Attention、序列 packing）
2. 实现阶段四的可靠性特性（容错训练、自动化测试）
3. 完善阶段五的工程化（配置管理、监控告警）

## 贡献指南

欢迎提交 PR 来帮助完善本项目！请确保：

- 代码符合项目风格
- 添加了必要的注释和文档
- 通过了相关测试
- 更新了本文档的检查清单（如有新增特性）

