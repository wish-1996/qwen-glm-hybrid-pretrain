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
- [x] 训练计划估算工具（数据量/avg tokens/steps/时间；与日志 tokens/s 对齐）

### 2.4 分布式训练

- [x] torchrun 兼容初始化
- [x] LOCAL\_RANK / RANK / WORLD\_SIZE 环境变量处理
- [x] DDP GPU 绑定
- [x] 分布式同步（all\_reduce for tokens）

## 阶段三：性能优化

## P0：必须先做（否则规模化基本不可谈）

> 本节是"立即执行"的 P0 计划表：优先解决**吞吐瓶颈/可扩展性瓶颈**，让 MoE + 长序列训练具备工程可行性。
> 你们已确认的选择：**MoE 库=DeepSpeed-MoE**，**启动方式=保持 torchrun+DDP**，**硬件=NVIDIA CUDA**。

### P0 计划表（建议按顺序推进）

| P0 条目 | 目标（Why） | 方案（How） | 代码改动范围（Where） | 验收标准（Done Definition） | 风险/备注 |
|---|---|---|---|---|---|
| P0-1：MoE 接入 DeepSpeed-MoE（替换 Python for-loop） | 去掉 `num_experts * top_k` 的 Python 循环瓶颈，为大专家数/大 batch/多机 EP 打基础 | 1) 保留现有 `SharedExpertMoE` 接口；2) 新增 `moe_backend=deepspeed` 分支，内部用 `deepspeed.moe.layer.MoE` 做 dispatch/combine + grouped GEMM；3) 共享专家（shared expert）保留为并行支路 `out = ds_moe_out + shared_expert_out` | `configs/model_config.py`（新增 MoE backend 配置）<br>`model/moe.py` / `model/moe_deepspeed.py`（新增 wrapper）<br>`model/hybrid_moe_model.py`（保持 import/调用不变或最小改动） | 1) 单卡 forward/backward 可跑通；2) DDP 8 卡 smoke 可跑通；3) MoE 部分无 Python token 循环；4) `aux_loss` 可用且曲线合理 | DeepSpeed 版本差异可能导致返回值不同（需做兼容 wrapper）；后续要做 EP/All2All 时再扩展 |
| P0-2：StandardAttention 接入 flash-attn（并避免 repeat_interleave 扩 KV） | 降低注意力计算时间与显存，给长上下文/packing 留空间 | 使用 flash-attn（优先支持 GQA/MQA），让 K/V 不做 head 维复制；保留 fallback 到原实现 | `model/hybrid_moe_model.py::StandardAttention`（新增 flash-attn 路径） | 1) 功能对齐（数值允许轻微误差）；2) 显存下降、吞吐提升；3) 允许后续接 varlen/packing | 依赖 CUDA/flash-attn 编译；需要在 README/脚本里给安装指引 |
| P0-3：GatedDeltaNet 训练分支去 token 级 for-loop（至少 chunk 化） | 线性注意力训练在长序列下避免 O(N) Python 循环 | 先做 chunk-wise（例如 128/256 token 一块），块内矢量化更新；后续可用 Triton/torch.compile 继续优化 | `model/hybrid_moe_model.py::GatedDeltaNet.forward`（训练分支） | 1) 去掉 `for t in range(N)`；2) 长序列（>=4k）吞吐显著改善；3) 训练 loss 正常下降 | 需要仔细处理 state 更新与数值稳定性；先以"正确+快很多"为目标 |

### P0 执行节奏（建议）

1. **先做 P0-1（MoE）**：这是当前最硬的性能瓶颈，且改动相对可控（模块边界清晰）。
2. **再做 P0-2（flash-attn）**：注意力是第二大头，且 flash-attn 接入收益稳定。
3. **最后做 P0-3（DeltaNet chunk 化）**：需要更多验证，但收益巨大。

### P0 验收脚本（建议你们在本仓库补齐）

- `scripts/smoke_1node_1gpu.sh`：新增 `MOE_BACKEND=deepspeed` 的 smoke case
- `scripts/smoke_1node_8gpu.sh`（新增）：torchrun 8 卡，跑 20~50 steps，打印 tok/s 与 loss 曲线

约定（便于 CI/回归）：
- smoke 默认使用 `--max_steps` 限制 optimizer steps（例如 30 steps），保证"验证链路"而不是"完整训练一轮"
- 关键开关：
  - `MOE_BACKEND=deepspeed`：启用 DeepSpeed-MoE 作为 MoE 内核（保持 torchrun+DDP 启动）
  - `USE_FLASH_ATTN=1`：启用 flash-attn（若环境缺依赖会自动回退）

### 3.1 显存优化

- [ ] 激活重计算（gradient checkpointing）
- [ ] ZeRO Stage 1/2/3 集成
- [ ] 混合并行（TP / PP / EP）集成
- [ ] CPU offload（ZeRO-Offload）

### 3.2 计算优化

- [x] Flash Attention 集成（基础版：可选开关 + 自动 fallback；varlen/packing mask 适配后续做）
- [x] DeltaNet 训练分支 chunk-wise（去掉 token 级 for-loop）
- [x] varlen 动态 padding（数据侧输出 attention_mask 的 0/1，可用于 flash-attn varlen unpad）
- [x] text-only sample packing（EOS 拼接，减少 padding 浪费）
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
- [ ] 数据集统计与版本化（rows/tokens/混合比例；用于 steps/epoch 计算的依据）

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

### 使用 DeepSpeed-MoE 后端（P0-1）

> 说明：这里仍然使用 torchrun+DDP 启动方式，仅替换 MoE 内核实现。

```bash
MOE_BACKEND=deepspeed torchrun --nproc_per_node=8 train/train_multimodal.py \
    --data_dir ./data \
    --tokenizer_path ./tokenizers/qwen3-0.6b \
    --batch_size 8 \
    --epochs 1 \
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

