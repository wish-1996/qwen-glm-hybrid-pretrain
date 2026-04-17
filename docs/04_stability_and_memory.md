# 04 训练稳定性与内存：BF16、重计算、梯度裁剪、ZeRO

## 训练稳定性

### 1. 混合精度训练

- **BF16**：Brain Float 16，提供更广的动态范围
- **FP16**：半精度浮点数，减少内存使用
- **实现**：使用 `torch.cuda.amp` 或 `torch.autocast`
- **优势**：
  - 减少内存使用
  - 加速计算
  - 提高训练稳定性

### 2. 梯度裁剪

- **描述**：限制梯度的范数，防止梯度爆炸
- **实现**：使用 `torch.nn.utils.clip_grad_norm_`
- **参数**：`max_grad_norm`，通常设置为 1.0
- **优势**：
  - 防止训练不稳定
  - 加速收敛

### 3. 学习率调度

- **线性 warmup**：从低学习率逐渐增加到目标学习率
- **余弦衰减**：学习率在训练后期逐渐衰减
- **实现**：使用 `transformers.get_linear_schedule_with_warmup`
- **优势**：
  - 防止训练初期的不稳定性
  - 提高模型性能

### 4. 权重初始化

- ** Xavier/Glorot 初始化**：适合线性层
- ** He 初始化**：适合 ReLU 激活函数
- **实现**：在模型构造函数中设置
- **优势**：
  - 防止梯度消失或爆炸
  - 加速收敛

## 内存优化

### 1. 激活重计算（Checkpointing）

- **描述**：在前向传播中不保存激活值，反向传播时重新计算
- **实现**：使用 `torch.utils.checkpoint`
- **适用场景**：深层模型，内存受限
- **优势**：
  - 减少内存使用
  - 允许训练更深的模型
- **劣势**：
  - 增加计算时间

### 2. 梯度累积

- **描述**：累积多个小批次的梯度，然后一次性更新参数
- **实现**：在训练循环中累积梯度，每 N 个批次更新一次
- **参数**：`gradient_accumulation_steps`
- **优势**：
  - 模拟更大的批次大小
  - 减少内存使用

### 3. ZeRO（Zero Redundancy Optimizer）

- **描述**：将优化器状态、梯度和参数分散到多个 GPU 上
- **实现**：使用 DeepSpeed 或 FairScale
- **级别**：
  - ZeRO-1：优化器状态分片
  - ZeRO-2：梯度分片
  - ZeRO-3：参数分片
- **优势**：
  - 显著减少内存使用
  - 支持更大的模型

### 4. 内存-efficient AdamW

- **描述**：优化 AdamW 优化器的内存使用
- **实现**：使用 `torch.optim.AdamW` 的 `foreach` 选项
- **优势**：
  - 减少优化器状态的内存使用
  - 加速优化器更新

## 实现示例

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        # 前向传播
        logits, past_states, aux_loss = model(...)
        loss = calculate_loss(logits, labels)
    
    # 反向传播
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 梯度累积

```python
gradient_accumulation_steps = 4

for i, batch in enumerate(train_loader):
    # 前向传播
    logits, past_states, aux_loss = model(...)
    loss = calculate_loss(logits, labels)
    loss = loss / gradient_accumulation_steps
    
    # 反向传播
    loss.backward()
    
    # 每 N 个批次更新一次参数
    if (i + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## 监控与调试

1. **内存监控**：
   - 使用 `torch.cuda.memory_allocated()`
   - 使用 `nvidia-smi` 命令行工具

2. **训练稳定性监控**：
   - 监控损失值的变化
   - 监控梯度范数
   - 监控学习率变化

3. **常见问题**：
   - **NaN 损失**：检查数据、学习率、模型初始化
   - **OOM 错误**：减少批次大小、使用梯度累积、启用混合精度
   - **训练不稳定**：调整学习率、使用梯度裁剪、检查数据质量
