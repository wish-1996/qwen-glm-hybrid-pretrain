# 03 并行与通信：DP/TP/PP/EP 与 all-to-all

## 并行策略

### 1. 数据并行（DP）

- **描述**：将数据分成多个批次，每个 GPU 处理不同的批次
- **实现**：使用 `torch.nn.parallel.DistributedDataParallel`
- **适用场景**：模型较小，适合在多个 GPU 上并行处理数据
- **通信**：梯度同步，使用 `all_reduce` 操作

### 2. 张量并行（TP）

- **描述**：将模型张量分割到多个 GPU 上
- **实现**：需要对模型进行修改，支持张量分割
- **适用场景**：模型较大，单个 GPU 无法容纳整个模型
- **通信**：前向和反向传播时需要跨 GPU 通信

### 3. 流水线并行（PP）

- **描述**：将模型层分割到多个 GPU 上，形成流水线
- **实现**：需要对模型进行修改，支持层分割
- **适用场景**：模型非常深，适合流水线并行
- **通信**：相邻 GPU 之间需要传递激活值和梯度

### 4. 专家并行（EP）

- **描述**：将 MoE 中的专家分散到多个 GPU 上
- **实现**：需要对 MoE 实现进行修改
- **适用场景**：MoE 模型，专家数量较多
- **通信**：需要 `all-to-all` 通信来交换专家激活值

## 通信机制

### 1. All-to-All 通信

- **描述**：所有 GPU 之间互相交换数据
- **适用场景**：MoE 专家并行，需要在不同 GPU 之间交换专家激活值
- **实现**：使用 `torch.distributed.all_to_all`

### 2. All-Reduce 通信

- **描述**：所有 GPU 对相同大小的张量执行归约操作
- **适用场景**：数据并行中的梯度同步
- **实现**：使用 `torch.distributed.all_reduce`

### 3. Broadcast 通信

- **描述**：从一个 GPU 向其他所有 GPU 广播数据
- **适用场景**：模型初始化、学习率更新等
- **实现**：使用 `torch.distributed.broadcast`

### 4. Send/Recv 通信

- **描述**：点对点通信
- **适用场景**：流水线并行中的层间通信
- **实现**：使用 `torch.distributed.send` 和 `torch.distributed.recv`

## 实现示例

### 数据并行示例

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 创建模型
model = HybridMMMoEModel(config)

# 包装为 DDP 模型
model = DistributedDataParallel(model, device_ids=[rank])
```

### 专家并行中的 All-to-All 通信

```python
# 在 MoE 前向传播中
# 1. 收集所有 GPU 上的 token
# 2. 路由到对应的专家
# 3. 使用 all-to-all 通信交换数据
# 4. 计算专家输出
# 5. 使用 all-to-all 通信返回结果
```

## 性能优化

1. **通信优化**：
   - 使用 NCCL 后端
   - 批量通信操作
   - 使用 `torch.distributed.Stream` 重叠计算和通信

2. **内存优化**：
   - 使用混合精度训练（FP16/BF16）
   - 梯度累积
   - 激活重计算

3. **调度优化**：
   - 动态批处理大小
   - 负载均衡
   - 流水线填充策略
