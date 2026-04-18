# 07 Checkpoint / 日志 / 评测

## 检查点（Checkpoint）

### 1. 保存策略

- **定期保存**：每 N 个 epoch 保存一次检查点
- **最佳模型保存**：保存验证损失最低的模型
- **最后模型保存**：保存训练结束时的模型

### 2. 检查点内容

- **模型权重**：`model.state_dict()`
- **优化器状态**：`optimizer.state_dict()`
- **学习率调度器状态**：`scheduler.state_dict()`
- **训练状态**：当前 epoch、步骤、损失等
- **随机种子状态**：确保可复现性

### 3. 实现示例

```python
# 保存检查点
def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'rng_states': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# 加载检查点
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 恢复随机种子状态
    if 'rng_states' in checkpoint:
        torch.set_rng_state(checkpoint['rng_states']['torch'])
        if torch.cuda.is_available() and checkpoint['rng_states']['cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_states']['cuda'])
    
    return checkpoint['epoch'], checkpoint['loss']
```

### 4. 分布式训练中的检查点

- **主进程保存**：只有 rank 0 进程保存检查点
- **所有进程加载**：所有进程加载相同的检查点
- **注意事项**：确保模型在保存前是 `module.` 前缀的状态 dict

## 日志（Logging）

### 1. 日志级别

- **DEBUG**：详细的调试信息
- **INFO**：一般信息，如训练进度、损失等
- **WARNING**：警告信息，如学习率调整、内存使用等
- **ERROR**：错误信息，如数据加载失败、模型错误等

### 2. 日志内容

- **训练信息**：epoch、步骤、损失、学习率等
- **模型信息**：参数量、激活参数量等
- **性能信息**：批次时间、GPU 利用率、内存使用等
- **数据信息**：数据加载时间、样本数量等

### 3. 日志实现

- **控制台输出**：使用 `print` 或 `logging` 模块
- **文件日志**：将日志写入文件
- **TensorBoard**：使用 `tensorboardX` 或 `torch.utils.tensorboard`

### 4. 实现示例

```python
import logging
from torch.utils.tensorboard import SummaryWriter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='training.log'
)
logger = logging.getLogger(__name__)

# 初始化 TensorBoard
writer = SummaryWriter('runs/experiment_name')

# 记录训练信息
def log_training(epoch, step, total_steps, loss, learning_rate):
    logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{total_steps}], Loss: {loss:.4f}, LR: {learning_rate:.6f}")
    writer.add_scalar('Loss/train', loss, epoch * total_steps + step)
    writer.add_scalar('Learning Rate', learning_rate, epoch * total_steps + step)

# 记录模型信息
def log_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e9:.2f}B")
    logger.info(f"Trainable parameters: {trainable_params / 1e9:.2f}B")
```

## 评测（Evaluation）

### 1. 评测指标

- **困惑度（PPL）**：评估语言模型的生成质量
- **准确率**：评估分类任务的性能
- **BLEU**：评估翻译任务的性能
- **F1 分数**：评估序列标注任务的性能
- **视觉问答准确率**：评估多模态模型的性能

### 2. 评测流程

1. **加载模型**：加载训练好的模型
2. **加载数据**：准备评测数据集
3. **模型评估**：在评测数据集上运行模型
4. **计算指标**：计算评测指标
5. **生成报告**：生成评测报告

### 3. 实现示例

```python
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits, _, aux_loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                use_cache=False
            )
            
            # 计算损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                          labels[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples
    ppl = math.exp(avg_loss)
    
    return avg_loss, ppl
```

### 4. 多模态评测

- **视觉问答（VQA）**：评估模型理解图像并回答问题的能力
- **图像描述生成**：评估模型生成图像描述的质量
- **图文检索**：评估模型在图像和文本之间的检索能力

### 5. 性能评测

- **推理速度**：评估模型的推理速度
- **内存使用**：评估模型的内存使用情况
- **吞吐量**：评估模型的吞吐量

## 最佳实践

1. **检查点管理**：
   - 定期清理旧检查点，避免磁盘空间不足
   - 使用版本控制，记录检查点的训练配置

2. **日志管理**：
   - 结构化日志，便于后续分析
   - 定期轮换日志文件，避免日志文件过大

3. **评测管理**：
   - 建立标准化的评测流程
   - 保存评测结果，便于比较不同模型的性能

4. **可复现性**：
   - 固定随机种子
   - 记录所有超参数
   - 保存完整的训练配置

## 下一步优化

1. **分布式评测**：支持多 GPU 并行评测
2. **自动化评测**：建立自动化评测流程
3. **模型分析**：分析模型的错误类型和性能瓶颈
4. **可视化工具**：使用可视化工具分析模型性能