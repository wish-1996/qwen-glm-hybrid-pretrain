"""
统一训练入口

说明：
本文件是项目级统一训练入口，不绑定特定模型命名，
用于整合不同训练脚本，提供一致的训练接口。
"""

import argparse
import os
import torch

from configs.model_config import ModelConfig
from model.hybrid_model import HybridMMMoEModel
from data.multimodal_data_loader import get_data_loader


def pretrain(args):
    """
    统一预训练函数
    
    Args:
        args: 命令行参数
    """
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载分词器
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 加载数据
    train_loader = get_data_loader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        image_size=args.image_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        distributed=args.distributed,
        rank=0,
        world_size=1,
        seed=args.seed,
    )
    
    # 初始化模型
    config = ModelConfig()
    model = HybridMMMoEModel(config, use_multimodal=True)
    model.to(device)
    
    # 分布式训练包装
    if args.distributed:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        model = DistributedDataParallel(model, device_ids=[rank])
    
    # 优化器和学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            # 计算损失
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 打印日志
            if (step + 1) % args.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss / len(train_loader),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Unified Pretraining Script')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizers/qwen3-0.6b', help='Tokenizer path')
    
    # 模型参数
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    
    # DataLoader 参数
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Enable pin_memory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')
    
    args = parser.parse_args()
    
    # 启动预训练
    pretrain(args)


if __name__ == "__main__":
    main()
