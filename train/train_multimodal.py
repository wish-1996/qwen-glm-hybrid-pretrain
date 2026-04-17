"""
多模态模型训练脚本
包含训练循环、优化策略、分布式训练配置
"""

import sys
import os
# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
import argparse
import time
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model.qwen35_tiny_model import Qwen35Model
from configs.qwen35_config import Qwen35Config
from data.multimodal_data_loader import get_data_loader


def train(args):
    """
    训练函数
    """
    # 初始化分布式训练
    if args.distributed:
        init_process_group(backend='nccl')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = torch.device(f'cuda:{rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 打印设备信息
    print(f"Using device: {device}")
    
    # 加载分词器
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Tokenizer loaded successfully.")
    
    # 加载数据
    print(f"Loading data from {args.data_dir}...")
    train_loader = get_data_loader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        image_size=args.image_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        distributed=args.distributed,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    print(f"Data loader created. Number of batches: {len(train_loader)}")
    
    # 初始化模型
    print("Initializing model...")
    config = Qwen35Config()
    model = Qwen35Model(config, use_multimodal=True)
    model.to(device)
    print("Model initialized and moved to device.")
    
    # 分布式训练包装
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        if args.distributed:
            # DDP 下我们在 DataLoader 里使用 DistributedSampler
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
        
        start_time = time.time()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            # 移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            # 构造文本 positions（3D：[t,0,0]）
            B, T = input_ids.shape
            t = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            text_positions = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)  # [B, T, 3]
            
            # 构造 labels（mask padding）
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # 前向传播
            logits, past_states, aux_loss = model(
                input_ids=input_ids,
                positions=text_positions,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False
            )
            
            # 计算主任务损失
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            main_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 计算总损失，添加辅助损失
            aux_loss_weight = 0.01
            loss = main_loss + aux_loss_weight * aux_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 打印日志
            if rank == 0 and step % args.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step}/{len(train_loader)}, Loss: {avg_loss:.4f}")
        
        # 计算 epoch 时间
        epoch_time = time.time() - start_time
        if rank == 0:
            avg_epoch_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.4f}")
        
        # 保存检查点
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict() if not args.distributed else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # 销毁进程组
    if args.distributed:
        destroy_process_group()
    
    print("Training completed!")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Multimodal Model Training')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory (should contain image_cache/ etc.)')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizers/qwen3-0.6b', help='Tokenizer path')
    
    # 模型参数
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')

    # DataLoader 参数（生产级最常用的几个）
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', help='Enable pin_memory for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data sampling')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 启动训练
    train(args)


if __name__ == "__main__":
    main()