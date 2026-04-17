"""
多模态模型评估和验证模块
包含模型评估、验证和检查点加载功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from transformers import AutoTokenizer

from model.qwen35_tiny import Qwen35Model, Qwen35Config
from data.multimodal_data_loader import get_data_loader


def evaluate(args):
    """
    评估函数
    """
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 加载数据
    val_loader = get_data_loader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        image_size=args.image_size,
        shuffle=False
    )
    
    # 初始化模型
    config = Qwen35Config()
    model = Qwen35Model(config, use_multimodal=True)
    model.to(device)
    
    # 加载检查点
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # 评估模式
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # 移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            # 前向传播
            outputs, _, aux_loss = model(
                input_ids=input_ids,
                pixel_values=pixel_values
            )
            
            # 计算损失
            loss = aux_loss * 0.1  # 暂时使用辅助损失
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    print(f"Validation Loss: {avg_loss:.4f}")
    
    return avg_loss

def load_checkpoint(model, checkpoint_path, device):
    """
    加载检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        model: 加载了权重的模型
        checkpoint: 检查点数据
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model, checkpoint

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='Multimodal Model Evaluation')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='d:\\study_project\\tran_qwen3_model\\data', help='Data directory')
    parser.add_argument('--tokenizer_path', type=str, default='d:\\study_project\\tran_qwen3_model\\tokenizers\\qwen3-0.6b', help='Tokenizer path')
    
    # 模型参数
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Checkpoint path')
    
    args = parser.parse_args()
    
    # 启动评估
    evaluate(args)


if __name__ == "__main__":
    main()