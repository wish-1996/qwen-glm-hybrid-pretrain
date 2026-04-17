"""
参考 Hugging Face 官方库实现的 Qwen3.5 MoE 训练代码
包含负载均衡损失、梯度累积、混合精度训练等特性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import time
import os
import json


# ==========================================
# 配置类 (参考官方配置设计)
# ==========================================
@dataclass
class Qwen35TrainingConfig:
    """训练配置类"""
    # 模型参数
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    num_experts: int = 192
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 8192
    shared_expert_intermediate_size: int = 2048
    
    # 训练参数
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # MoE 特定参数
    router_aux_loss_coef: float = 0.001  # 负载均衡损失系数
    
    # 系统参数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    seed: int = 42
    
    # 检查点参数
    output_dir: str = "./outputs/qwen3_5_official"
    save_steps: int = 5000
    logging_steps: int = 100
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# ==========================================
# 负载均衡损失函数 (参考官方实现)
# ==========================================
def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算负载均衡损失 (Load Balancing Loss)
    参考 Switch Transformer 论文实现
    
    Args:
        gate_logits: 门控网络的 logits，形状为 (batch_size * seq_len, num_experts)
        num_experts: 专家数量
        top_k: 每个 token 选择的专家数量
        attention_mask: 注意力掩码，用于排除 padding token
    
    Returns:
        负载均衡损失
    """
    if gate_logits is None:
        return torch.tensor(0.0, device=gate_logits.device if gate_logits is not None else "cpu")
    
    # 计算路由权重
    routing_weights = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
    
    # 选择 top-k 专家
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    
    # 创建专家掩码
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()
    
    if attention_mask is None:
        # 计算每个专家接收的 token 比例
        tokens_per_expert = torch.mean(expert_mask, dim=0)
        # 计算路由到每个专家的平均概率
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # 处理 attention mask
        # 将 mask 扩展到与 expert_mask 相同的形状
        expert_attention_mask = (
            attention_mask.unsqueeze(-1).unsqueeze(-1)
            .expand(-1, -1, top_k, num_experts)
            .reshape(-1, top_k, num_experts)
        )
        
        # 计算每个专家接收的 token 比例 (考虑 mask)
        tokens_per_expert = torch.sum(
            expert_mask * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)
        
        # 计算路由概率
        router_per_expert_mask = (
            attention_mask.unsqueeze(-1)
            .expand(-1, -1, num_experts)
            .reshape(-1, num_experts)
        )
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_mask, dim=0
        ) / torch.sum(router_per_expert_mask, dim=0)
    
    # 计算负载均衡损失
    # 目标是让每个专家接收的 token 比例和路由概率相等
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# ==========================================
# 优化器配置 (参考官方 AdamW 设置)
# ==========================================
def create_optimizer(model: nn.Module, config: Qwen35TrainingConfig):
    """
    创建 AdamW 优化器，对 bias 和 LayerNorm 参数不进行权重衰减
    """
    # 分离需要和不需���权重衰减的参数
    decay_parameters = []
    no_decay_parameters = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 不对 bias 和 LayerNorm 参数进行权重衰减
        if "bias" in name or "norm" in name.lower() or "ln" in name.lower():
            no_decay_parameters.append(param)
        else:
            decay_parameters.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_parameters,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    return optimizer


# ==========================================
# 学习率调度器 (余弦退火 + 预热)
# ==========================================
def create_scheduler(optimizer, config: Qwen35TrainingConfig):
    """创建学习率调度器"""
    
    def lr_lambda(current_step: int):
        if current_step < config.warmup_steps:
            # 线性预热
            return float(current_step) / float(max(1, config.warmup_steps))
        # 余弦退火
        progress = float(current_step - config.warmup_steps) / float(
            max(1, config.max_steps - config.warmup_steps)
        )
        return max(
            config.min_learning_rate / config.learning_rate,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================
# 改进的 MoE 模块 (参考官方实现)
# ==========================================
class Qwen35MoEExperts(nn.Module):
    """
    专家集合，使用 3D 张量存储权重 (参考官方实现)
    """
    def __init__(self, config: Qwen35TrainingConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        
        # 使用 3D 张量存储专家权重，形状为 (num_experts, 2*intermediate_dim, hidden_dim)
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        
        self.act_fn = nn.SiLU()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (seq_len, hidden_dim)
            top_k_index: (seq_len, top_k)
            top_k_weights: (seq_len, top_k)
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # 找出哪些专家被激活
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, seq_len)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        # 只计算被激活的专家
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            
            # 找出选择该专家的 token
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            
            # 计算专家输出
            gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            
            # 加权并累加
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        
        return final_hidden_states


class Qwen35TopKRouter(nn.Module):
    """Top-K 路由器"""
    def __init__(self, config: Qwen35TrainingConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        
        # 初始化
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            router_logits: (seq_len, num_experts)
            router_scores: (seq_len, top_k)
            router_indices: (seq_len, top_k)
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
        
        # 选择 top-k
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # 归一化权重
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        
        return router_logits, router_top_value, router_indices


class Qwen35SparseMoEBlock(nn.Module):
    """稀疏 MoE 块 (包含共享专家)"""
    def __init__(self, config: Qwen35TrainingConfig):
        super().__init__()
        self.gate = Qwen35TopKRouter(config)
        self.experts = Qwen35MoEExperts(config)
        
        # 共享专家
        self.shared_expert = nn.Sequential(
            nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.shared_expert_intermediate_size, config.hidden_size, bias=False),
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: 输出隐藏状态
            router_logits: 用于计算负载均衡损失
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        
        # 共享专家路径
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        
        # 稀疏专家路径
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        
        # 门控融合
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        
        # 合并输出
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        
        return expert_output, router_logits


# ==========================================
# 简化的模型架构 (用于训练)
# ==========================================
class Qwen35DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, config: Qwen35TrainingConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 自注意力 (简化版)
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        
        # MoE
        self.mlp = Qwen35SparseMoEBlock(config)
        
        # LayerNorm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        
        # 自注意力
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        hidden_states = residual + hidden_states
        
        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, router_logits


class Qwen35Model(nn.Module):
    """完整的 Qwen3.5 MoE 模型"""
    def __init__(self, config: Qwen35TrainingConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.embed_tokens.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            包含 loss, logits, router_logits 的字典
        """
        # 嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 创建因果掩码
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
                diagonal=1
            )
        else:
            causal_mask = attention_mask
        
        # 通过各层
        all_router_logits = []
        for layer in self.layers:
            hidden_states, router_logits = layer(hidden_states, causal_mask)
            all_router_logits.append(router_logits)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 计算 logits
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 语言建模损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # 负载均衡损失
            aux_loss = load_balancing_loss_func(
                torch.cat(all_router_logits, dim=0),
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
            
            # 总损失
            loss = loss + self.config.router_aux_loss_coef * aux_loss
        
        return {
            "loss": loss,
            "logits": logits,
            "router_logits": all_router_logits,
        }


# ==========================================
# 训练器 (参考官方 Trainer 设计)
# ==========================================
class Qwen35Trainer:
    """Qwen3.5 MoE 训练器"""
    
    def __init__(self, model: Qwen35Model, config: Qwen35TrainingConfig):
        self.model = model.to(config.device)
        self.config = config
        
        # 创建优化器
        self.optimizer = create_optimizer(model, config)
        
        # 创建学习率调度器
        self.scheduler = create_scheduler(self.optimizer, config)
        
        # 混合精度训练
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 日志
        self.log_history = []
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 将数据移动到设备
        input_ids = batch["input_ids"].to(self.config.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.config.device)
        labels = batch.get("labels", None)
        if labels is not None:
            labels = labels.to(self.config.device)
        
        # 混合精度训练
        if self.config.mixed_precision:
            with autocast():
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
            
            # 反向传播
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs["loss"] / self.config.gradient_accumulation_steps
            loss.backward()
        
        # 梯度累积
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.mixed_precision:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # 更新参数
                self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """完整训练流程"""
        print(f"开始训练，总步数: {self.config.max_steps}")
        print(f"设备: {self.config.device}")
        print(f"混合精度: {self.config.mixed_precision}")
        print(f"梯度累积步数: {self.config.gradient_accumulation_steps}")
        
        self.optimizer.zero_grad()
        
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_dataloader:
                step_metrics = self.train_step(batch)
                epoch_loss += step_metrics["loss"]
                num_batches += 1
                
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    print(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {step_metrics['loss']:.4f} | "
                        f"LR: {step_metrics['learning_rate']:.2e}"
                    )
                    
                    self.log_history.append({
                        "step": self.global_step,
                        "loss": step_metrics["loss"],
                        "learning_rate": step_metrics["learning_rate"],
                    })
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # 检查是否达到最大步数
                if self.global_step >= self.config.max_steps:
                    break
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {self.epoch} 完成，平均损失: {avg_loss:.4f}")
            
            # 评估
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"评估损失: {eval_loss:.4f}")
        
        # 保存最终模型
        self.save_checkpoint(is_final=True)
        print("训练完成！")
    
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)
                labels = batch.get("labels", None)
                if labels is not None:
                    labels = labels.to(self.config.device)
                
                if self.config.mixed_precision:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask, labels)
                else:
                    outputs = self.model(input_ids, attention_mask, labels)
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, is_final: bool = False):
        """保存检查点"""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            "final" if is_final else f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # 保存优化器状态
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # 保存调度器状态
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.bin")
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # 保存配置
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # 保存训练状态
        state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "global_step": self.global_step,
                "epoch": self.epoch,
                "log_history": self.log_history,
            }, f, indent=2)
        
        print(f"检查点已保存到: {checkpoint_dir}")


# ==========================================
# 测试运行
# ==========================================
if __name__ == "__main__":
    # 创建配置
    config = Qwen35TrainingConfig()
    
    # 创建模型
    model = Qwen35Model(config)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print("模型统计信息")
    print(f"{'='*50}")
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    print(f"层数: {config.num_hidden_layers}")
    print(f"隐藏层维度: {config.hidden_size}")
    print(f"专家数量: {config.num_experts}")
    print(f"每token专家数: {config.num_experts_per_tok}")
    print(f"{'='*50}\n")
    
    # 测试前向传播
    print("测试前向传播...")
    batch_size = 2
    seq_len = 128
    
    dummy_input = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
    }
    
    outputs = model(**dummy_input)
    print(f"损失: {outputs['loss'].item():.4f}")
    print(f"Logits 形状: {outputs['logits'].shape}")
    print("\n模型创建成功！")
