"""
MoE (Mixture-of-Experts) 模块

包含：
- SwiGLU 激活函数
- SharedExpertMoE 实现

设计目标：
1. 可独立阅读/测试/复用
2. 便于后续生产化（capacity / dropless / EP）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# SwiGLU 激活函数
# ==========================================
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        out = torch.nn.functional.silu(gate) * up
        return self.down_proj(out)


# ==========================================
# Shared Expert MoE（native：功能验证版，存在 Python 循环瓶颈）
# ==========================================
class SharedExpertMoENative(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.load_balancing_weight = 0.01  # 辅助损失的权重
        
        # 稀疏专家 (Sparse Experts): 动态路由，使用 SwiGLU
        self.experts = nn.ModuleList([
            SwiGLU(self.hidden_size, self.intermediate_size)
            for _ in range(self.num_experts)
        ])
        
        # 共享专家 (Shared Expert): 所有 token 都会经过
        self.shared_expert = SwiGLU(self.hidden_size, self.intermediate_size)
        
        # 门控网络
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

    def forward(self, x):
        # x: 输入 hidden states，形状 [B, N, H]，示例：[2, 228, 2048]
        # B=批次大小, N=序列长度, H=hidden_size
        B, N, D = x.shape
        
        # 展平以便处理：[B, N, H] -> [B*N, H]
        flat_x = x.view(-1, D)
        
        # 路由得分计算
        # gate: [B*N, H] -> [B*N, num_experts]
        # 示例：[228, 2048] -> [228, 192]
        router_logits = self.gate(flat_x)
        # 计算 softmax 概率：[B*N, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)
        # 取 top-k：[B*N, top_k]
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 归一化权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # --- 计算负载均衡损失 (仅在训练时) ---
        if self.training:
            # 1. 计算"路由器建议"的负载 (Router Prob)
            # routing_weights.mean(dim=0): [B*N, E] -> [E]，每个专家被建议的平均概率
            router_prob_expert = routing_weights.mean(dim=0)  # [num_experts]，示例：[192]
            
            # 2. 计算"实际发生"的负载 (Expert Frequency)
            expert_mask = torch.zeros_like(routing_weights)  # [B*N, num_experts]
            # 填充 top-k 的位置为 1
            expert_mask.scatter_(1, top_k_indices, 1)
            # 计算实际频率：[num_experts]
            expert_frequency = expert_mask.mean(dim=0)  # [num_experts]
            
            # 3. 计算辅助损失 (Aux Loss)
            # 公式：sum(router_prob * expert_frequency) * num_experts
            aux_loss = (router_prob_expert * expert_frequency).sum() * self.num_experts
            
            # 将辅助损失保存为属性，方便外部获取
            self.aux_loss = aux_loss
        
        # 初始化输出：[B*N, H]
        sparse_out = torch.zeros_like(flat_x)
        
        # 遍历每个专家
        for expert_idx in range(self.num_experts):
            # 找出选择当前专家的所有 token 位置
            # 遍历 top-k 个位置
            for k in range(self.top_k):
                # mask: [B*N]，当前专家在第 k 个位置被选中的 token
                mask = (top_k_indices[:, k] == expert_idx)
                if mask.any():
                    # 获取选中的 token：[selected_num, H]
                    selected_tokens = flat_x[mask]
                    # 获取对应的权重：[selected_num]
                    weights = top_k_weights[mask, k]
                    # 计算专家输出：[selected_num, H]
                    expert_output = self.experts[expert_idx](selected_tokens)
                    # 加权并累加到输出：[selected_num, H]
                    sparse_out[mask] += expert_output * weights.unsqueeze(-1)
        
        # 添加共享专家的输出（所有 token 都经过）
        # shared_out: [B*N, H]
        shared_out = self.shared_expert(flat_x)
        # 总输出 = 稀疏专家输出 + 共享专家输出
        total_out = sparse_out + shared_out
        
        # 恢复原始形状：[B*N, H] -> [B, N, H]
        total_out = total_out.view(B, N, D)
        
        return total_out


# ==========================================
# Shared Expert MoE（统一入口：按 config.moe_backend 选择实现）
# - native：本文件的 SharedExpertMoENative（仅用于功能验证）
# - deepspeed：model/moe_deepspeed.py::SharedExpertMoEDeepSpeed（P0-1）
# ==========================================
class SharedExpertMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        backend = str(getattr(config, "moe_backend", "native")).lower().strip()

        if backend == "deepspeed":
            from .moe_deepspeed import SharedExpertMoEDeepSpeed

            self.impl = SharedExpertMoEDeepSpeed(config)
        elif backend in ("native", "", "none"):
            self.impl = SharedExpertMoENative(config)
        else:
            raise ValueError(f"Unknown moe_backend={backend}, expected 'native' or 'deepspeed'")

        # 训练脚本会读取 layer.moe.aux_loss（保持兼容）
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        y = self.impl(x)
        self.aux_loss = getattr(self.impl, "aux_loss", torch.tensor(0.0, device=y.device))
        return y
