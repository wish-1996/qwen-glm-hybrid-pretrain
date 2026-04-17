"""
Qwen3.5 模型配置文件
"""

from dataclasses import dataclass


@dataclass
class Qwen35Config:
    """
    Qwen3.5 模型配置
    """
    # 模型基本参数
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_kv_heads: int = 4  # GQA: 16 Q heads share 4 KV heads (4:1)
    num_experts: int = 192
    top_k: int = 4
    intermediate_size: int = 7168  # hidden_size * 3.5
    num_layers: int = 28
    vocab_size: int = 151936  # 与分词器匹配
    max_seq_length: int = 40960
    
    # 注意力参数
    head_dim: int = None
    
    # 训练参数
    load_balancing_weight: float = 0.01
    
    # 多模态参数
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 14  # 14x14=196 patches
    
    def __post_init__(self):
        """
        初始化后处理
        """
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
