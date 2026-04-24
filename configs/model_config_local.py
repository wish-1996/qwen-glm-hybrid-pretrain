"""
本地调试模型配置

目标：4060-8G 也能跑通训练链路
"""

from dataclasses import dataclass


@dataclass
class LocalModelConfig:
    """
    本地调试小模型配置
    """
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_kv_heads: int = 2
    num_experts: int = 8
    top_k: int = 2
    intermediate_size: int = 1408
    num_layers: int = 4
    vocab_size: int = 151936
    max_seq_length: int = 512

    head_dim: int = None

    rope_scaling_type: str = "linear"
    rope_scaling_factor: float = 2.0
    rope_scaling_base_len: int = 512

    load_balancing_weight: float = 0.01

    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 14

    image_pad_token_id: int = 151655

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
