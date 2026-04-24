"""
生产级 7B 目标配置

按当前 per-layer MoE 实现口径对齐 7B/0.6B
"""

from dataclasses import dataclass


@dataclass
class Prod7BModelConfig:
    """
    生产级 7B 目标配置
    """
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_kv_heads: int = 4
    num_experts: int = 128
    top_k: int = 4
    intermediate_size: int = 5734
    num_layers: int = 28
    vocab_size: int = 151936
    max_seq_length: int = 40960

    head_dim: int = None

    rope_scaling_type: str = "linear"
    rope_scaling_factor: float = 2.0
    rope_scaling_base_len: int = 4096

    load_balancing_weight: float = 0.01

    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 14

    image_pad_token_id: int = 151655

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
