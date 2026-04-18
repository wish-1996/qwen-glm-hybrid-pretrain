"""
项目级模型配置

说明：
仓库早期以 Qwen3.5 作为参考实现，因此历史配置命名带 qwen35。
目前项目定位升级为"qwen + glm 思路融合的生产级预训练工程框架"，
对外建议统一使用本文件中的 ModelConfig。
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    项目级模型配置
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

    # -------------------------
    # RoPE / 长上下文扩展（先做最小可用：linear scaling）
    # -------------------------
    # rope_scaling_type:
    # - "none"       : 不做 scaling（超出训练长度属于 RoPE 外推，效果通常会掉）
    # - "linear"     : 线性缩放（把 pos_t 除以 factor），工程改动最小，适合作为第一阶段
    # - "ntk"        : 预留（后续实现）
    # - "dynamic_ntk": 预留（后续实现，推理时按实际长度动态调频率）
    #
    # 注意：在 3D RoPE（t,h,w）里，通常只对文本时间轴 t 做 scaling；
    # 图像 h/w 是空间坐标，不建议跟着上下文延长策略一起缩放。
    rope_scaling_type: str = "linear"
    rope_scaling_factor: float = 2.0
    # dynamic_ntk 需要知道“训练时的上下文长度”（基准长度）
    # - 例如你计划按阶段训练：4k 起步，则 base=4096
    # - 当实际序列长度 L_current > base 时，dynamic_ntk 会令 alpha=max(1, L_current/base)
    rope_scaling_base_len: int = 4096

    # 训练参数
    load_balancing_weight: float = 0.01

    # 多模态参数
    image_size: int = 224
    patch_size: int = 16
    num_patches: int = 14  # 14x14=196 patches

    # --------
    # 多模态"占位符 token"（让多模态更像生产）
    # --------
    # 说明：
    # 现在训练脚本会把 image patch tokens "显式占位"进 input_ids：
    #   input_ids_total = [<image_pad> * T_img] + [text tokens]
    # 然后在模型 forward 里用 vision_encoder 输出的 image_embeds 去替换这段占位符的 embedding。
    #
    # 这样做的好处：
    # 1) input_ids 的 token 序列里能"看见"图片位置（更像 Qwen-VL / LLaVA 等生产管线）
    # 2) 更容易和指令数据格式对齐（prompt 里明确插入 <image> / <image_pad>）
    # 3) 更容易做 packing / 多图 / 视频等扩展（只要改变占位符区间即可）
    #
    # 注意：不同 tokenizer 的 image token id 可能不同。这里给一个默认值（Qwen 系常见）。
    image_pad_token_id: int = 151655  # "<|image_pad|>" (见 tokenizer_config.json)

    def __post_init__(self):
        """
        初始化后处理
        """
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
