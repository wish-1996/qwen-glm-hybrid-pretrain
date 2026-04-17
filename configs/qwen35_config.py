"""
兼容层：历史文件名 qwen35_config.py

说明：
仓库早期以 Qwen3.5 作为参考实现，因此配置命名带 qwen35。
目前项目定位升级为“qwen + glm 思路融合的生产级预训练工程框架”，
对外建议统一使用 configs/model_config.py 中的 ModelConfig。
"""

from .model_config import ModelConfig as Qwen35Config
