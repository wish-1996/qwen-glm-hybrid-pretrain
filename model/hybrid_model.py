"""
项目级模型入口：HybridMMMoEModel

你当前仓库里可运行的模型实现仍在 `model/hybrid_moe_model.py`（包含 VisionEncoder + MoE + Transformer）。

为了：
1) 去掉对外 API 的 qwen35 命名；
2) 让训练/评测脚本统一 import 一个稳定入口；

这里先做“薄封装（re-export）”：把 Qwen35Model 直接作为 HybridMMMoEModel 导出。
等后续你把模型文件拆分（vision/moe/backbone）后，再把实现迁移到这里即可。
"""

from __future__ import annotations

# 重要：使用相对导入，保证在 package 方式与脚本方式都尽量稳定
from .hybrid_moe_model import HybridMMMoEModel

__all__ = ["HybridMMMoEModel"]
