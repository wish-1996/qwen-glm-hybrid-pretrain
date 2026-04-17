"""
统一训练入口（推荐）

为什么需要它：
1) 让项目对外只有一个稳定入口：`python -m train.pretrain ...`
2) 具体训练实现可以在内部演进（多模态/纯文本/视频等），不影响使用方式

当前仓库里真正的训练实现以 `train/train_multimodal.py` 为准（包含：
- 多模态对齐 Step 1
- aux_loss（MoE）叠加
- DDP sampler 支持
）

因此这里做“薄封装”：直接转发到 `train_multimodal.main()`。
"""

from .train_multimodal import main


if __name__ == "__main__":
    main()
