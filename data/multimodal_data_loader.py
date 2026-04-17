"""
生产化多模态数据加载（图像 + 文本）

设计目标（对标生产级预训练的"关键原则"）：
1) 可配置：通过参数控制路径、列名、图像尺寸、最大长度、数据混合比例等
2) 健壮：CSV/BOM/缺列/坏图像/坏行都能容错，不因单条数据中断训练
3) 可扩展：后续可平滑扩展到 video tokens / 多数据源混合 / packing
4) 性能友好：PIL 解码 + 简单 transform；DataLoader 支持 num_workers / pin_memory

注意：
这份实现先把你给的逻辑（CSV 图文 + ultrafineweb_zh 文本混入）整理成"可维护版本"。
真正生产预训练建议进一步：
- 使用 webdataset/parquet/arrow 流式读取
- 采用更高效的图像解码（opencv/accimage/turbojpeg）
- 引入 sample packing 与跨样本边界 loss mask
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# 允许加载截断/不完整的图片，避免训练被少量坏样本打断
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 增加 CSV 字段大小限制（防止超长字段报错）
csv.field_size_limit(1_000_000)  # 1MB


@dataclass
class MultimodalDataConfig:
    # 数据根目录
    data_dir: str

    # CSV 图文数据
    csv_filename: str = "68a5eee7-fde2-4787-8900-169b46fbcd93.csv"
    image_cache_subdir: str = "image_cache"
    # CSV 可能出现的列名候选（按优先级）
    url_columns: Tuple[str, ...] = ("url", "URL")
    text_columns: Tuple[str, ...] = ("cap_seg", "text", "caption")

    # 额外文本数据（ultrafineweb_zh）
    ultrafineweb_subdir: str = "ultrafineweb_zh"
    ultrafineweb_jsonl: str = "sample_5.jsonl"
    ultrafineweb_mix_ratio: float = 0.2
    # mix_ratio=0.2 表示：目标上约 20% 来自 ultrafineweb；实现为"采样时按概率抽取"

    # 处理参数
    max_length: int = 512
    image_size: int = 224

    # 训练可控性
    seed: int = 42

    # 失败容错
    max_bad_samples: int = 10_000  # 防止全是坏样本导致死循环

    # fallback（当 CSV 没读到有效样本时）
    allow_synthetic_fallback: bool = True
    synthetic_texts: Tuple[str, ...] = (
        "This is a test image",
        "A sample image for training",
        "Multimodal learning example",
        "Image and text data",
        "Deep learning training data",
    )


def _strip_bom(s: str) -> str:
    # CSV 头里可能带 BOM
    return s[1:] if s.startswith("\ufeff") else s


def _safe_open_image(path: str, image_size: int) -> Optional[torch.Tensor]:
    """
    返回：float32 tensor, shape [3, H, W], range [0,1]
    失败返回 None
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if image_size is not None:
                img = img.resize((image_size, image_size))
            # PIL -> numpy -> torch（生产上可替换为更快的 decode/transform）
            x = torch.from_numpy(np.array(img))
            x = x.permute(2, 0, 1).contiguous().float() / 255.0
            return x
    except Exception:
        return None


class MultimodalDataset(Dataset):
    """
    一个"图像+文本"样本池 + "文本-only（混入图像占位）"样本池的混合数据集。

    说明：
    - 图文样本来自 CSV + image_cache（只取本地已缓存的图片）
    - 文本样本来自 ultrafineweb_zh 的 jsonl（这里简单实现：在采样时按概率抽取）
    - 文本样本为了保持 batch 结构一致，会随机复用一张真实图片（或 fallback 图）
      （这只是为了让训练脚本先跑通；真正原生多模态预训练应引入 <image> token/patch tokens 的一致表示）
    """

    def __init__(self, tokenizer, cfg: MultimodalDataConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        random.seed(cfg.seed)

        self.image_cache_dir = os.path.join(cfg.data_dir, cfg.image_cache_subdir)
        self.ultrafineweb_dir = os.path.join(cfg.data_dir, cfg.ultrafineweb_subdir)

        self.mm_pairs: List[Tuple[str, str]] = []  # (image_path, text)
        self.text_only: List[str] = []

        self._load_csv_pairs()
        self._load_ultrafineweb()

        self._ensure_fallback_data()

        # 预先缓存一份可复用图片池（文本-only 需要随机挑一张图来对齐 batch）
        self._image_pool = [p for p, _ in self.mm_pairs]

    def _load_csv_pairs(self) -> None:
        csv_path = os.path.join(self.cfg.data_dir, self.cfg.csv_filename)
        if not os.path.exists(csv_path):
            return
        if not os.path.isdir(self.image_cache_dir):
            return

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return

            fieldnames = list(reader.fieldnames)
            cleaned_fieldnames = [_strip_bom(x) for x in fieldnames]
            field_map = {old: new for old, new in zip(fieldnames, cleaned_fieldnames)}

            for row in reader:
                try:
                    cleaned_row = {field_map.get(k, k): v for k, v in row.items()}

                    url = None
                    for c in self.cfg.url_columns:
                        if c in cleaned_row and cleaned_row[c]:
                            url = cleaned_row[c]
                            break
                    if not url:
                        continue

                    text = None
                    for c in self.cfg.text_columns:
                        if c in cleaned_row and cleaned_row[c]:
                            text = cleaned_row[c]
                            break
                    if not text:
                        text = "This is an image"

                    image_filename = os.path.basename(url)
                    image_path = os.path.join(self.image_cache_dir, image_filename)
                    if os.path.exists(image_path):
                        self.mm_pairs.append((image_path, text))
                except Exception:
                    continue

    def _load_ultrafineweb(self) -> None:
        jsonl_path = os.path.join(self.ultrafineweb_dir, self.cfg.ultrafineweb_jsonl)
        if not os.path.exists(jsonl_path):
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text")
                if isinstance(t, str) and t:
                    self.text_only.append(t)

    def _ensure_fallback_data(self) -> None:
        if len(self.mm_pairs) > 0:
            return
        if not self.cfg.allow_synthetic_fallback:
            return

        os.makedirs(self.image_cache_dir, exist_ok=True)

        img_files: List[str] = []
        for fn in os.listdir(self.image_cache_dir):
            lower = fn.lower()
            if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
                img_files.append(fn)

        if img_files:
            default_img = os.path.join(self.image_cache_dir, img_files[0])
        else:
            default_img = os.path.join(self.image_cache_dir, "dummy.jpg")
            if not os.path.exists(default_img):
                img = Image.new("RGB", (self.cfg.image_size, self.cfg.image_size), color="white")
                img.save(default_img)

        for t in self.cfg.synthetic_texts:
            self.mm_pairs.append((default_img, t))

    def __len__(self) -> int:
        return len(self.mm_pairs)

    def _sample_text(self) -> Optional[str]:
        if not self.text_only:
            return None
        if self.cfg.ultrafineweb_mix_ratio <= 0:
            return None
        if random.random() < self.cfg.ultrafineweb_mix_ratio:
            return random.choice(self.text_only)
        return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bad = 0
        while True:
            injected_text = self._sample_text()
            if injected_text is not None:
                text = injected_text
                image_path = random.choice(self._image_pool)
            else:
                image_path, text = self.mm_pairs[idx]

            pixel_values = _safe_open_image(image_path, self.cfg.image_size)
            if pixel_values is None:
                bad += 1
                if bad > self.cfg.max_bad_samples:
                    raise RuntimeError("Too many bad image samples, please check dataset integrity.")
                idx = random.randrange(len(self.mm_pairs))
                continue

            enc = self.tokenizer(
                text,
                max_length=self.cfg.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }


def build_dataloader(
    tokenizer,
    cfg: MultimodalDataConfig,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = MultimodalDataset(tokenizer=tokenizer, cfg=cfg)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=cfg.seed,
            drop_last=drop_last,
        )
        shuffle = False

    def _seed_worker(worker_id: int):
        worker_seed = (cfg.seed + rank * 10_000 + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_seed_worker if num_workers and num_workers > 0 else None,
    )


def get_data_loader(
    data_dir,
    tokenizer,
    batch_size,
    max_length,
    image_size,
    *,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
):
    cfg = MultimodalDataConfig(
        data_dir=data_dir,
        max_length=max_length,
        image_size=image_size,
        seed=seed,
    )
    return build_dataloader(
        tokenizer=tokenizer,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
