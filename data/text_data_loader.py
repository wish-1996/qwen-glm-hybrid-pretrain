"""
Text-only 数据加载 + packing（P1）

支持：
1) JSONL：每行 {"text": "..."}（适合小规模/本地调试）
2) Parquet：大规模语料（适合多卡热身/预训练）

特性：
- sample packing：用 EOS 分隔把多条样本拼到 max_length，减少 padding 浪费
- Parquet 流式读取：不会一次性把数据全读到内存
- 多卡分片：Parquet 模式按"文件维度"做 shard（file_index % world_size == rank）
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class TextDataConfig:
    data_dir: str
    max_length: int = 1024
    seed: int = 42

    # 数据格式：jsonl/parquet
    text_format: str = "jsonl"

    # jsonl
    jsonl_path: str = "ultrafineweb_zh/sample_5.jsonl"
    jsonl_text_key: str = "text"

    # parquet
    parquet_glob: str = "ultrafineweb_zh/*.parquet"
    parquet_text_column: str = ""  # 空则自动探测：text/content/passage

    # packing
    packing: bool = True
    min_packed_tokens: int = 128


class TextJsonlDataset(Dataset):
    def __init__(self, tokenizer, cfg: TextDataConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg
        random.seed(cfg.seed)

        path = cfg.jsonl_path
        if not os.path.isabs(path):
            path = os.path.join(cfg.data_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"jsonl not found: {path}")

        self.texts: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get(cfg.jsonl_text_key)
                if isinstance(t, str) and t:
                    self.texts.append(t)

        if not self.texts:
            raise RuntimeError(f"no valid text found in {path}")

        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.texts[idx]
        enc = self.tokenizer(
            t,
            max_length=self.cfg.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0).to(torch.long),
            "attention_mask": enc["attention_mask"].squeeze(0).to(torch.long),
        }


class ParquetTextIterableDataset(IterableDataset):
    """
    Parquet 流式读取 + 多卡分片（按文件 shard）。
    """

    def __init__(self, tokenizer, cfg: TextDataConfig, *, rank: int, world_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.rank = int(rank)
        self.world_size = int(world_size)
        random.seed(cfg.seed + self.rank * 10_000)

        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id

        pattern = cfg.parquet_glob
        if not os.path.isabs(pattern):
            pattern = os.path.join(cfg.data_dir, pattern)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no parquet matched: {pattern}")

        # shard by rank
        if self.world_size > 1:
            files = [p for i, p in enumerate(files) if (i % self.world_size) == self.rank]
        if not files:
            raise RuntimeError(f"rank={self.rank}: no parquet files after sharding, check glob={cfg.parquet_glob}")

        self.files = files

    def _infer_text_column(self, schema_names: List[str]) -> str:
        if self.cfg.parquet_text_column:
            if self.cfg.parquet_text_column not in schema_names:
                raise ValueError(f"parquet_text_column={self.cfg.parquet_text_column} not in schema: {schema_names}")
            return self.cfg.parquet_text_column
        for cand in ("text", "content", "passage"):
            if cand in schema_names:
                return cand
        raise ValueError(f"cannot infer text column from parquet schema: {schema_names}, please pass --parquet_text_column")

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        try:
            import pyarrow.dataset as ds  # type: ignore
        except Exception as e:
            raise ImportError("parquet 模式需要 pyarrow：pip install pyarrow") from e

        dataset = ds.dataset(self.files, format="parquet")
        text_col = self._infer_text_column(list(dataset.schema.names))

        # scanner streaming（兼容较老 pyarrow，不用 to_table(limit=...)）
        scanner = dataset.scanner(columns=[text_col], batch_size=1024)
        for batch in scanner.to_batches():
            arr = batch.column(0)
            for t in arr.to_pylist():
                if not isinstance(t, str) or not t:
                    continue
                enc = self.tokenizer(
                    t,
                    max_length=self.cfg.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt",
                )
                yield {
                    "input_ids": enc["input_ids"].squeeze(0).to(torch.long),
                    "attention_mask": enc["attention_mask"].squeeze(0).to(torch.long),
                }


def _pack_samples(
    samples: List[Dict[str, torch.Tensor]],
    *,
    max_length: int,
    eos_id: int,
    pad_id: int,
    min_packed_tokens: int,
) -> Dict[str, torch.Tensor]:
    packed: List[torch.Tensor] = []
    cur: List[int] = []

    for s in samples:
        ids = s["input_ids"].tolist()
        if cur and cur[-1] != eos_id:
            cur.append(eos_id)
        if len(cur) + len(ids) > max_length:
            if len(cur) >= min_packed_tokens:
                packed.append(torch.tensor(cur[:max_length], dtype=torch.long))
            cur = []
        cur.extend(ids[: max(0, max_length - len(cur))])

    if cur:
        packed.append(torch.tensor(cur[:max_length], dtype=torch.long))
    if not packed:
        packed = [torch.tensor([eos_id], dtype=torch.long)]

    B = len(packed)
    input_ids = torch.full((B, max_length), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((B, max_length), dtype=torch.long)
    for i, ids in enumerate(packed):
        L = min(max_length, int(ids.numel()))
        input_ids[i, :L] = ids[:L]
        attention_mask[i, :L] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_text_dataloader(
    *,
    data_dir: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    packing: bool,
    num_workers: int,
    pin_memory: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    seed: int,
    text_format: str = "jsonl",
    jsonl_path: str = "ultrafineweb_zh/sample_5.jsonl",
    parquet_glob: str = "ultrafineweb_zh/*.parquet",
    parquet_text_column: str = "",
) -> DataLoader:
    cfg = TextDataConfig(
        data_dir=data_dir,
        max_length=int(max_length),
        seed=int(seed),
        packing=bool(packing),
        text_format=str(text_format).lower(),
        jsonl_path=jsonl_path,
        parquet_glob=parquet_glob,
        parquet_text_column=parquet_text_column,
    )

    if cfg.text_format == "parquet":
        dataset = ParquetTextIterableDataset(tokenizer, cfg, rank=rank, world_size=world_size)
        sampler = None
        shuffle = False
    else:
        dataset = TextJsonlDataset(tokenizer, cfg)
        sampler = None
        shuffle = True
        if distributed:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
            )
            shuffle = False

    def _collate(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        eos_id = getattr(dataset, "eos_id", None)
        pad_id = getattr(dataset, "pad_id", None)
        if eos_id is None:
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
        if pad_id is None:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        if not packing:
            lengths = [int(s["input_ids"].numel()) for s in samples]
            max_len = min(int(max_length), max(lengths))
            input_ids = torch.full((len(samples), max_len), int(pad_id), dtype=torch.long)
            attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
            for i, s in enumerate(samples):
                L = min(max_len, int(s["input_ids"].numel()))
                input_ids[i, :L] = s["input_ids"][:L]
                attention_mask[i, :L] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        return _pack_samples(
            samples,
            max_length=int(max_length),
            eos_id=int(eos_id),
            pad_id=int(pad_id),
            min_packed_tokens=int(cfg.min_packed_tokens),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_collate,
    )
