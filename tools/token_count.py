import argparse
import fnmatch
import os
from dataclasses import dataclass
from typing import Optional

import pyarrow.parquet as pq
from transformers import AutoTokenizer


class ByteTokenizer:
    def __init__(self):
        self.eos_token_id = 256

    def __len__(self) -> int:
        return 257

    def __call__(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_token_type_ids: bool = False,
    ) -> dict:
        input_ids = []
        for t in texts:
            if t is None:
                input_ids.append([])
                continue
            b = t.encode("utf-8", errors="ignore")
            input_ids.append(list(b))
        return {"input_ids": input_ids}


def load_tokenizer(tokenizer_path: str, local_files_only: bool) -> object:
    try:
        if os.path.isdir(tokenizer_path):
            files = set(os.listdir(tokenizer_path))
            if "tokenizer.json" not in files and not (("vocab.json" in files) and ("merges.txt" in files)):
                raise FileNotFoundError(
                    f"tokenizer directory seems incomplete: {tokenizer_path}. "
                    f"need tokenizer.json or (vocab.json + merges.txt). found={sorted(list(files))}"
                )
        tok = None
        last_err = None
        for use_fast in (True, False):
            try:
                tok = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    use_fast=use_fast,
                    local_files_only=bool(local_files_only),
                )
                break
            except Exception as e:
                last_err = e
                tok = None
        if tok is None:
            raise last_err or RuntimeError("tokenizer load failed")
        if tok.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id is None")
        tok.model_max_length = 10**12
        return tok
    except Exception as e:
        print(f"tokenizer load failed, fallback to ByteTokenizer. error={type(e).__name__}: {e}")
        return ByteTokenizer()


def find_parquet_files(data_dir: str, pattern: str) -> list[str]:
    files = []
    for name in os.listdir(data_dir):
        if name.endswith(".parquet") and fnmatch.fnmatch(name, pattern):
            files.append(os.path.join(data_dir, name))
    files.sort()
    return files


@dataclass
class CountResult:
    rows: int
    tokens: int


def count_tokens_in_parquet(
    parquet_path: str,
    tokenizer,
    text_col: str,
    batch_rows: int,
    max_rows: Optional[int],
    add_eos: bool,
) -> CountResult:
    pf = pq.ParquetFile(parquet_path)
    rows = 0
    tokens = 0
    eos = getattr(tokenizer, "eos_token_id", None)

    for batch in pf.iter_batches(batch_size=batch_rows, columns=[text_col]):
        data = batch.to_pydict()
        texts = data.get(text_col, [])
        if not texts:
            continue
        enc = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for ids in enc["input_ids"]:
            if not ids:
                continue
            rows += 1
            tokens += len(ids)
            if add_eos and eos is not None:
                tokens += 1
            if max_rows is not None and rows >= max_rows:
                return CountResult(rows=rows, tokens=tokens)
    return CountResult(rows=rows, tokens=tokens)


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet 文件路径或目录路径")
    parser.add_argument("--glob", default="*.parquet", help="当 input 是目录时，匹配哪些 parquet 文件")
    parser.add_argument("--text_col", default="content", help="文本列名（默认 content）")
    parser.add_argument("--tokenizer_path", default=r"tokenizers\\qwen3-0.6b", help="本地 tokenizer 目录或 HF 名称")
    parser.add_argument("--local_files_only", action="store_true", help="离线加载 tokenizer")
    parser.add_argument("--batch_rows", type=int, default=512, help="每批读取多少行（越大越快但更吃内存）")
    parser.add_argument("--max_rows", type=int, default=None, help="最多统计多少行（抽样/快速估算）")
    parser.add_argument("--no_eos", action="store_true", help="不把 EOS 计入 token 总数（默认计入）")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path, local_files_only=bool(args.local_files_only))
    add_eos = not bool(args.no_eos)

    paths = []
    if os.path.isdir(args.input):
        paths = find_parquet_files(args.input, args.glob)
    else:
        paths = [args.input]

    if not paths:
        raise FileNotFoundError(f"no parquet found: input={args.input} glob={args.glob}")

    total_rows = 0
    total_tokens = 0
    print(f"tokenizer: {type(tokenizer).__name__}")
    print(f"vocab_size: {fmt_int(len(tokenizer))}")
    print(f"eos_token_id: {getattr(tokenizer, 'eos_token_id', None)}")
    print(f"add_eos: {add_eos}")
    print(f"text_col: {args.text_col}")
    print(f"files: {len(paths)}")

    for p in paths:
        r = count_tokens_in_parquet(
            parquet_path=p,
            tokenizer=tokenizer,
            text_col=args.text_col,
            batch_rows=int(args.batch_rows),
            max_rows=int(args.max_rows) if args.max_rows is not None else None,
            add_eos=add_eos,
        )
        total_rows += r.rows
        total_tokens += r.tokens
        avg = (r.tokens / r.rows) if r.rows else 0.0
        print(f"[{os.path.basename(p)}] rows={fmt_int(r.rows)} tokens={fmt_int(r.tokens)} avg={avg:.2f}")

    avg_total = (total_tokens / total_rows) if total_rows else 0.0
    print("----- total -----")
    print(f"rows: {fmt_int(total_rows)}")
    print(f"tokens: {fmt_int(total_tokens)}")
    print(f"avg_tokens_per_row: {avg_total:.2f}")


if __name__ == "__main__":
    main()
