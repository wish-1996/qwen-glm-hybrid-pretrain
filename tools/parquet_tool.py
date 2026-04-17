import argparse
import json
import os
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _iter_records_from_batch(batch: pa.RecordBatch, columns: list[str]) -> Iterable[dict]:
    data = batch.to_pydict()
    num_rows = batch.num_rows
    for i in range(num_rows):
        yield {col: data[col][i] for col in columns}


def _open_parquet(path: str) -> pq.ParquetFile:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pq.ParquetFile(path)


def preview_parquet(path: str, head: int, columns: Optional[list[str]] = None) -> None:
    pf = _open_parquet(path)
    schema = pf.schema_arrow
    print(f"path: {path}")
    print(f"num_row_groups: {pf.num_row_groups}")
    print(f"num_rows: {pf.metadata.num_rows}")
    print("schema:")
    print(schema)

    if head <= 0:
        return

    selected_columns = columns or schema.names
    table = pf.read_row_group(0, columns=selected_columns)
    print(f"columns: {selected_columns}")
    print(table.slice(0, head).to_pandas())


def parquet_to_jsonl(
    input_path: str,
    output_path: str,
    text_col: str,
    keep_columns: Optional[list[str]],
    batch_rows: int,
    max_rows: Optional[int],
) -> None:
    pf = _open_parquet(input_path)
    schema = pf.schema_arrow
    if text_col not in schema.names:
        raise ValueError(f"text_col not found: {text_col}. available={schema.names}")

    if keep_columns is None:
        columns = [text_col]
    else:
        missing = [c for c in keep_columns if c not in schema.names]
        if missing:
            raise ValueError(f"columns not found: {missing}. available={schema.names}")
        columns = keep_columns

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            for row in _iter_records_from_batch(batch, columns):
                if keep_columns is None:
                    out = {"text": row[text_col]}
                else:
                    out = row
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
                if max_rows is not None and written >= max_rows:
                    print(f"wrote_rows: {written}")
                    return
    print(f"wrote_rows: {written}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet 文件路径")
    parser.add_argument("--head", type=int, default=3, help="预览前 N 行（0 表示不预览）")
    parser.add_argument("--columns", nargs="*", default=None, help="预览/导出时选择的列（默认全部）")
    parser.add_argument("--to_jsonl", default=None, help="输出 JSONL 路径（可选）")
    parser.add_argument("--text_col", default="content", help="导出 text 字段来源列名")
    parser.add_argument(
        "--keep_columns",
        nargs="*",
        default=None,
        help="导出时保留这些列（不传则只导出 {text: <text_col>}）",
    )
    parser.add_argument("--batch_rows", type=int, default=2048, help="导出时每批读取行数")
    parser.add_argument("--max_rows", type=int, default=None, help="最多导出行数（用于抽样）")
    args = parser.parse_args()

    preview_parquet(args.input, head=args.head, columns=args.columns)

    if args.to_jsonl:
        parquet_to_jsonl(
            input_path=args.input,
            output_path=args.to_jsonl,
            text_col=args.text_col,
            keep_columns=args.keep_columns,
            batch_rows=args.batch_rows,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()

