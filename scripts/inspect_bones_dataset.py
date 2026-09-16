#!/usr/bin/env python3
"""Print BONES-SEED metadata and G1 CSV schema information."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq

from apex_getup.demonstrations.paths import discover_bones_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    paths = discover_bones_dataset(args.dataset_root)
    parquet = pq.ParquetFile(paths.metadata)
    csv_paths = sorted(paths.g1_csv_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no CSV files under {paths.g1_csv_root}")
    with csv_paths[0].open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        csv_columns = reader.fieldnames or []
        csv_samples = [next(reader, None) for _ in range(args.samples)]

    print(f"dataset root: {paths.root}")
    print(f"metadata: {paths.metadata}")
    print(f"G1 CSV root: {paths.g1_csv_root}")
    print(f"metadata motions: {parquet.metadata.num_rows}")
    print(f"G1 CSV files: {len(csv_paths)}")
    print(f"metadata columns ({len(parquet.schema_arrow.names)}):")
    print("  " + "\n  ".join(parquet.schema_arrow.names))
    print(f"sample G1 CSV: {csv_paths[0].relative_to(paths.root)}")
    print(f"G1 fields ({len(csv_columns)}):")
    print("  " + "\n  ".join(csv_columns))
    print("sample metadata entries:")
    sample_columns = [
        "filename", "move_duration_frames", "move_g1_path", "category",
        "content_short_description",
    ]
    for row in pq.read_table(paths.metadata, columns=sample_columns).slice(0, args.samples).to_pylist():
        print(f"  {row}")
    print("sample G1 rows:")
    for row in csv_samples:
        print(f"  {row}")


if __name__ == "__main__":
    main()

