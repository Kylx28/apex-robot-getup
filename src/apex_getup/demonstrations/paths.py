"""Dataset path discovery without motion-specific hard-coding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BonesDatasetPaths:
    root: Path
    metadata: Path
    g1_csv_root: Path


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "datasets" / "bones-seed"


def discover_bones_dataset(root: Path | str | None = None) -> BonesDatasetPaths:
    """Find the newest metadata parquet and G1 CSV tree below ``root``."""
    dataset_root = Path(root or default_dataset_root()).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"BONES-SEED root not found: {dataset_root}")
    parquet_files = sorted((dataset_root / "metadata").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no metadata parquet found under {dataset_root / 'metadata'}")
    g1_csv_root = dataset_root / "g1" / "csv"
    if not g1_csv_root.is_dir():
        raise FileNotFoundError(f"G1 CSV directory not found: {g1_csv_root}")
    return BonesDatasetPaths(dataset_root, parquet_files[-1], g1_csv_root)

