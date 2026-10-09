#!/usr/bin/env python3
"""Regenerate plots for one or more training runs and compare common metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from apex_getup.training import generate_training_plots, load_metrics, plot_run_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--smoothing-window", type=int, default=10)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/training/comparison")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoothing_window <= 0:
        raise ValueError("--smoothing-window must be positive")
    readable: list[Path] = []
    for run in args.runs:
        records = load_metrics(run)
        if not records:
            print(f"skipping {run}: no metrics/scalars.csv records")
            continue
        paths = generate_training_plots(
            run, smoothing_window=args.smoothing_window
        )
        print(f"generated {len(paths)} plots in {run / 'plots'}")
        readable.append(run)
    if len(readable) >= 2:
        comparison = plot_run_comparison(
            readable,
            args.output_dir,
            smoothing_window=args.smoothing_window,
        )
        if comparison is None:
            print("no requested comparison metrics were common to every run")
        else:
            print(f"generated comparison plot at {comparison}")


if __name__ == "__main__":
    main()
