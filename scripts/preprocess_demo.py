#!/usr/bin/env python3
"""Load and resample one BONES G1 trajectory into a deterministic NPZ schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from apex_getup.demonstrations import discover_bones_dataset, load_motion
from apex_getup.env import G1EnvConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--control-frequency", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    paths = discover_bones_dataset(args.dataset_root)
    trajectory = load_motion(args.motion, paths)
    config = G1EnvConfig(control_frequency=args.control_frequency)
    config.validate()
    processed = trajectory.resample(config.control_timestep)
    output = args.output or Path("datasets/processed/getup") / f"{processed.motion_id}.npz"
    processed.save(output, control_frequency=args.control_frequency)
    print(
        f"saved {len(processed.time)} samples at {args.control_frequency:g} Hz "
        f"({processed.duration:.3f} s) to {output}"
    )


if __name__ == "__main__":
    main()

