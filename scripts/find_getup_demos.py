#!/usr/bin/env python3
"""Rank likely get-up demonstrations from BONES-SEED metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from apex_getup.demonstrations.paths import discover_bones_dataset
from apex_getup.demonstrations.search import find_getup_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-mirrors", action="store_true")
    args = parser.parse_args()
    paths = discover_bones_dataset(args.dataset_root)
    candidates = find_getup_candidates(
        paths, include_mirrors=args.include_mirrors, limit=args.limit
    )
    print(f"Found ranked candidates in {paths.metadata} (showing {len(candidates)}):")
    for candidate in candidates:
        print(
            f"[{candidate.score:3d}] {candidate.motion_id} ({candidate.frames} frames)\n"
            f"      {candidate.description}\n"
            f"      movement={candidate.movement_type}; pose={candidate.body_position}; props={candidate.props}\n"
            f"      {candidate.path}"
        )
    print("\nNo candidate was selected automatically; pass an ID above to replay_demo.py.")


if __name__ == "__main__":
    main()
