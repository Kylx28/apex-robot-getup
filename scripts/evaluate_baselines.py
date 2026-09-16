#!/usr/bin/env python3
"""Evaluate fixed-action get-up baselines and record BONES replay context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from apex_getup.task import GetUpTask, GetUpTaskConfig, evaluate_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baselines/milestone3.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GetUpTaskConfig(episode_duration=args.episode_duration)
    nominal_task = GetUpTask(config, seed=0)
    nominal_action, violation = nominal_task.env.joint_target_to_action(
        nominal_task.env.nominal_pose
    )
    if np.any(violation):
        raise RuntimeError("nominal pose is outside the action space")
    nominal = evaluate_episode(
        nominal_task, lambda observation: nominal_action, seed=0
    ).metrics
    center_task = GetUpTask(config, seed=0)
    center = evaluate_episode(
        center_task, lambda observation: np.zeros(29), seed=0
    ).metrics
    calibration_path = Path("artifacts/calibration/gain_sweep.json")
    bones_context: dict[str, object] | None = None
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        bones_context = calibration.get("selected")
    payload = {
        "nominal_pose_controller": nominal,
        "zero_action_center_controller": center,
        "actions_are_equivalent": bool(np.allclose(nominal_action, 0.0)),
        "note": (
            "Zero action is the physical joint-range center; it is not the nominal pose."
        ),
        "bones_pd_replay_context": bones_context,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
