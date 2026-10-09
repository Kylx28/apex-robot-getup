#!/usr/bin/env python3
"""Evaluate fixed-reference residual PPO against pure BONES replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from apex_getup.control import ResidualActionAdapter
from apex_getup.demonstrations import DemonstrationActionPrior
from apex_getup.apex.diagnostics import (
    per_joint_residual_magnitude,
    save_residual_diagnostics,
)
from apex_getup.rl import PPOAgent, VecNormalizer
from apex_getup.rl.sb3_backend import SB3PolicyAdapter, load_sb3_policy
from apex_getup.task import (
    GetUpTask, GetUpTaskConfig, ObservationConfig,
    RESIDUAL_REFERENCE_OBSERVATION_SIZE, STATE_ONLY_OBSERVATION_SIZE,
    PAPER_REFERENCE_OBSERVATION_SIZE,
)
from apex_getup.training import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="training YAML supplying reference path and simulator timing",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.25,
        help="physical residual authority in radians",
    )
    parser.add_argument(
        "--mode", choices=("residual", "reference-only", "both"), default="both"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=None,
        help="override the config episode duration (default: config value)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--observation-mode",
        choices=("state_only", "residual_reference", "paper_reference"),
        default=None,
        help="defaults to the mode implied by the checkpoint input size",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render the first episode of each selected mode in a live MuJoCo viewer",
    )
    return parser.parse_args()


def evaluate_mode(
    agent: PPOAgent | SB3PolicyAdapter,
    scale: float,
    episodes: int,
    task_config: GetUpTaskConfig,
    *,
    render: bool = False,
    normalizer: VecNormalizer | None = None,
) -> tuple[dict[str, object], object]:
    episode_metrics: list[dict[str, float]] = []
    representative = None
    for seed in range(episodes):
        task = GetUpTask(task_config, seed=seed)
        prior = DemonstrationActionPrior(
            task.reset_trajectory,
            task.env,
            interpolate_reference=task.config.interpolate_reference,
        )
        viewer = None
        callback = None
        if render and seed == 0:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(task.env.model, task.env.data)
            wall_start = time.monotonic()

            def viewer_callback(current_task: GetUpTask) -> None:
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep(
                        max(
                            0.0,
                            wall_start
                            + current_task.env.simulation_time
                            - time.monotonic(),
                        )
                    )

            callback = viewer_callback
        try:
            result = evaluate_episode(
                task,
            agent.deterministic_action,
                seed=seed,
                step_callback=callback,
                action_adapter=ResidualActionAdapter(prior, scale),
                observation_transform=(
                    None
                    if normalizer is None
                    else lambda observation: normalizer.normalize_observations(
                        observation, update=False
                    )
                )
        )
        finally:
            if viewer is not None:
                viewer.close()
        representative = representative or result
        episode_metrics.append(result.metrics)
    mean = {
        key: (
            None
            if all(np.isnan(row[key]) for row in episode_metrics)
            else float(np.nanmean([row[key] for row in episode_metrics]))
        )
        for key in episode_metrics[0]
    }
    mean["success_rate"] = mean.pop("success")
    assert representative is not None
    payload: dict[str, object] = {
        "residual_scale": scale,
        "mean": mean,
        "episodes": episode_metrics,
    }
    if scale > 0:
        payload["per_joint_residual_magnitude_rad"] = (
            per_joint_residual_magnitude(representative.traces)
        )
    return payload, representative


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not np.isfinite(args.residual_scale) or args.residual_scale < 0:
        raise ValueError("--residual-scale must be non-negative and finite")
    if args.checkpoint.suffix.lower() == ".zip":
        agent, metadata = load_sb3_policy(
            args.checkpoint, device=args.device
        )
    else:
        agent, metadata = PPOAgent.load(args.checkpoint, device=args.device)
    implied_modes = {
        STATE_ONLY_OBSERVATION_SIZE: "state_only",
        RESIDUAL_REFERENCE_OBSERVATION_SIZE: "residual_reference",
        PAPER_REFERENCE_OBSERVATION_SIZE: "paper_reference",
    }
    if agent.observation_size not in implied_modes:
        raise ValueError(
            f"unsupported checkpoint observation size: {agent.observation_size}"
        )
    observation_mode = args.observation_mode or implied_modes[agent.observation_size]
    expected_size = {
        "state_only": STATE_ONLY_OBSERVATION_SIZE,
        "residual_reference": RESIDUAL_REFERENCE_OBSERVATION_SIZE,
        "paper_reference": PAPER_REFERENCE_OBSERVATION_SIZE,
    }[observation_mode]
    if agent.observation_size != expected_size:
        raise ValueError(
            f"checkpoint expects {agent.observation_size} observations but "
            f"{observation_mode} provides {expected_size}"
        )
    payload: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": metadata,
        "observation_mode": observation_mode,
    }
    normalizer = (
        None
        if agent.normalization_state is None
        else VecNormalizer.from_state_dict(
            agent.normalization_state, training=False, num_envs=1
        )
    )
    base_task = (
        GetUpTaskConfig()
        if args.config is None
        else load_training_config(args.config).task
    )
    episode_duration = (
        base_task.episode_duration
        if args.episode_duration is None
        else args.episode_duration
    )
    task_config = replace(
        base_task,
        episode_duration=episode_duration,
        observation=ObservationConfig(mode=observation_mode),
    )
    representative = None
    evaluations: dict[str, object] = {}
    reset_results: dict[str, object] = {}
    if args.mode in {"residual", "both"}:
        reset_results["residual"], result = evaluate_mode(
                agent,
            args.residual_scale,
                args.episodes,
            task_config,
            render=args.render,
            normalizer=normalizer,
        )
        representative = representative or result
        if args.mode in {"reference-only", "both"}:
            reset_results["reference_only"], result = evaluate_mode(
                agent,
            0.0,
                args.episodes,
                task_config,
                render=args.render,
                normalizer=normalizer,
        )
            representative = representative or result
    evaluations["fixed"] = reset_results
    payload["evaluations"] = evaluations
    if args.plot is not None and representative is not None:
        save_residual_diagnostics(
            representative.traces, args.plot, base_task.control_frequency ** -1
        )
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
