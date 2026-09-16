#!/usr/bin/env python3
"""Train the task-only fixed-state PPO baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path

import numpy as np

from apex_getup.rl import PPOAgent, PPOConfig, PPOTrainer
from apex_getup.task import GetUpTask, GetUpTaskConfig, evaluate_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ppo/milestone3"))
    parser.add_argument("--checkpoint-frequency", type=int, default=10_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--episode-duration", type=float, default=10.0)
    return parser.parse_args()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _save_curves(rows: list[dict[str, float]], output_dir: Path) -> None:
    if not rows:
        return
    csv_path = output_dir / "learning_curves.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/apex-getup-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["environment_steps"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(steps, [row["episode_return"] for row in rows])
    axes[0, 0].set_ylabel("mean episode return")
    axes[0, 1].plot(steps, [row["maximum_pelvis_height"] for row in rows])
    axes[0, 1].set_ylabel("maximum pelvis height (m)")
    axes[1, 0].plot(steps, [row["success_rate"] for row in rows])
    axes[1, 0].set_ylabel("success rate")
    axes[1, 1].plot(steps, [row["actor_loss"] for row in rows], label="actor")
    axes[1, 1].plot(steps, [row["critic_loss"] for row in rows], label="critic")
    axes[1, 1].set_ylabel("loss")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)
    figure.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.total_steps <= 0 or args.checkpoint_frequency <= 0:
        raise ValueError("step counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_config = GetUpTaskConfig(episode_duration=args.episode_duration)
    ppo_config = PPOConfig(
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
    )
    task = GetUpTask(task_config, seed=args.seed)
    agent = PPOAgent(task.observation_size, task.action_size, ppo_config)
    trainer = PPOTrainer(task, agent, seed=args.seed)
    used_config = {
        "task": _jsonable(asdict(task_config)),
        "ppo": _jsonable(asdict(ppo_config)),
        "total_steps": args.total_steps,
        "device": args.device,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(used_config, indent=2), encoding="utf-8"
    )
    curves: list[dict[str, float]] = []
    environment_steps = 0
    next_checkpoint = args.checkpoint_frequency
    best_score = -np.inf
    all_episodes: list[dict[str, float]] = []
    while environment_steps < args.total_steps:
        rollout_size = min(ppo_config.rollout_steps, args.total_steps - environment_steps)
        buffer, episodes = trainer.collect_rollout(rollout_size)
        update = agent.update(buffer)
        environment_steps += rollout_size
        all_episodes.extend(episodes)
        recent = episodes or all_episodes[-1:]
        row = {
            "environment_steps": float(environment_steps),
            "episode_return": float(np.mean([item["episode_return"] for item in recent])),
            "success_rate": float(np.mean([item["success"] for item in recent])),
            "maximum_pelvis_height": float(
                np.mean([item["maximum_pelvis_height"] for item in recent])
            ),
            "final_pelvis_height": float(
                np.mean([item["final_pelvis_height"] for item in recent])
            ),
            "uprightness": float(np.mean([item["final_uprightness"] for item in recent])),
            "standing_bonus": float(np.mean([item["standing_bonus"] for item in recent])),
            "action_magnitude": float(np.mean([item["action_magnitude"] for item in recent])),
            "action_delta_magnitude": float(
                np.mean([item["action_delta_magnitude"] for item in recent])
            ),
            "mean_torque": float(np.mean([item["mean_torque"] for item in recent])),
            "torque_saturation": float(
                np.mean([item["torque_saturation"] for item in recent])
            ),
            **update,
        }
        curves.append(row)
        logging.info(
            "steps=%d return=%.1f pelvis_max=%.3f success=%.2f actor=%.4f critic=%.3f",
            environment_steps,
            row["episode_return"],
            row["maximum_pelvis_height"],
            row["success_rate"],
            row["actor_loss"],
            row["critic_loss"],
        )
        if environment_steps >= next_checkpoint or environment_steps == args.total_steps:
            agent.save(
                args.output_dir / f"checkpoint_{environment_steps:09d}.npz",
                metadata={"environment_steps": environment_steps},
            )
            next_checkpoint += args.checkpoint_frequency
            evaluation_task = GetUpTask(task_config, seed=args.seed)
            evaluation = evaluate_episode(
                evaluation_task, agent.deterministic_action, seed=args.seed
            )
            selection_score = (
                1_000_000.0 * evaluation.metrics["success"]
                + evaluation.metrics["episode_return"]
            )
            if selection_score > best_score:
                best_score = selection_score
                agent.save(
                    args.output_dir / "best.npz",
                    metadata={
                        "environment_steps": environment_steps,
                        "selection_metric": "success_then_episode_return",
                        "evaluation": evaluation.metrics,
                    },
                )
            (args.output_dir / "latest_evaluation.json").write_text(
                json.dumps(evaluation.metrics, indent=2), encoding="utf-8"
            )
        _save_curves(curves, args.output_dir)
    agent.save(
        args.output_dir / "final.npz",
        metadata={"environment_steps": environment_steps},
    )
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps({"curves": curves, "episodes": all_episodes}, indent=2),
        encoding="utf-8",
    )
    print(f"completed {environment_steps} environment steps in {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
