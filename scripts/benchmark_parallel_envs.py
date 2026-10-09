#!/usr/bin/env python3
"""Benchmark batched policy inference plus independent MuJoCo environments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from apex_getup.demonstrations import DemonstrationActionPrior
from apex_getup.rl import PPOAgent, PPOTrainer, VecNormalizer
from apex_getup.task import GetUpTask
from apex_getup.training import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/residual_policy_milestone_1.yaml"),
    )
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=128,
        help="control steps per environment and benchmark",
    )
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/parallel_envs.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_steps <= 0 or args.warmup_steps < 0:
        raise ValueError("rollout steps must be positive and warmup non-negative")
    if any(count <= 0 for count in args.num_envs):
        raise ValueError("environment counts must be positive")
    config = load_training_config(args.config)
    ppo_config = replace(
        config.ppo,
        backend="native",
        device=args.device,
        rollout_steps=args.rollout_steps,
        minibatch_size=config.ppo.minibatch_size,
        update_epochs=1,
    )
    results: list[dict[str, float | int | str]] = []
    for count in args.num_envs:
        task = GetUpTask(config.task, seed=config.seed)
        prior = DemonstrationActionPrior(task.reset_trajectory, task.env)
        agent = PPOAgent(task.observation_size, task.action_size, ppo_config)
        trainer = PPOTrainer(
            task,
            agent,
            seed=config.seed,
            action_prior=prior,
            residual_scale=config.residual_scale,
            num_envs=count,
            normalizer=VecNormalizer(
                task.observation_size,
                count,
                ppo_config.gamma,
                config.normalization,
                training=True,
            ),
        )
        if args.warmup_steps:
            trainer.collect_rollout(args.warmup_steps)
        buffer, _ = trainer.collect_rollout(args.rollout_steps)
        update_start = perf_counter()
        agent.update(buffer)
        metrics = {
            **trainer.last_rollout_metrics,
            "ppo_update_time": perf_counter() - update_start,
        }
        row: dict[str, float | int | str] = {
            "num_envs": count,
            "device": str(agent.device),
            "rollout_steps_per_env": args.rollout_steps,
            "transitions": buffer.transition_count,
            **metrics,
        }
        results.append(row)
        print(
            f"num_envs={count:>2}  "
            f"throughput={metrics['environment_steps_per_second']:.1f} steps/s  "
            f"inference={metrics['policy_inference_time']:.3f}s  "
            f"simulation={metrics['simulation_rollout_time']:.3f}s  "
            f"ppo_update={metrics['ppo_update_time']:.3f}s"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
