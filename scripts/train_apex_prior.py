#!/usr/bin/env python3
"""Train PPO with either the decaying APEX prior or a fixed BONES residual."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from apex_getup.apex import (
    FixedCoefficientPriorAdapter,
    PriorCoefficientSchedule,
    PriorScheduleConfig,
)
from apex_getup.control import ResidualActionAdapter
from apex_getup.demonstrations import DemonstrationActionPrior
from apex_getup.rl import PPOAgent, PPOConfig, PPOTrainer, VecNormalizer
from apex_getup.task import (
    GetUpTask, GetUpTaskConfig, ObservationConfig, ResetPerturbationConfig, evaluate_episode,
)
from apex_getup.training import (
    FRAME_ZERO_SELECTION_METRIC,
    TrainingMetrics,
    TrainingRunConfig,
    frame_zero_selection_key,
    is_better_frame_zero_evaluation,
    load_training_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML training configuration; explicit CLI options override it",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-frequency", type=int)
    parser.add_argument("--evaluation-frequency", type=int)
    parser.add_argument(
        "--warm-start", type=Path,
        help="initialize weights, optimizer, and normalization from a .pt checkpoint",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--initial-action-std", type=float)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument(
        "--ppo-backend",
        choices=("native", "sb3"),
        help="PPO implementation (default: value from YAML, normally native)",
    )
    parser.add_argument("--episode-duration", type=float)
    parser.add_argument("--prior-lambda", type=float)
    parser.add_argument("--prior-k", type=float)
    parser.add_argument("--fixed-prior-coefficient", type=float)
    parser.add_argument(
        "--control-mode",
        choices=("decaying-prior", "residual"),
    )
    parser.add_argument(
        "--observation-mode",
        choices=("state_only", "residual_reference", "paper_reference"),
        help=(
            "actor observation; defaults to residual_reference for residual "
            "control and state_only for decaying-prior control"
        ),
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        help="physical residual authority in radians (paper value: 0.25)",
    )
    parser.add_argument("--smoothing-window", type=int)
    return parser.parse_args()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    base = (
        TrainingRunConfig()
        if args.config is None
        else load_training_config(args.config)
    )
    seed = getattr(args, "seed", base.seed)
    control_mode = getattr(args, "control_mode", base.control_mode)
    residual_scale = getattr(args, "residual_scale", base.residual_scale)
    output_dir = getattr(args, "output_dir", base.output_dir)
    total_steps = getattr(args, "total_steps", base.total_steps)
    checkpoint_frequency = getattr(
        args, "checkpoint_frequency", base.checkpoint_frequency
    )
    evaluation_frequency = getattr(
        args, "evaluation_frequency", base.evaluation_frequency
    )
    warm_start_checkpoint = getattr(
        args, "warm_start", base.warm_start_checkpoint
    )
    smoothing_window = getattr(args, "smoothing_window", base.smoothing_window)
    num_envs = getattr(args, "num_envs", base.num_envs)
    observation_mode = getattr(args, "observation_mode", None)
    if observation_mode is None:
        observation_mode = (
            base.task.observation.mode
            if args.config is not None
            else (
                "residual_reference"
                if control_mode == "residual"
                else "state_only"
            )
        )
    task_config = replace(
        base.task,
        episode_duration=getattr(
            args, "episode_duration", base.task.episode_duration
        ),
        observation=replace(base.task.observation, mode=observation_mode),
    )
    ppo_config = replace(
        base.ppo,
        backend=getattr(args, "ppo_backend", base.ppo.backend),
        seed=seed,
        device=getattr(args, "device", base.ppo.device),
        rollout_steps=getattr(args, "rollout_steps", base.ppo.rollout_steps),
        minibatch_size=getattr(
            args, "minibatch_size", base.ppo.minibatch_size
        ),
        update_epochs=getattr(args, "update_epochs", base.ppo.update_epochs),
        learning_rate=getattr(args, "learning_rate", base.ppo.learning_rate),
        gamma=getattr(args, "gamma", base.ppo.gamma),
        gae_lambda=getattr(args, "gae_lambda", base.ppo.gae_lambda),
        clip_range=getattr(args, "clip_range", base.ppo.clip_range),
        initial_action_std=getattr(
            args, "initial_action_std", base.ppo.initial_action_std
        ),
        target_kl=getattr(args, "target_kl", base.ppo.target_kl),
    )
    schedule_config = replace(
        base.prior_schedule,
        decay_lambda=getattr(
            args, "prior_lambda", base.prior_schedule.decay_lambda
        ),
        decay_steps=getattr(args, "prior_k", base.prior_schedule.decay_steps),
        fixed_coefficient=getattr(
            args,
            "fixed_prior_coefficient",
            base.prior_schedule.fixed_coefficient,
        ),
    )
    if (
        total_steps <= 0
        or checkpoint_frequency <= 0
        or evaluation_frequency <= 0
        or smoothing_window <= 0
        or num_envs <= 0
    ):
        raise ValueError("step counts must be positive")
    task_config.validate()
    ppo_config.validate()
    schedule_config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(residual_scale) or residual_scale < 0:
        raise ValueError("residual_scale must be non-negative and finite")
    schedule = PriorCoefficientSchedule(schedule_config)
    task = GetUpTask(task_config, seed=seed)
    prior = DemonstrationActionPrior(
        task.reset_trajectory,
        task.env,
        interpolate_reference=task.config.interpolate_reference,
    )
    if ppo_config.backend == "sb3":
        if warm_start_checkpoint is not None:
            raise ValueError(
                "SB3 warm starts are not yet supported; start a new SB3 run"
            )
        from apex_getup.rl.sb3_backend import train_sb3

        train_sb3(
            task=task,
            ppo_config=ppo_config,
            normalization_config=base.normalization,
            output_dir=output_dir,
            total_steps=total_steps,
            checkpoint_frequency=checkpoint_frequency,
            evaluation_frequency=evaluation_frequency,
            smoothing_window=smoothing_window,
            num_envs=num_envs,
            seed=seed,
            control_mode=control_mode,
            residual_scale=residual_scale,
            coefficient_schedule=schedule,
            source_yaml=args.config,
        )
        return
    agent = PPOAgent(task.observation_size, task.action_size, ppo_config)
    warm_start_metadata: dict[str, object] | None = None
    if warm_start_checkpoint is not None:
        if not warm_start_checkpoint.is_file():
            raise FileNotFoundError(
                f"warm-start checkpoint not found: {warm_start_checkpoint}"
            )
        warm_start_metadata = agent.warm_start(warm_start_checkpoint)
    trainer_kwargs = (
        {"residual_scale": residual_scale}
        if control_mode == "residual"
        else {"coefficient_schedule": schedule}
    )
    if agent.normalization_state is None:
        normalizer = VecNormalizer(
            task.observation_size,
            num_envs,
            ppo_config.gamma,
            base.normalization,
            training=True,
        )
    else:
        normalizer = VecNormalizer.from_state_dict(
            agent.normalization_state, training=True, num_envs=num_envs
        )
        if normalizer.observation_size != task.observation_size:
            raise ValueError("warm-start normalization observation size does not match")
        if not np.isclose(normalizer.gamma, ppo_config.gamma):
            raise ValueError("warm-start normalization gamma does not match PPO gamma")
        # Running statistics transfer, but per-environment returns belong to the
        # old in-progress episodes and must not leak into fresh resets.
        normalizer.discounted_returns.fill(0.0)
    trainer = PPOTrainer(
        task,
        agent,
        seed=seed,
        action_prior=prior,
        num_envs=num_envs,
        normalizer=normalizer,
        **trainer_kwargs,
    )
    training_metrics = TrainingMetrics(output_dir)
    used_config = {
        # Persist the resolved horizon when episode_duration was derived from
        # the reference duration plus its terminal hold.
        "task": _jsonable(asdict(task.config)),
        "ppo": _jsonable(asdict(ppo_config)),
        "prior_schedule": (
            None
            if control_mode == "residual"
            else _jsonable(asdict(schedule_config))
        ),
        "control_mode": control_mode,
        "observation_mode": observation_mode,
        "residual_scale": residual_scale,
        "residual_control": _jsonable(asdict(residual_config)),
        "total_steps": total_steps,
        "checkpoint_frequency": checkpoint_frequency,
        "evaluation_frequency": evaluation_frequency,
        "warm_start": (
            None if warm_start_checkpoint is None else str(warm_start_checkpoint)
        ),
        "warm_start_metadata": warm_start_metadata,
        "warm_start_resets_environment_steps": True,
        "device": ppo_config.device,
        "smoothing_window": smoothing_window,
        "training": {"num_envs": num_envs},
        "normalization": _jsonable(asdict(normalizer.config)),
        "source_yaml": None if args.config is None else str(args.config),
    }
    (output_dir / "config.json").write_text(
        json.dumps(used_config, indent=2), encoding="utf-8"
    )
    environment_steps = 0
    next_checkpoint = checkpoint_frequency
    next_evaluation = evaluation_frequency
    best_evaluation: dict[str, float] | None = None
    while environment_steps < total_steps:
        remaining_steps = total_steps - environment_steps
        rollout_size = min(
            ppo_config.rollout_steps,
            int(np.ceil(remaining_steps / num_envs)),
        )
        buffer, episodes = trainer.collect_rollout(rollout_size)
        update_start = perf_counter()
        update = agent.update(buffer)
        update["ppo_update_time"] = perf_counter() - update_start
        update.update(trainer.last_rollout_metrics)
        environment_steps += buffer.transition_count
        row = training_metrics.record_update(environment_steps, episodes, update)
        logging.info(
            "steps=%d c=%.3f return=%.1f pelvis=%.3f upright=%.3f "
            "success=%.2f clip=%.3f throughput=%.0f steps/s",
            environment_steps,
            1.0 if control_mode == "residual" else schedule(environment_steps),
            row.get("episode_return", np.nan),
            row.get("maximum_pelvis_height", np.nan),
            row.get("maximum_uprightness", np.nan),
            row.get("success_rate", np.nan),
            row.get("executed_action_clipping_fraction", np.nan),
            row.get("environment_steps_per_second", np.nan),
        )
        if environment_steps >= next_checkpoint or environment_steps == total_steps:
            agent.save(
                output_dir / f"checkpoint_{environment_steps:09d}.pt",
                metadata={"environment_steps": environment_steps},
                normalization_state=trainer.normalizer.state_dict(),
            )
            while next_checkpoint <= environment_steps:
                next_checkpoint += checkpoint_frequency
        if environment_steps >= next_evaluation or environment_steps == total_steps:
            coefficient = schedule(environment_steps)
            fixed_task_config = replace(
                task_config, reset_perturbation=ResetPerturbationConfig()
            )
            guided_task = GetUpTask(fixed_task_config, seed=seed)
            guided_prior = DemonstrationActionPrior(
                guided_task.reset_trajectory,
                guided_task.env,
                interpolate_reference=guided_task.config.interpolate_reference,
            )
            guided_adapter = (
                ResidualActionAdapter(guided_prior, residual_scale)
                if control_mode == "residual"
                else FixedCoefficientPriorAdapter(guided_prior, coefficient)
            )
            guided = evaluate_episode(
                guided_task,
                agent.deterministic_action,
                seed=seed,
                action_adapter=guided_adapter,
                observation_transform=lambda observation: trainer.normalizer.normalize_observations(
                    observation, update=False
                ),
            )
            prior_free_task = GetUpTask(fixed_task_config, seed=seed)
            prior_free_prior = DemonstrationActionPrior(
                prior_free_task.reset_trajectory,
                prior_free_task.env,
                interpolate_reference=prior_free_task.config.interpolate_reference,
            )
            baseline_adapter = (
                ResidualActionAdapter(prior_free_prior, 0.0)
                if control_mode == "residual"
                else FixedCoefficientPriorAdapter(prior_free_prior, 0.0)
            )
            prior_free = evaluate_episode(
                prior_free_task,
                agent.deterministic_action,
                seed=seed,
                action_adapter=baseline_adapter,
                observation_transform=lambda observation: trainer.normalizer.normalize_observations(
                    observation, update=False
                ),
            )
            frame_zero_selection = (
                guided if control_mode == "residual" else prior_free
            )
            training_metrics.record_evaluation(
                environment_steps, frame_zero_selection.metrics
            )

            selection_metrics = frame_zero_selection.metrics
            selection_metric = FRAME_ZERO_SELECTION_METRIC
            selection_key = frame_zero_selection_key(selection_metrics)
            improved = is_better_frame_zero_evaluation(
                selection_metrics, best_evaluation
            )
            selection_mode = (
                "residual_frame_zero"
                if control_mode == "residual"
                else "prior_free_frame_zero"
            )
            if improved:
                best_evaluation = dict(selection_metrics)
                agent.save(
                    output_dir / "best.pt",
                    metadata={
                        "environment_steps": environment_steps,
                        "selection_metric": selection_metric,
                        "selection_key": list(selection_key),
                        "selection_mode": selection_mode,
                        "frame_zero_evaluation": frame_zero_selection.metrics,
                    },
                    normalization_state=trainer.normalizer.state_dict(),
                )
            evaluation_payload = {
                "control_mode": control_mode,
                "coefficient": (
                    1.0 if control_mode == "residual" else coefficient
                ),
                "residual_scale": residual_scale,
                "environment_steps": environment_steps,
                "best_checkpoint_updated": improved,
                "selection_metric": selection_metric,
                "selection_key": list(selection_key),
                "selection_mode": selection_mode,
                "fixed": {
                    "controlled": guided.metrics,
                    "baseline": prior_free.metrics,
                },
            }
            (output_dir / "latest_evaluation.json").write_text(
                json.dumps(evaluation_payload, indent=2), encoding="utf-8"
            )
            if improved:
                (output_dir / "best_evaluation.json").write_text(
                    json.dumps(
                        {
                            "environment_steps": environment_steps,
                            "selection_metric": selection_metric,
                            "selection_key": list(selection_key),
                            "selection_mode": selection_mode,
                            "metrics": selection_metrics,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            logging.info(
                "frame-zero eval steps=%d success=%.0f stable=%.2fs "
                "final_height=%.3f final_upright=%.3f",
                environment_steps,
                frame_zero_selection.metrics["success"],
                frame_zero_selection.metrics[
                    "maximum_consecutive_standing_duration"
                ],
                frame_zero_selection.metrics["final_pelvis_height"],
                frame_zero_selection.metrics["final_uprightness"],
            )
            while next_evaluation <= environment_steps:
                next_evaluation += evaluation_frequency
    agent.save(
        output_dir / "final.pt",
        metadata={"environment_steps": environment_steps},
        normalization_state=trainer.normalizer.state_dict(),
    )
    training_metrics.finish(smoothing_window=smoothing_window)
    print(
        f"completed {environment_steps} {control_mode} environment steps "
        f"in {output_dir}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
