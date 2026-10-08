"""Stable-Baselines3 PPO adapter for the shared get-up task."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from apex_getup.apex import (
    DemonstrationActionPrior,
    FixedCoefficientPriorAdapter,
    PriorCoefficientSchedule,
    ResidualActionAdapter,
    compose_apex_action,
    compose_residual_joint_target,
)
from apex_getup.rl.config import PPOConfig
from apex_getup.rl.normalization import NormalizationConfig
from apex_getup.rl.trainer import EpisodeAccumulator
from apex_getup.task import (
    GetUpTask,
    ResetPerturbationConfig,
    evaluate_episode,
)
from apex_getup.training import (
    FRAME_ZERO_SELECTION_METRIC,
    TrainingMetrics,
    frame_zero_selection_key,
    is_better_frame_zero_evaluation,
)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class SB3GetUpEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium view of :class:`GetUpTask` with shared action composition."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: GetUpTask,
        *,
        control_mode: str,
        residual_scale: float,
        coefficient_schedule: PriorCoefficientSchedule,
    ) -> None:
        super().__init__()
        self.task = task
        self.control_mode = control_mode
        self.residual_scale = float(residual_scale)
        self.coefficient_schedule = coefficient_schedule
        self.global_environment_steps = 0
        self.prior = DemonstrationActionPrior(
            task.reset_trajectory,
            task.env,
            interpolate_reference=task.config.interpolate_reference,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(task.observation_size,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(task.action_size,),
            dtype=np.float32,
        )
        self.episode = EpisodeAccumulator()

    def set_global_environment_steps(self, steps: int) -> None:
        self.global_environment_steps = int(steps)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation = self.task.reset(seed=seed)
        self.episode = EpisodeAccumulator(
            initial_pelvis_height=self.task.initial_pelvis_height,
            initial_uprightness=self.task.initial_uprightness,
        )
        return observation.astype(np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        policy_action = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )
        if policy_action.shape != (self.task.action_size,):
            raise ValueError("SB3 action has the wrong shape")
        prior_action = self.prior.action_at_time(
            self.task.current_reference_time
        )
        physical_target = None
        reference_target = None
        if self.control_mode == "residual":
            coefficient = 1.0
            action_scale = self.residual_scale
            reference_target = self.prior.joint_target_at_time(
                self.task.current_reference_time
            )
            physical_target, clipped = compose_residual_joint_target(
                policy_action,
                reference_target,
                self.residual_scale,
                self.task.env.joint_limits,
            )
            executed_action, _ = self.task.env.joint_target_to_action(
                physical_target
            )
        else:
            coefficient = self.coefficient_schedule(
                self.global_environment_steps
            )
            action_scale = 1.0
            executed_action, clipped = compose_apex_action(
                policy_action, prior_action, coefficient
            )
        result = self.task.step(
            executed_action,
            policy_action=policy_action,
            physical_joint_target=physical_target,
        )
        self.episode.add(result.reward, result.info)
        self.episode.add_action_statistics(
            policy_action,
            prior_action,
            executed_action,
            coefficient,
            clipped,
            action_scale,
            reference_target,
        )
        info = dict(result.info)
        if result.terminated or result.truncated:
            info["episode_metrics"] = self.episode.finish(result.info)
        return (
            result.observation.astype(np.float32),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            info,
        )


class _ShapeEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal environment used only to attach loaded VecNormalize stats."""

    def __init__(self, observation_size: int, action_size: int) -> None:
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (observation_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (action_size,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        raise RuntimeError("shape-only environment cannot be stepped")


class SB3PolicyAdapter:
    """Expose an SB3 model through the deterministic policy API used by eval."""

    def __init__(self, model: PPO, normalizer: VecNormalize | None) -> None:
        self.model = model
        self.normalizer = normalizer
        self.observation_size = int(model.observation_space.shape[0])
        self.action_size = int(model.action_space.shape[0])
        self.normalization_state = None

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if self.normalizer is not None:
            value = self.normalizer.normalize_obs(value)
        action, _ = self.model.predict(value, deterministic=True)
        return np.asarray(action, dtype=np.float64)


def load_sb3_policy(
    checkpoint: Path | str,
    *,
    device: str = "auto",
    normalization_path: Path | str | None = None,
) -> tuple[SB3PolicyAdapter, dict[str, Any]]:
    source = Path(checkpoint)
    model = PPO.load(source, device=device)
    normalizer_source = (
        Path(normalization_path)
        if normalization_path is not None
        else source.with_name(f"{source.stem}_vecnormalize.pkl")
    )
    normalizer = None
    if normalizer_source.is_file():
        shape_env = DummyVecEnv(
            [lambda: _ShapeEnv(
                int(model.observation_space.shape[0]),
                int(model.action_space.shape[0]),
            )]
        )
        normalizer = VecNormalize.load(normalizer_source, shape_env)
        normalizer.training = False
        normalizer.norm_reward = False
    metadata_path = source.with_name(f"{source.stem}_metadata.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    return SB3PolicyAdapter(model, normalizer), metadata


def _evaluate(
    model: PPO,
    vec_normalize: VecNormalize,
    task: GetUpTask,
    residual_scale: float,
    control_mode: str,
    coefficient: float,
    *,
    seed: int = 0,
) -> dict[str, float]:
    prior = DemonstrationActionPrior(
        task.reset_trajectory,
        task.env,
        interpolate_reference=task.config.interpolate_reference,
    )
    adapter = (
        ResidualActionAdapter(prior, residual_scale)
        if control_mode == "residual"
        else FixedCoefficientPriorAdapter(prior, coefficient)
    )

    def policy(observation: np.ndarray) -> np.ndarray:
        normalized = vec_normalize.normalize_obs(
            np.asarray(observation, dtype=np.float32)
        )
        action, _ = model.predict(normalized, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    return evaluate_episode(
        task,
        policy,
        seed=seed,
        action_adapter=adapter
    ).metrics


class GetUpSB3Callback(BaseCallback):
    def __init__(
        self,
        *,
        output_dir: Path,
        task: GetUpTask,
        metrics: TrainingMetrics,
        checkpoint_frequency: int,
        evaluation_frequency: int,
        seed: int,
        control_mode: str,
        residual_scale: float,
        coefficient_schedule: PriorCoefficientSchedule,
    ) -> None:
        super().__init__(verbose=0)
        self.output_dir = output_dir
        self.task_template = task
        self.metrics = metrics
        self.checkpoint_frequency = checkpoint_frequency
        self.evaluation_frequency = evaluation_frequency
        self.seed = seed
        self.control_mode = control_mode
        self.residual_scale = residual_scale
        self.coefficient_schedule = coefficient_schedule
        self.next_checkpoint = checkpoint_frequency
        self.next_evaluation = evaluation_frequency
        self.best_evaluation: dict[str, float] | None = None
        self.completed_episodes: list[dict[str, float]] = []
        self.rollout_start = perf_counter()

    @property
    def vec_normalize(self) -> VecNormalize:
        env = self.model.get_env()
        if not isinstance(env, VecNormalize):
            raise TypeError("SB3 training requires VecNormalize")
        return env

    def _save(self, stem: str, metadata: dict[str, object]) -> None:
        self.model.save(self.output_dir / f"{stem}.zip")
        self.vec_normalize.save(self.output_dir / f"{stem}_vecnormalize.pkl")
        (self.output_dir / f"{stem}_metadata.json").write_text(
            json.dumps(_jsonable(metadata), indent=2), encoding="utf-8"
        )

    def _on_step(self) -> bool:
        self.training_env.env_method(
            "set_global_environment_steps", self.num_timesteps
        )
        for info in self.locals.get("infos", []):
            if "episode_metrics" in info:
                summary = dict(info["episode_metrics"])
                summary["environment_step"] = float(self.num_timesteps)
                self.completed_episodes.append(summary)
        if self.num_timesteps >= self.next_checkpoint:
            self._save(
                f"checkpoint_{self.num_timesteps:09d}",
                {"environment_steps": self.num_timesteps, "backend": "sb3"},
            )
            while self.next_checkpoint <= self.num_timesteps:
                self.next_checkpoint += self.checkpoint_frequency
        if self.num_timesteps >= self.next_evaluation:
            self._run_evaluation()
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += self.evaluation_frequency
        return True

    def _on_rollout_start(self) -> None:
        self.rollout_start = perf_counter()

    def _on_rollout_end(self) -> None:
        elapsed = max(perf_counter() - self.rollout_start, 1e-12)
        logger = self.model.logger.name_to_value
        update = {
            "actor_loss": logger.get("train/policy_gradient_loss"),
            "critic_loss": logger.get("train/value_loss"),
            "entropy": (
                None
                if logger.get("train/entropy_loss") is None
                else -float(logger["train/entropy_loss"])
            ),
            "approximate_kl": logger.get("train/approx_kl"),
            "clip_fraction": logger.get("train/clip_fraction"),
            "action_std": logger.get("train/std"),
            "environment_steps_per_second": (
                self.model.n_steps * self.training_env.num_envs / elapsed
            ),
        }
        self.metrics.record_update(
            self.num_timesteps, self.completed_episodes, update
        )
        self.completed_episodes = []

    def _run_evaluation(self) -> None:
        fixed_config = replace(
            self.task_template.config,
            reset_perturbation=ResetPerturbationConfig(),
        )
        fixed_task = GetUpTask(fixed_config, seed=self.seed)
        coefficient = self.coefficient_schedule(self.num_timesteps)
        frame_zero = _evaluate(
            self.model,
            self.vec_normalize,
            fixed_task,
            self.residual_scale,
            self.control_mode,
            coefficient if self.control_mode != "residual" else 1.0,
            seed=self.seed,
        )
        self.metrics.record_evaluation(self.num_timesteps, frame_zero)
        selection = frame_zero
        improved = is_better_frame_zero_evaluation(
            selection, self.best_evaluation
        )
        selection_metric = FRAME_ZERO_SELECTION_METRIC
        selection_key = frame_zero_selection_key(selection)
        if improved:
            self.best_evaluation = dict(selection)
            self._save(
                "best",
                {
                    "environment_steps": self.num_timesteps,
                    "backend": "sb3",
                    "selection_metric": selection_metric,
                    "selection_key": list(selection_key),
                    "frame_zero_evaluation": frame_zero,
                },
            )
        payload = {
            "backend": "sb3",
            "environment_steps": self.num_timesteps,
            "best_checkpoint_updated": improved,
            "selection_metric": selection_metric,
            "selection_key": list(selection_key),
            "fixed": {"controlled": frame_zero},
        }
        (self.output_dir / "latest_evaluation.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logging.info(
            "SB3 frame-zero eval steps=%d success=%.0f height=%.3f upright=%.3f",
            self.num_timesteps,
            frame_zero["success"],
            frame_zero["final_pelvis_height"],
            frame_zero["final_uprightness"],
        )


def train_sb3(
    *,
    task: GetUpTask,
    ppo_config: PPOConfig,
    normalization_config: NormalizationConfig,
    output_dir: Path,
    total_steps: int,
    checkpoint_frequency: int,
    evaluation_frequency: int,
    smoothing_window: int,
    num_envs: int,
    seed: int,
    control_mode: str,
    residual_scale: float,
    coefficient_schedule: PriorCoefficientSchedule,
    source_yaml: Path | None,
) -> None:
    """Train SB3 PPO while retaining the project's task/control semantics."""
    if ppo_config.backend != "sb3":
        raise ValueError("train_sb3 requires ppo.backend=sb3")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_factories: list[Callable[[], gym.Env]] = []
    for index in range(num_envs):
        current = task if index == 0 else GetUpTask(task.config, seed=seed + index)
        task_factories.append(
            lambda current=current: SB3GetUpEnv(
                current,
                control_mode=control_mode,
                residual_scale=residual_scale,
                coefficient_schedule=coefficient_schedule,
            )
        )
    vector_env = DummyVecEnv(task_factories)
    normalized_env = VecNormalize(
        vector_env,
        training=True,
        norm_obs=normalization_config.normalize_observations,
        norm_reward=normalization_config.normalize_rewards,
        clip_obs=normalization_config.observation_clip,
        clip_reward=normalization_config.reward_clip,
        gamma=ppo_config.gamma,
        epsilon=normalization_config.epsilon,
    )
    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {
            "pi": list(ppo_config.actor_hidden_dimensions),
            "vf": list(ppo_config.critic_hidden_dimensions),
        },
        "log_std_init": float(np.log(ppo_config.initial_action_std)),
    }
    model = PPO(
        "MlpPolicy",
        normalized_env,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.rollout_steps,
        batch_size=ppo_config.minibatch_size,
        n_epochs=ppo_config.update_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        ent_coef=ppo_config.entropy_coefficient,
        vf_coef=ppo_config.value_coefficient,
        max_grad_norm=ppo_config.maximum_gradient_norm,
        target_kl=ppo_config.target_kl,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=ppo_config.device,
        verbose=1,
    )
    used_config = {
        "backend": "sb3",
        "task": _jsonable(asdict(task.config)),
        "ppo": _jsonable(asdict(ppo_config)),
        "normalization": _jsonable(asdict(normalization_config)),
        "control_mode": control_mode,
        "residual_scale": residual_scale,
        "total_steps": total_steps,
        "checkpoint_frequency": checkpoint_frequency,
        "evaluation_frequency": evaluation_frequency,
        "training": {"num_envs": num_envs},
        "source_yaml": None if source_yaml is None else str(source_yaml),
    }
    (output_dir / "config.json").write_text(
        json.dumps(used_config, indent=2), encoding="utf-8"
    )
    metrics = TrainingMetrics(output_dir)
    callback = GetUpSB3Callback(
        output_dir=output_dir,
        task=task,
        metrics=metrics,
        checkpoint_frequency=checkpoint_frequency,
        evaluation_frequency=evaluation_frequency,
        seed=seed,
        control_mode=control_mode,
        residual_scale=residual_scale,
        coefficient_schedule=coefficient_schedule,
    )
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        callback._save(
            "final",
            {"environment_steps": model.num_timesteps, "backend": "sb3"},
        )
    finally:
        metrics.finish(smoothing_window=smoothing_window)
        normalized_env.close()
    print(f"completed {model.num_timesteps} SB3 PPO environment steps in {output_dir}")
