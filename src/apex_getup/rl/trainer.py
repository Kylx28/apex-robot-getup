"""Rollout collection kept separate from the PPO update."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from apex_getup.apex import PriorCoefficientSchedule, compose_apex_action
from apex_getup.control import compose_residual_joint_target
from apex_getup.demonstrations import DemonstrationActionPrior
from apex_getup.rl.buffer import RolloutBuffer
from apex_getup.rl.ppo import PPOAgent
from apex_getup.rl.normalization import VecNormalizer
from apex_getup.task import GetUpTask


@dataclass
class EpisodeAccumulator:
    initial_pelvis_height: float = 0.0
    initial_uprightness: float = 0.0
    episode_return: float = 0.0
    maximum_pelvis_height: float = -np.inf
    maximum_uprightness: float = -np.inf
    stable_standing_indicator: float = 0.0
    action_magnitude: float = 0.0
    action_delta_magnitude: float = 0.0
    mean_torque: float = 0.0
    torque_saturation: float = 0.0
    prior_coefficient: float = 0.0
    policy_action_absolute: float = 0.0
    prior_action_absolute: float = 0.0
    executed_action_absolute: float = 0.0
    executed_action_clipping: float = 0.0
    scaled_policy_action_absolute: float = 0.0
    policy_to_prior_ratio: float = 0.0
    maximum_stable_standing_duration: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)
    steps: int = 0

    def add(self, reward: float, info: dict[str, object]) -> None:
        self.episode_return += reward
        self.maximum_pelvis_height = max(
            self.maximum_pelvis_height, float(info["pelvis_height"])
        )
        self.maximum_uprightness = max(
            self.maximum_uprightness, float(info["uprightness"])
        )
        self.stable_standing_indicator += float(info["stable_standing_indicator"])
        self.action_magnitude += float(info["action_magnitude"])
        self.action_delta_magnitude += float(info["action_delta_magnitude"])
        self.mean_torque += float(info["mean_torque"])
        self.torque_saturation += float(info["torque_saturation_fraction"])
        self.maximum_stable_standing_duration = max(
            self.maximum_stable_standing_duration,
            float(info.get("stable_success_duration", 0.0)),
        )
        components = info.get("reward_components", {})
        if isinstance(components, dict):
            for name, value in components.items():
                try:
                    scalar = float(value) if np.asarray(value).ndim == 0 else np.nan
                except (TypeError, ValueError):
                    scalar = np.nan
                if np.isfinite(scalar):
                    self.reward_components[name] = (
                        self.reward_components.get(name, 0.0) + scalar
                    )
        self.steps += 1

    def add_action_statistics(
        self,
        policy_action: np.ndarray,
        prior_action: np.ndarray,
        executed_action: np.ndarray,
        prior_coefficient: float,
        clipping_mask: np.ndarray,
        policy_action_scale: float,
        reference_joint_target: np.ndarray | None = None,
    ) -> None:
        self.prior_coefficient += prior_coefficient
        self.policy_action_absolute += float(np.mean(np.abs(policy_action)))
        self.prior_action_absolute += float(np.mean(np.abs(prior_action)))
        self.executed_action_absolute += float(np.mean(np.abs(executed_action)))
        self.executed_action_clipping += float(np.mean(clipping_mask))
        scaled_policy = policy_action_scale * policy_action
        scaled_absolute = float(np.mean(np.abs(scaled_policy)))
        prior_absolute = (
            float(np.mean(np.abs(prior_action)))
            if reference_joint_target is None
            else float(np.mean(np.abs(reference_joint_target)))
        )
        self.scaled_policy_action_absolute += scaled_absolute
        self.policy_to_prior_ratio += (
            scaled_absolute / prior_absolute if prior_absolute > 1e-12 else 0.0
        )

    def finish(self, info: dict[str, object]) -> dict[str, float]:
        denominator = max(1, self.steps)
        summary = {
            "episode_return": self.episode_return,
            "initial_pelvis_height": self.initial_pelvis_height,
            "initial_uprightness": self.initial_uprightness,
            "success": float(info["episode_success"]),
            "time_to_stand": np.nan
            if info["first_success_time"] is None
            else float(info["first_success_time"]),
            "stable_standing_duration": self.maximum_stable_standing_duration,
            "consecutive_standing_duration": self.maximum_stable_standing_duration,
            "stable_standing_indicator": (
                self.stable_standing_indicator / denominator
            ),
            "maximum_pelvis_height": self.maximum_pelvis_height,
            "final_pelvis_height": float(info["pelvis_height"]),
            "maximum_uprightness": self.maximum_uprightness,
            "final_uprightness": float(info["uprightness"]),
            "action_magnitude": self.action_magnitude / denominator,
            "action_delta_magnitude": self.action_delta_magnitude / denominator,
            "mean_torque": self.mean_torque / denominator,
            "torque_saturation": self.torque_saturation / denominator,
            "prior_coefficient": self.prior_coefficient / denominator,
            "mean_absolute_policy_action": self.policy_action_absolute / denominator,
            "mean_absolute_prior_action": self.prior_action_absolute / denominator,
            "mean_absolute_executed_action": self.executed_action_absolute / denominator,
            "executed_action_clipping_fraction": (
                self.executed_action_clipping / denominator
            ),
            "mean_absolute_scaled_policy_action": (
                self.scaled_policy_action_absolute / denominator
            ),
            "scaled_policy_to_prior_ratio": self.policy_to_prior_ratio / denominator,
        }
        summary.update(
            {f"reward/{name}": value for name, value in self.reward_components.items()}
        )
        return summary


class PPOTrainer:
    def __init__(
        self,
        task: GetUpTask,
        agent: PPOAgent,
        seed: int = 0,
        *,
        action_prior: DemonstrationActionPrior | None = None,
        coefficient_schedule: PriorCoefficientSchedule | None = None,
        residual_scale: float | None = None,
        num_envs: int = 1,
        normalizer: VecNormalizer | None = None,
    ):
        if action_prior is None and (
            coefficient_schedule is not None or residual_scale is not None
        ):
            raise ValueError(
                "a schedule or residual scale requires an action prior"
            )
        if action_prior is not None and coefficient_schedule is None and residual_scale is None:
            raise ValueError("an action prior requires a schedule or residual scale")
        if coefficient_schedule is not None and residual_scale is not None:
            raise ValueError("decaying-prior and residual modes are mutually exclusive")
        if residual_scale is not None and (
            not np.isfinite(residual_scale) or residual_scale < 0
        ):
            raise ValueError("residual_scale must be non-negative and finite")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.task = task
        self.tasks = [task] + [
            GetUpTask(task.config, seed=seed + index)
            for index in range(1, num_envs)
        ]
        self.num_envs = num_envs
        self.agent = agent
        if normalizer is not None and (
            normalizer.num_envs != num_envs
            or normalizer.observation_size != task.observation_size
        ):
            raise ValueError("normalizer dimensions must match the trainer")
        self.normalizer = normalizer
        self.action_prior = action_prior
        self.coefficient_schedule = coefficient_schedule
        self.residual_scale = residual_scale
        self.global_environment_steps = 0
        self.raw_observations = np.stack(
            [
                item.reset(seed=seed + index)
                for index, item in enumerate(self.tasks)
            ]
        )
        self.observations = (
            self.raw_observations.copy()
            if self.normalizer is None
            else self.normalizer.normalize_observations(self.raw_observations)
        )
        self.current_episodes = [
            self._new_episode_accumulator(item) for item in self.tasks
        ]
        # Compatibility aliases for existing single-environment diagnostics.
        self.observation = self.observations[0]
        self.current_episode = self.current_episodes[0]
        self.last_rollout_metrics: dict[str, float] = {}

    def _new_episode_accumulator(self, task: GetUpTask) -> EpisodeAccumulator:
        return EpisodeAccumulator(
            initial_pelvis_height=task.initial_pelvis_height,
            initial_uprightness=task.initial_uprightness,
        )

    def collect_rollout(
        self, steps: int | None = None
    ) -> tuple[RolloutBuffer, list[dict[str, float]]]:
        count = steps or self.agent.config.rollout_steps
        buffer = RolloutBuffer.create(
            count,
            self.task.observation_size,
            self.task.action_size,
            self.num_envs,
        )
        completed: list[dict[str, float]] = []
        last_dones = np.zeros(self.num_envs, dtype=np.bool_)
        inference_time = 0.0
        simulation_time = 0.0
        rollout_start = perf_counter()
        for _ in range(count):
            inference_start = perf_counter()
            policy_actions, latent_actions, log_probabilities, values = (
                self.agent.sample_actions(self.observations)
            )
            inference_time += perf_counter() - inference_start
            prior_actions = np.zeros_like(policy_actions)
            prior_coefficients = np.zeros(self.num_envs)
            policy_action_scales = np.ones(self.num_envs)
            physical_joint_targets: np.ndarray | None = None
            reference_joint_targets: np.ndarray | None = None
            if self.action_prior is None:
                executed_actions, clipping_masks = compose_apex_action(
                    policy_actions, prior_actions, 0.0
                )
            else:
                for index, current_task in enumerate(self.tasks):
                    prior_actions[index] = self.action_prior.action_at_time(
                        current_task.current_reference_time
                    )
                if self.residual_scale is not None:
                    prior_coefficients.fill(1.0)
                    policy_action_scales.fill(self.residual_scale)
                    physical_joint_targets = np.empty_like(policy_actions)
                    reference_joint_targets = np.empty_like(policy_actions)
                    executed_actions = np.empty_like(policy_actions)
                    clipping_masks = np.zeros_like(policy_actions, dtype=np.bool_)
                    for index, current_task in enumerate(self.tasks):
                        reference_joint_target = (
                            self.action_prior.joint_target_at_time(
                                current_task.current_reference_time
                            )
                        )
                        reference_joint_targets[index] = reference_joint_target
                        target, clipped = compose_residual_joint_target(
                            policy_actions[index],
                            reference_joint_target,
                            self.residual_scale,
                            current_task.env.joint_limits,
                        )
                        physical_joint_targets[index] = target
                        clipping_masks[index] = clipped
                        executed_actions[index], _ = (
                            current_task.env.joint_target_to_action(target)
                        )
                else:
                    assert self.coefficient_schedule is not None
                    coefficient = self.coefficient_schedule(
                        self.global_environment_steps
                    )
                    prior_coefficients.fill(coefficient)
                    executed_actions, clipping_masks = compose_apex_action(
                        policy_actions, prior_actions, coefficient
                    )
            raw_rewards = np.zeros(self.num_envs)
            dones = np.zeros(self.num_envs, dtype=np.bool_)
            timeout_observations: list[np.ndarray] = []
            timeout_indices: list[int] = []
            timeout_bootstrap_values = np.zeros(self.num_envs)
            next_raw_observations = np.empty_like(self.raw_observations)
            simulation_start = perf_counter()
            for index, current_task in enumerate(self.tasks):
                result = current_task.step(
                    executed_actions[index],
                    policy_action=policy_actions[index],
                    physical_joint_target=(
                        None
                        if physical_joint_targets is None
                        else physical_joint_targets[index]
                    ),
                )
                done = result.terminated or result.truncated
                raw_rewards[index] = result.reward
                dones[index] = done
                self.current_episodes[index].add(result.reward, result.info)
                self.current_episodes[index].add_action_statistics(
                    policy_actions[index],
                    prior_actions[index],
                    executed_actions[index],
                    prior_coefficients[index],
                    clipping_masks[index],
                    policy_action_scales[index],
                    None
                    if reference_joint_targets is None
                    else reference_joint_targets[index],
                )
                if result.truncated and not result.terminated:
                    timeout_indices.append(index)
                    timeout_observations.append(result.observation)
                if done:
                    summary = self.current_episodes[index].finish(result.info)
                    summary["environment_step"] = float(
                        self.global_environment_steps + index + 1
                    )
                    summary["environment_index"] = float(index)
                    completed.append(summary)
                    next_raw_observations[index] = current_task.reset()
                    self.current_episodes[index] = self._new_episode_accumulator(
                        current_task
                    )
                else:
                    next_raw_observations[index] = result.observation
            next_observations = (
                next_raw_observations.copy()
                if self.normalizer is None
                else self.normalizer.normalize_observations(next_raw_observations)
            )
            if timeout_observations:
                timeout_batch = np.stack(timeout_observations)
                if self.normalizer is not None:
                    timeout_batch = self.normalizer.normalize_observations(
                        timeout_batch, update=False
                    )
                timeout_bootstrap_values[timeout_indices] = self.agent.values(
                    timeout_batch
                )
            rewards = (
                raw_rewards
                if self.normalizer is None
                else self.normalizer.normalize_rewards(raw_rewards, dones)
            )
            simulation_time += perf_counter() - simulation_start
            buffer.add(
                self.observations,
                policy_actions,
                executed_actions,
                prior_actions,
                prior_coefficients,
                policy_action_scales,
                clipping_masks,
                latent_actions,
                log_probabilities,
                rewards,
                values,
                dones,
                timeout_bootstrap_values,
            )
            self.global_environment_steps += self.num_envs
            self.raw_observations = next_raw_observations
            self.observations = next_observations
            last_dones = dones
        next_values = self.agent.values(self.observations)
        next_values[last_dones] = 0.0
        buffer.finish(
            next_values, self.agent.config.gamma, self.agent.config.gae_lambda
        )
        elapsed = perf_counter() - rollout_start
        self.last_rollout_metrics = {
            "environment_steps_per_second": buffer.transition_count / elapsed,
            "policy_inference_time": inference_time,
            "simulation_rollout_time": simulation_time,
        }
        self.observation = self.observations[0]
        self.current_episode = self.current_episodes[0]
        return buffer, completed
