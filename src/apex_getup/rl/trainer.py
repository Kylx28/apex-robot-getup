"""Rollout collection kept separate from the PPO update."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apex_getup.rl.buffer import RolloutBuffer
from apex_getup.rl.ppo import PPOAgent
from apex_getup.task import GetUpTask


@dataclass
class EpisodeAccumulator:
    episode_return: float = 0.0
    maximum_pelvis_height: float = -np.inf
    standing_bonus: float = 0.0
    action_magnitude: float = 0.0
    action_delta_magnitude: float = 0.0
    mean_torque: float = 0.0
    torque_saturation: float = 0.0
    steps: int = 0

    def add(self, reward: float, info: dict[str, object]) -> None:
        self.episode_return += reward
        self.maximum_pelvis_height = max(
            self.maximum_pelvis_height, float(info["pelvis_height"])
        )
        self.standing_bonus += float(info["standing_bonus"])
        self.action_magnitude += float(info["action_magnitude"])
        self.action_delta_magnitude += float(info["action_delta_magnitude"])
        self.mean_torque += float(info["mean_torque"])
        self.torque_saturation += float(info["torque_saturation_fraction"])
        self.steps += 1

    def finish(self, info: dict[str, object]) -> dict[str, float]:
        denominator = max(1, self.steps)
        return {
            "episode_return": self.episode_return,
            "success": float(info["episode_success"]),
            "time_to_stand": np.nan
            if info["first_success_time"] is None
            else float(info["first_success_time"]),
            "maximum_pelvis_height": self.maximum_pelvis_height,
            "final_pelvis_height": float(info["pelvis_height"]),
            "final_uprightness": float(info["uprightness"]),
            "standing_bonus": self.standing_bonus,
            "action_magnitude": self.action_magnitude / denominator,
            "action_delta_magnitude": self.action_delta_magnitude / denominator,
            "mean_torque": self.mean_torque / denominator,
            "torque_saturation": self.torque_saturation / denominator,
        }


class PPOTrainer:
    def __init__(self, task: GetUpTask, agent: PPOAgent, seed: int = 0):
        self.task = task
        self.agent = agent
        self.observation = task.reset(seed=seed)
        self.current_episode = EpisodeAccumulator()

    def collect_rollout(
        self, steps: int | None = None
    ) -> tuple[RolloutBuffer, list[dict[str, float]]]:
        count = steps or self.agent.config.rollout_steps
        buffer = RolloutBuffer.create(
            count, self.task.observation_size, self.task.action_size
        )
        completed: list[dict[str, float]] = []
        last_done = False
        for _ in range(count):
            policy_action, latent_action, log_probability, value = (
                self.agent.sample_action(self.observation)
            )
            # Kept as a separate array for later APEX action composition.
            executed_action = policy_action.copy()
            result = self.task.step(executed_action)
            done = result.terminated or result.truncated
            buffer.add(
                self.observation,
                policy_action,
                executed_action,
                latent_action,
                log_probability,
                result.reward,
                value,
                done,
            )
            self.current_episode.add(result.reward, result.info)
            self.observation = result.observation
            last_done = done
            if done:
                completed.append(self.current_episode.finish(result.info))
                self.current_episode = EpisodeAccumulator()
                self.observation = self.task.reset()
        next_value = 0.0 if last_done else self.agent.value(self.observation)
        buffer.finish(
            next_value, self.agent.config.gamma, self.agent.config.gae_lambda
        )
        return buffer, completed
