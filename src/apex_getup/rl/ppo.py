"""Dependency-light vanilla PPO with a tanh-squashed Gaussian actor."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex_getup.rl.buffer import RolloutBuffer
from apex_getup.rl.config import PPOConfig
from apex_getup.rl.networks import Adam, MLP


class PPOAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: PPOConfig | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.config.validate()
        self.observation_size = observation_size
        self.action_size = action_size
        self.rng = np.random.default_rng(self.config.seed)
        self.actor = MLP(
            observation_size,
            self.config.actor_hidden_dimensions,
            action_size,
            self.rng,
            output_scale=0.01,
        )
        self.critic = MLP(
            observation_size,
            self.config.critic_hidden_dimensions,
            1,
            self.rng,
        )
        self.log_std = np.full(action_size, np.log(self.config.initial_action_std))
        self.optimizer = Adam(self.parameters, self.config.learning_rate)

    @property
    def parameters(self) -> list[NDArray[np.float64]]:
        return [*self.actor.parameters, *self.critic.parameters, self.log_std]

    def value(self, observation: NDArray[np.float64]) -> float:
        return float(np.asarray(self.critic.forward(observation)).reshape(-1)[0])

    def deterministic_action(self, observation: NDArray[np.float64]) -> NDArray[np.float64]:
        mean = np.asarray(self.actor.forward(observation)).reshape(-1)
        return np.tanh(mean)

    def sample_action(
        self, observation: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        mean = np.asarray(self.actor.forward(observation)).reshape(-1)
        standard_deviation = np.exp(self.log_std)
        latent = mean + standard_deviation * self.rng.standard_normal(self.action_size)
        action = np.tanh(latent)
        log_probability = self._log_probability(latent[None, :], action[None, :], mean[None, :])[0]
        return action, latent, float(log_probability), self.value(observation)

    def _log_probability(
        self,
        latent: NDArray[np.float64],
        action: NDArray[np.float64],
        mean: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        inverse_variance = np.exp(-2.0 * self.log_std)
        normal = -0.5 * np.square(latent - mean) * inverse_variance
        normal -= self.log_std + 0.5 * np.log(2.0 * np.pi)
        jacobian = np.log(1.0 - np.square(action) + 1e-6)
        return np.sum(normal - jacobian, axis=1)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        size = buffer.size
        advantages = buffer.advantages[:size].copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        metrics: dict[str, list[float]] = {
            "actor_loss": [], "critic_loss": [], "entropy": [],
            "approximate_kl": [], "clip_fraction": [], "gradient_norm": [],
        }
        for _ in range(self.config.update_epochs):
            order = self.rng.permutation(size)
            for start in range(0, size, self.config.minibatch_size):
                indexes = order[start : start + self.config.minibatch_size]
                observations = buffer.observations[indexes]
                latent = buffer.latent_actions[indexes]
                actions = buffer.policy_actions[indexes]
                old_log_probability = buffer.log_probabilities[indexes]
                batch_advantages = advantages[indexes]
                returns = buffer.returns[indexes]
                mean, actor_cache = self.actor.forward(observations, cache=True)
                values_output, critic_cache = self.critic.forward(observations, cache=True)
                values = values_output[:, 0]
                new_log_probability = self._log_probability(latent, actions, mean)
                log_ratio = np.clip(new_log_probability - old_log_probability, -20.0, 20.0)
                ratio = np.exp(log_ratio)
                clipped_ratio = np.clip(
                    ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
                )
                surrogate = np.minimum(
                    ratio * batch_advantages, clipped_ratio * batch_advantages
                )
                actor_loss = -float(np.mean(surrogate))
                active = ((batch_advantages >= 0) & (ratio <= 1.0 + self.config.clip_range)) | (
                    (batch_advantages < 0) & (ratio >= 1.0 - self.config.clip_range)
                )
                log_probability_gradient = (
                    -active.astype(np.float64) * ratio * batch_advantages / len(indexes)
                )
                inverse_variance = np.exp(-2.0 * self.log_std)
                actor_output_gradient = log_probability_gradient[:, None] * (
                    latent - mean
                ) * inverse_variance
                actor_gradients = self.actor.backward(actor_output_gradient, actor_cache)
                log_std_gradient = np.sum(
                    log_probability_gradient[:, None]
                    * (np.square(latent - mean) * inverse_variance - 1.0),
                    axis=0,
                ) - self.config.entropy_coefficient
                value_error = values - returns
                critic_loss = float(np.mean(np.square(value_error)))
                critic_output_gradient = (
                    2.0 * self.config.value_coefficient * value_error[:, None] / len(indexes)
                )
                critic_gradients = self.critic.backward(
                    critic_output_gradient, critic_cache
                )
                gradient_norm = self.optimizer.step(
                    [*actor_gradients, *critic_gradients, log_std_gradient],
                    self.config.maximum_gradient_norm,
                )
                metrics["actor_loss"].append(actor_loss)
                metrics["critic_loss"].append(critic_loss)
                metrics["entropy"].append(
                    float(np.sum(self.log_std + 0.5 * np.log(2.0 * np.pi * np.e)))
                )
                metrics["approximate_kl"].append(
                    float(np.mean(old_log_probability - new_log_probability))
                )
                metrics["clip_fraction"].append(
                    float(np.mean(np.abs(ratio - 1.0) > self.config.clip_range))
                )
                metrics["gradient_norm"].append(gradient_norm)
        self.log_std[:] = np.clip(self.log_std, -4.0, 1.0)
        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def save(self, path: Path | str, *, metadata: dict[str, object] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, object] = {
            "observation_size": np.array(self.observation_size),
            "action_size": np.array(self.action_size),
            "config_json": np.array(json.dumps(asdict(self.config))),
            "metadata_json": np.array(json.dumps(metadata or {})),
            "log_std": self.log_std,
        }
        for prefix, network in (("actor", self.actor), ("critic", self.critic)):
            for index, (weight, bias) in enumerate(zip(network.weights, network.biases)):
                arrays[f"{prefix}_weight_{index}"] = weight
                arrays[f"{prefix}_bias_{index}"] = bias
        np.savez_compressed(destination, **arrays)
        return destination

    @classmethod
    def load(cls, path: Path | str) -> tuple["PPOAgent", dict[str, object]]:
        with np.load(Path(path), allow_pickle=False) as archive:
            values = json.loads(str(archive["config_json"].item()))
            values["actor_hidden_dimensions"] = tuple(values["actor_hidden_dimensions"])
            values["critic_hidden_dimensions"] = tuple(values["critic_hidden_dimensions"])
            agent = cls(
                int(archive["observation_size"]),
                int(archive["action_size"]),
                PPOConfig(**values),
            )
            agent.log_std[:] = archive["log_std"]
            for prefix, network in (("actor", agent.actor), ("critic", agent.critic)):
                for index in range(len(network.weights)):
                    network.weights[index][:] = archive[f"{prefix}_weight_{index}"]
                    network.biases[index][:] = archive[f"{prefix}_bias_{index}"]
            metadata = json.loads(str(archive["metadata_json"].item()))
        return agent, metadata
