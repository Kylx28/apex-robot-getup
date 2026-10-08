"""Vectorized running observation and discounted-return normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class NormalizationConfig:
    normalize_observations: bool = False
    normalize_rewards: bool = False
    epsilon: float = 1e-8
    observation_clip: float = 10.0
    reward_clip: float = 10.0

    def validate(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("normalization epsilon must be positive and finite")
        if not np.isfinite(self.observation_clip) or self.observation_clip <= 0:
            raise ValueError("observation_clip must be positive and finite")
        if not np.isfinite(self.reward_clip) or self.reward_clip <= 0:
            raise ValueError("reward_clip must be positive and finite")


class RunningMeanVariance:
    """Numerically stable parallel mean/variance accumulator."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.variance = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: ArrayLike) -> None:
        batch = np.asarray(values, dtype=np.float64)
        expected_dimensions = self.mean.ndim + 1
        if batch.ndim != expected_dimensions or batch.shape[1:] != self.mean.shape:
            raise ValueError(
                f"running-stat batch must have shape [N, {self.mean.shape}]"
            )
        if len(batch) == 0 or not np.all(np.isfinite(batch)):
            raise ValueError("running-stat batch must be nonempty and finite")
        batch_mean = np.mean(batch, axis=0)
        batch_variance = np.var(batch, axis=0)
        batch_count = float(len(batch))
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        second_moment = (
            self.variance * self.count
            + batch_variance * batch_count
            + np.square(delta) * self.count * batch_count / total
        )
        self.mean = new_mean
        self.variance = second_moment / total
        self.count = total

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64)
        variance = np.asarray(state["variance"], dtype=np.float64)
        count = float(state["count"])
        if (
            mean.shape != self.mean.shape
            or variance.shape != self.variance.shape
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(variance))
            or np.any(variance < 0)
            or not np.isfinite(count)
            or count <= 0
        ):
            raise ValueError("invalid running mean/variance state")
        self.mean = mean.copy()
        self.variance = variance.copy()
        self.count = count


class VecNormalizer:
    """VecNormalize-style preprocessing shared across parallel environments."""

    def __init__(
        self,
        observation_size: int,
        num_envs: int,
        gamma: float,
        config: NormalizationConfig | None = None,
        *,
        training: bool = True,
    ) -> None:
        self.config = config or NormalizationConfig()
        self.config.validate()
        if observation_size <= 0 or num_envs <= 0 or not 0.0 < gamma <= 1.0:
            raise ValueError("normalizer dimensions and gamma are invalid")
        self.observation_size = observation_size
        self.num_envs = num_envs
        self.gamma = float(gamma)
        self.training = bool(training)
        self.observation_rms = RunningMeanVariance((observation_size,))
        self.return_rms = RunningMeanVariance(())
        self.discounted_returns = np.zeros(num_envs, dtype=np.float64)

    def set_training(self, training: bool) -> None:
        self.training = bool(training)

    def normalize_observations(
        self, observations: ArrayLike, *, update: bool | None = None
    ) -> NDArray[np.float64]:
        values = np.asarray(observations, dtype=np.float64)
        squeeze = values.ndim == 1
        batch = values[None, :] if squeeze else values
        if batch.ndim != 2 or batch.shape[1] != self.observation_size:
            raise ValueError("observations have the wrong shape")
        should_update = self.training if update is None else update
        if self.config.normalize_observations and should_update:
            self.observation_rms.update(batch)
        if self.config.normalize_observations:
            normalized = (batch - self.observation_rms.mean) / np.sqrt(
                self.observation_rms.variance + self.config.epsilon
            )
            normalized = np.clip(
                normalized,
                -self.config.observation_clip,
                self.config.observation_clip,
            )
        else:
            normalized = batch.copy()
        if not np.all(np.isfinite(normalized)):
            raise FloatingPointError("normalized observations contain NaN or Inf")
        return normalized[0] if squeeze else normalized

    def normalize_rewards(
        self, rewards: ArrayLike, dones: ArrayLike
    ) -> NDArray[np.float64]:
        reward_values = np.asarray(rewards, dtype=np.float64)
        done_values = np.asarray(dones, dtype=np.bool_)
        if reward_values.shape != (self.num_envs,) or done_values.shape != (
            self.num_envs,
        ):
            raise ValueError("rewards and dones must have shape [num_envs]")
        if not np.all(np.isfinite(reward_values)):
            raise ValueError("rewards must be finite")
        if not self.config.normalize_rewards:
            return reward_values.copy()
        if self.training:
            self.discounted_returns = (
                self.gamma * self.discounted_returns + reward_values
            )
            self.return_rms.update(self.discounted_returns)
        normalized = reward_values / np.sqrt(
            self.return_rms.variance + self.config.epsilon
        )
        normalized = np.clip(
            normalized, -self.config.reward_clip, self.config.reward_clip
        )
        if self.training:
            self.discounted_returns[done_values] = 0.0
        return normalized

    def state_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "observation_size": self.observation_size,
            "num_envs": self.num_envs,
            "gamma": self.gamma,
            "observation_rms": self.observation_rms.state_dict(),
            "return_rms": self.return_rms.state_dict(),
            "discounted_returns": self.discounted_returns.copy(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        training: bool,
        num_envs: int | None = None,
    ) -> "VecNormalizer":
        environment_count = int(num_envs or state["num_envs"])
        result = cls(
            int(state["observation_size"]),
            environment_count,
            float(state["gamma"]),
            NormalizationConfig(**dict(state["config"])),
            training=training,
        )
        result.observation_rms.load_state_dict(dict(state["observation_rms"]))
        result.return_rms.load_state_dict(dict(state["return_rms"]))
        stored_returns = np.asarray(state["discounted_returns"], dtype=np.float64)
        if stored_returns.shape == (environment_count,):
            result.discounted_returns = stored_returns.copy()
        return result
