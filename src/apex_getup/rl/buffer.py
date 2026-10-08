"""Vectorized PPO rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray


def compute_gae(
    rewards: NDArray[np.float64],
    values: NDArray[np.float64],
    dones: NDArray[np.bool_],
    next_value: ArrayLike,
    gamma: float,
    gae_lambda: float,
    timeout_bootstrap_values: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute GAE independently for every environment.

    Inputs may be ``[T]`` for compatibility or ``[T, N]`` for vectorized
    collection. Done flags prevent return leakage across automatic resets.
    """
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must have equal shape")
    if rewards.ndim not in {1, 2}:
        raise ValueError("GAE arrays must have shape [T] or [T, N]")
    squeeze = rewards.ndim == 1
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float64)
    value_tensor = torch.as_tensor(values, dtype=torch.float64)
    done_tensor = torch.as_tensor(dones, dtype=torch.bool)
    if timeout_bootstrap_values is None:
        timeout_tensor = torch.zeros_like(reward_tensor)
    else:
        if timeout_bootstrap_values.shape != rewards.shape:
            raise ValueError("timeout bootstrap values must match rewards")
        timeout_tensor = torch.as_tensor(
            timeout_bootstrap_values, dtype=torch.float64
        )
    if squeeze:
        reward_tensor = reward_tensor[:, None]
        value_tensor = value_tensor[:, None]
        done_tensor = done_tensor[:, None]
        timeout_tensor = timeout_tensor[:, None]
    environment_count = reward_tensor.shape[1]
    next_value_tensor = torch.as_tensor(next_value, dtype=torch.float64).reshape(-1)
    if next_value_tensor.shape != (environment_count,):
        raise ValueError("next_value must have one value per environment")
    advantages = torch.zeros_like(reward_tensor)
    last_advantage = torch.zeros(environment_count, dtype=torch.float64)
    for index in range(reward_tensor.shape[0] - 1, -1, -1):
        next_estimate = (
            next_value_tensor
            if index == reward_tensor.shape[0] - 1
            else value_tensor[index + 1]
        )
        nonterminal = (~done_tensor[index]).to(torch.float64)
        bootstrap = next_estimate * nonterminal + timeout_tensor[index]
        delta = reward_tensor[index] + gamma * bootstrap - value_tensor[index]
        last_advantage = (
            delta + gamma * gae_lambda * nonterminal * last_advantage
        )
        advantages[index] = last_advantage
    returns = advantages + value_tensor
    if squeeze:
        return advantages[:, 0].numpy(), returns[:, 0].numpy()
    return advantages.numpy(), returns.numpy()


@dataclass
class RolloutBuffer:
    """Time-major rollout arrays with shape ``[T, N, ...]``."""

    observations: NDArray[np.float64]
    policy_actions: NDArray[np.float64]
    executed_actions: NDArray[np.float64]
    prior_actions: NDArray[np.float64]
    prior_coefficients: NDArray[np.float64]
    policy_action_scales: NDArray[np.float64]
    executed_action_clipped: NDArray[np.bool_]
    latent_actions: NDArray[np.float64]
    log_probabilities: NDArray[np.float64]
    rewards: NDArray[np.float64]
    values: NDArray[np.float64]
    dones: NDArray[np.bool_]
    timeout_bootstrap_values: NDArray[np.float64]
    advantages: NDArray[np.float64]
    returns: NDArray[np.float64]
    size: int = 0

    @classmethod
    def create(
        cls,
        capacity: int,
        observation_size: int,
        action_size: int,
        num_envs: int = 1,
    ) -> "RolloutBuffer":
        if capacity <= 0 or num_envs <= 0:
            raise ValueError("capacity and num_envs must be positive")
        return cls(
            observations=np.zeros((capacity, num_envs, observation_size)),
            policy_actions=np.zeros((capacity, num_envs, action_size)),
            executed_actions=np.zeros((capacity, num_envs, action_size)),
            prior_actions=np.zeros((capacity, num_envs, action_size)),
            prior_coefficients=np.zeros((capacity, num_envs)),
            policy_action_scales=np.ones((capacity, num_envs)),
            executed_action_clipped=np.zeros(
                (capacity, num_envs, action_size), dtype=np.bool_
            ),
            latent_actions=np.zeros((capacity, num_envs, action_size)),
            log_probabilities=np.zeros((capacity, num_envs)),
            rewards=np.zeros((capacity, num_envs)),
            values=np.zeros((capacity, num_envs)),
            dones=np.zeros((capacity, num_envs), dtype=np.bool_),
            timeout_bootstrap_values=np.zeros((capacity, num_envs)),
            advantages=np.zeros((capacity, num_envs)),
            returns=np.zeros((capacity, num_envs)),
        )

    @property
    def capacity(self) -> int:
        return self.rewards.shape[0]

    @property
    def num_envs(self) -> int:
        return self.rewards.shape[1]

    @property
    def transition_count(self) -> int:
        return self.size * self.num_envs

    def _batch(self, value: ArrayLike, tail: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value)
        expected = (self.num_envs, *tail)
        if self.num_envs == 1 and array.shape == tail:
            array = array[None, ...]
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        return array

    def add(
        self,
        observation: ArrayLike,
        policy_action: ArrayLike,
        executed_action: ArrayLike,
        prior_action: ArrayLike,
        prior_coefficient: ArrayLike,
        policy_action_scale: ArrayLike,
        executed_action_clipped: ArrayLike,
        latent_action: ArrayLike,
        log_probability: ArrayLike,
        reward: ArrayLike,
        value: ArrayLike,
        done: ArrayLike,
        timeout_bootstrap_value: ArrayLike | None = None,
    ) -> None:
        if self.size >= self.capacity:
            raise IndexError("rollout buffer is full")
        index = self.size
        action_size = self.policy_actions.shape[-1]
        observation_size = self.observations.shape[-1]
        self.observations[index] = self._batch(
            observation, (observation_size,), "observation"
        )
        for destination, source, name in (
            (self.policy_actions, policy_action, "policy_action"),
            (self.executed_actions, executed_action, "executed_action"),
            (self.prior_actions, prior_action, "prior_action"),
            (self.executed_action_clipped, executed_action_clipped, "clipping"),
            (self.latent_actions, latent_action, "latent_action"),
        ):
            destination[index] = self._batch(source, (action_size,), name)
        for destination, source, name in (
            (self.prior_coefficients, prior_coefficient, "prior_coefficient"),
            (self.policy_action_scales, policy_action_scale, "policy_action_scale"),
            (self.log_probabilities, log_probability, "log_probability"),
            (self.rewards, reward, "reward"),
            (self.values, value, "value"),
            (self.dones, done, "done"),
        ):
            destination[index] = self._batch(source, (), name)
        if timeout_bootstrap_value is not None:
            self.timeout_bootstrap_values[index] = self._batch(
                timeout_bootstrap_value, (), "timeout_bootstrap_value"
            )
        self.size += 1

    def finish(
        self, next_values: ArrayLike, gamma: float, gae_lambda: float
    ) -> None:
        next_array = np.asarray(next_values, dtype=np.float64).reshape(-1)
        if next_array.shape != (self.num_envs,):
            raise ValueError("next_values must have shape [num_envs]")
        advantages, returns = compute_gae(
            self.rewards[: self.size],
            self.values[: self.size],
            self.dones[: self.size],
            next_array,
            gamma,
            gae_lambda,
            self.timeout_bootstrap_values[: self.size],
        )
        self.advantages[: self.size] = advantages
        self.returns[: self.size] = returns

    def flatten(self, array: np.ndarray) -> np.ndarray:
        """Flatten populated ``[T, N, ...]`` entries into PPO batch order."""
        populated = array[: self.size]
        return populated.reshape((self.transition_count, *populated.shape[2:]))
