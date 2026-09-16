"""PPO rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def compute_gae(
    rewards: NDArray[np.float64],
    values: NDArray[np.float64],
    dones: NDArray[np.bool_],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute bootstrapped GAE advantages and value targets."""
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must have equal shape")
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_estimate = next_value if index == len(rewards) - 1 else values[index + 1]
        nonterminal = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_estimate * nonterminal - values[index]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
    return advantages, advantages + values


@dataclass
class RolloutBuffer:
    observations: NDArray[np.float64]
    policy_actions: NDArray[np.float64]
    executed_actions: NDArray[np.float64]
    latent_actions: NDArray[np.float64]
    log_probabilities: NDArray[np.float64]
    rewards: NDArray[np.float64]
    values: NDArray[np.float64]
    dones: NDArray[np.bool_]
    advantages: NDArray[np.float64]
    returns: NDArray[np.float64]
    size: int = 0

    @classmethod
    def create(cls, capacity: int, observation_size: int, action_size: int) -> "RolloutBuffer":
        return cls(
            observations=np.zeros((capacity, observation_size)),
            policy_actions=np.zeros((capacity, action_size)),
            executed_actions=np.zeros((capacity, action_size)),
            latent_actions=np.zeros((capacity, action_size)),
            log_probabilities=np.zeros(capacity),
            rewards=np.zeros(capacity),
            values=np.zeros(capacity),
            dones=np.zeros(capacity, dtype=np.bool_),
            advantages=np.zeros(capacity),
            returns=np.zeros(capacity),
        )

    @property
    def capacity(self) -> int:
        return len(self.rewards)

    def add(
        self,
        observation: NDArray[np.float64],
        policy_action: NDArray[np.float64],
        executed_action: NDArray[np.float64],
        latent_action: NDArray[np.float64],
        log_probability: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        if self.size >= self.capacity:
            raise IndexError("rollout buffer is full")
        index = self.size
        self.observations[index] = observation
        self.policy_actions[index] = policy_action
        self.executed_actions[index] = executed_action
        self.latent_actions[index] = latent_action
        self.log_probabilities[index] = log_probability
        self.rewards[index] = reward
        self.values[index] = value
        self.dones[index] = done
        self.size += 1

    def finish(self, next_value: float, gamma: float, gae_lambda: float) -> None:
        advantages, returns = compute_gae(
            self.rewards[: self.size],
            self.values[: self.size],
            self.dones[: self.size],
            next_value,
            gamma,
            gae_lambda,
        )
        self.advantages[: self.size] = advantages
        self.returns[: self.size] = returns
