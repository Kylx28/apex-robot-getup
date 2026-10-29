"""Shape contract for the asymmetric, multi-critic APEX rollout buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ApexRolloutSpec:
    """Dimensions required when extending the existing vector rollout buffer.

    A full buffer will store actor and critic observations separately and use
    ``[T, N, 2]`` arrays for rewards, values, advantages, returns, and timeout
    bootstrap values. Policy actions and log probabilities retain their
    existing meanings and shapes.
    """

    capacity: int
    num_envs: int
    actor_observation_size: int
    critic_observation_size: int
    action_size: int = 29
    critic_count: int = 2

    def validate(self) -> None:
        dimensions = (
            self.capacity,
            self.num_envs,
            self.actor_observation_size,
            self.critic_observation_size,
            self.action_size,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all APEX rollout dimensions must be positive")
        if self.critic_count != 2:
            raise ValueError("full APEX requires style and task critics")

    @property
    def transition_count(self) -> int:
        self.validate()
        return self.capacity * self.num_envs

    @property
    def critic_array_shape(self) -> tuple[int, int, int]:
        self.validate()
        return self.capacity, self.num_envs, self.critic_count


@dataclass
class ApexRolloutBuffer:
    """Time-major storage for asymmetric APEX PPO rollouts.

    Actor and critic observations are deliberately stored separately. Reward,
    value, timeout-bootstrap, advantage, and return arrays end in a two-element
    critic axis ordered as ``(style, task)``.

    Core shapes are::

        actor_observations       [T, N, actor_observation_size]
        critic_observations      [T, N, critic_observation_size]
        actions                  [T, N, action_size]
        log_probabilities        [T, N]
        group arrays             [T, N, 2]
        dones                    [T, N]

    Style and task returns are computed independently; they are combined only
    later when constructing the actor objective.
    """

    actor_observations: NDArray[np.float64]
    critic_observations: NDArray[np.float64]
    policy_actions: NDArray[np.float64]
    executed_actions: NDArray[np.float64]
    prior_actions: NDArray[np.float64]
    prior_coefficients: NDArray[np.float64]
    executed_action_clipped: NDArray[np.bool_]
    latent_actions: NDArray[np.float64]
    log_probabilities: NDArray[np.float64]
    group_rewards: NDArray[np.float64]
    group_values: NDArray[np.float64]
    dones: NDArray[np.bool_]
    timeout_bootstrap_values: NDArray[np.float64]
    group_advantages: NDArray[np.float64]
    group_returns: NDArray[np.float64]
    size: int = 0

    @classmethod
    def create(
        cls,
        capacity: int,
        actor_observation_size: int,
        critic_observation_size: int,
        action_size: int = 29,
        num_envs: int = 1,
    ) -> "ApexRolloutBuffer":
        """Allocate an empty rollout with the canonical APEX shapes."""
        spec = ApexRolloutSpec(
            capacity=capacity,
            num_envs=num_envs,
            actor_observation_size=actor_observation_size,
            critic_observation_size=critic_observation_size,
            action_size=action_size,
        )
        spec.validate()
        action_shape = (capacity, num_envs, action_size)
        group_shape = spec.critic_array_shape
        return cls(
            actor_observations=np.zeros(
                (capacity, num_envs, actor_observation_size), dtype=np.float64
            ),
            critic_observations=np.zeros(
                (capacity, num_envs, critic_observation_size), dtype=np.float64
            ),
            policy_actions=np.zeros(action_shape, dtype=np.float64),
            executed_actions=np.zeros(action_shape, dtype=np.float64),
            prior_actions=np.zeros(action_shape, dtype=np.float64),
            prior_coefficients=np.zeros((capacity, num_envs), dtype=np.float64),
            executed_action_clipped=np.zeros(action_shape, dtype=np.bool_),
            latent_actions=np.zeros(action_shape, dtype=np.float64),
            log_probabilities=np.zeros((capacity, num_envs), dtype=np.float64),
            group_rewards=np.zeros(group_shape, dtype=np.float64),
            group_values=np.zeros(group_shape, dtype=np.float64),
            dones=np.zeros((capacity, num_envs), dtype=np.bool_),
            timeout_bootstrap_values=np.zeros(group_shape, dtype=np.float64),
            group_advantages=np.zeros(group_shape, dtype=np.float64),
            group_returns=np.zeros(group_shape, dtype=np.float64),
        )

    def _batch(
        self, value: ArrayLike, tail: tuple[int, ...], name: str
    ) -> np.ndarray:
        array = np.asarray(value)
        expected = (self.num_envs, *tail)
        if self.num_envs == 1 and array.shape == tail:
            array = array[None, ...]
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
        return array

    def add(
        self,
        actor_observations: ArrayLike,
        critic_observations: ArrayLike,
        policy_actions: ArrayLike,
        executed_actions: ArrayLike,
        prior_actions: ArrayLike,
        prior_coefficients: ArrayLike,
        executed_action_clipped: ArrayLike,
        latent_actions: ArrayLike,
        log_probabilities: ArrayLike,
        group_rewards: ArrayLike,
        group_values: ArrayLike,
        dones: ArrayLike,
        timeout_bootstrap_values: ArrayLike | None = None,
    ) -> None:
        """Append one vectorized environment step."""
        if self.size >= self.capacity:
            raise IndexError("rollout buffer is full")
        index = self.size
        actor_size = self.actor_observations.shape[-1]
        critic_size = self.critic_observations.shape[-1]
        action_size = self.policy_actions.shape[-1]
        group_tail = (self.critic_count,)

        self.actor_observations[index] = self._batch(
            actor_observations, (actor_size,), "actor_observations"
        )
        self.critic_observations[index] = self._batch(
            critic_observations, (critic_size,), "critic_observations"
        )
        for destination, source, name in (
            (self.policy_actions, policy_actions, "policy_actions"),
            (self.executed_actions, executed_actions, "executed_actions"),
            (self.prior_actions, prior_actions, "prior_actions"),
            (
                self.executed_action_clipped,
                executed_action_clipped,
                "executed_action_clipped",
            ),
            (self.latent_actions, latent_actions, "latent_actions"),
        ):
            destination[index] = self._batch(source, (action_size,), name)
        for destination, source, tail, name in (
            (
                self.prior_coefficients,
                prior_coefficients,
                (),
                "prior_coefficients",
            ),
            (
                self.log_probabilities,
                log_probabilities,
                (),
                "log_probabilities",
            ),
            (self.group_rewards, group_rewards, group_tail, "group_rewards"),
            (self.group_values, group_values, group_tail, "group_values"),
            (self.dones, dones, (), "dones"),
        ):
            destination[index] = self._batch(source, tail, name)
        if timeout_bootstrap_values is not None:
            self.timeout_bootstrap_values[index] = self._batch(
                timeout_bootstrap_values,
                group_tail,
                "timeout_bootstrap_values",
            )
        self.size += 1

    def finish(
        self,
        next_group_values: ArrayLike,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute independent style/task GAE and return targets."""
        next_values = self._batch(
            next_group_values, (self.critic_count,), "next_group_values"
        ).astype(np.float64, copy=False)
        last_advantage = np.zeros(
            (self.num_envs, self.critic_count), dtype=np.float64
        )
        for index in range(self.size - 1, -1, -1):
            next_estimate = (
                next_values
                if index == self.size - 1
                else self.group_values[index + 1]
            )
            nonterminal = (~self.dones[index]).astype(np.float64)[:, None]
            bootstrap = (
                next_estimate * nonterminal
                + self.timeout_bootstrap_values[index]
            )
            delta = (
                self.group_rewards[index]
                + gamma * bootstrap
                - self.group_values[index]
            )
            last_advantage = (
                delta
                + gamma * gae_lambda * nonterminal * last_advantage
            )
            self.group_advantages[index] = last_advantage
        self.group_returns[: self.size] = (
            self.group_advantages[: self.size]
            + self.group_values[: self.size]
        )

    def flatten(self, array: np.ndarray) -> np.ndarray:
        """Flatten populated ``[T, N, ...]`` entries for PPO minibatches."""
        populated = array[: self.size]
        return populated.reshape(
            (self.transition_count, *populated.shape[2:])
        )

    @property
    def capacity(self) -> int:
        """Maximum number of stored control steps per environment."""
        return int(self.group_rewards.shape[0])

    @property
    def num_envs(self) -> int:
        """Number of independent environments represented by each step."""
        return int(self.group_rewards.shape[1])

    @property
    def critic_count(self) -> int:
        """Number of reward/value streams; full APEX requires two."""
        return int(self.group_rewards.shape[2])

    @property
    def transition_count(self) -> int:
        """Number of populated environment transitions."""
        return self.size * self.num_envs
