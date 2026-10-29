"""Observation contracts for an asymmetric APEX actor and critic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ApexObservations:
    """One or more actor/critic observations with matching leading shapes.

    The actor observation must remain reference-free. The critic observation
    may contain privileged BONES reference state during training.
    """

    actor: NDArray[np.float64]
    critic: NDArray[np.float64]

    @classmethod
    def from_arrays(
        cls,
        actor: ArrayLike,
        critic: ArrayLike,
    ) -> "ApexObservations":
        actor_array = np.asarray(actor, dtype=np.float64)
        critic_array = np.asarray(critic, dtype=np.float64)
        if actor_array.ndim < 1 or critic_array.ndim < 1:
            raise ValueError("actor and critic observations must have a feature axis")
        if actor_array.shape[:-1] != critic_array.shape[:-1]:
            raise ValueError("actor and critic observations must have matching batches")
        if actor_array.shape[-1] <= 0 or critic_array.shape[-1] <= 0:
            raise ValueError("observation feature dimensions must be positive")
        if not np.all(np.isfinite(actor_array)) or not np.all(
            np.isfinite(critic_array)
        ):
            raise ValueError("APEX observations must be finite")
        return cls(actor=actor_array.copy(), critic=critic_array.copy())

    @property
    def actor_size(self) -> int:
        return int(self.actor.shape[-1])

    @property
    def critic_size(self) -> int:
        return int(self.critic.shape[-1])


def build_apex_observations(
    actor_observation: ArrayLike,
    critic_observation: ArrayLike,
) -> ApexObservations:
    """Validate and copy a synchronized actor/critic observation pair."""
    return ApexObservations.from_arrays(actor_observation, critic_observation)
