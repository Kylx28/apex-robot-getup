"""Configuration owned by the full APEX algorithm.

The existing PPO and prior-schedule configurations remain the source of truth
for their respective subsystems. This module contains only settings specific
to combining APEX reward groups and asymmetric observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ApexConfig:
    """Multi-critic APEX settings."""

    actor_observation_mode: str = "state_only"
    critic_observation_mode: str = "paper_reference"
    style_advantage_weight: float = 0.5
    task_advantage_weight: float = 0.5
    normalize_group_advantages: bool = True

    def validate(self) -> None:
        if self.actor_observation_mode != "state_only":
            raise ValueError(
                "the deployable APEX actor must use state_only observations"
            )
        if self.critic_observation_mode != "paper_reference":
            raise ValueError("APEX critics must use paper_reference observations")
        weights = np.asarray(
            [self.style_advantage_weight, self.task_advantage_weight],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("APEX advantage weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("at least one APEX advantage weight must be positive")

    @property
    def advantage_weights(self) -> tuple[float, float]:
        """Weights in canonical ``(style, task)`` critic order."""
        self.validate()
        return self.style_advantage_weight, self.task_advantage_weight
