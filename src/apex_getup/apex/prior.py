"""Decaying APEX action-prior scheduling and composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.demonstrations import DemonstrationActionPrior


@dataclass(frozen=True)
class PriorScheduleConfig:
    """Coefficient ``c_n = decay_lambda ** (n / decay_steps)``."""

    decay_lambda: float = 0.1
    decay_steps: float = 100_000.0
    fixed_coefficient: float | None = None

    def validate(self) -> None:
        if not 0.0 < self.decay_lambda <= 1.0:
            raise ValueError("decay_lambda must lie in (0, 1]")
        if not np.isfinite(self.decay_steps) or self.decay_steps <= 0:
            raise ValueError("decay_steps must be positive and finite")
        if self.fixed_coefficient is not None and not (
            np.isfinite(self.fixed_coefficient)
            and 0.0 <= self.fixed_coefficient <= 1.0
        ):
            raise ValueError("fixed_coefficient must lie in [0, 1]")


class PriorCoefficientSchedule:
    def __init__(self, config: PriorScheduleConfig | None = None):
        self.config = config or PriorScheduleConfig()
        self.config.validate()

    def __call__(self, global_environment_step: int) -> float:
        if global_environment_step < 0:
            raise ValueError("global_environment_step must be non-negative")
        if self.config.fixed_coefficient is not None:
            return self.config.fixed_coefficient
        return float(
            self.config.decay_lambda
            ** (global_environment_step / self.config.decay_steps)
        )


def compose_apex_action(
    policy_action: ArrayLike,
    prior_action: ArrayLike,
    coefficient: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return ``clip(policy_action + coefficient * prior_action, -1, 1)``."""
    policy = np.asarray(policy_action, dtype=np.float64)
    prior = np.asarray(prior_action, dtype=np.float64)
    if policy.shape != prior.shape:
        raise ValueError("policy_action and prior_action must have equal shapes")
    if not np.all(np.isfinite(policy)) or not np.all(np.isfinite(prior)):
        raise ValueError("action composition inputs must be finite")
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be non-negative and finite")
    unbounded = policy + coefficient * prior
    clipped = (unbounded < -1.0) | (unbounded > 1.0)
    return np.clip(unbounded, -1.0, 1.0), clipped


class FixedCoefficientPriorAdapter:
    """Evaluation adapter for a fixed guided or prior-free coefficient."""

    def __init__(self, prior: DemonstrationActionPrior, coefficient: float):
        if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
            raise ValueError("evaluation coefficient must lie in [0, 1]")
        self.prior = prior
        self.coefficient = float(coefficient)

    def __call__(
        self, task: Any, policy_action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], dict[str, object]]:
        prior_action = self.prior.action_at_time(task.current_reference_time)
        executed_action, clipping = compose_apex_action(
            policy_action, prior_action, self.coefficient
        )
        return executed_action, {
            "prior_action": prior_action,
            "prior_coefficient": self.coefficient,
            "executed_action_clipped": clipping,
            "policy_action_scale": 1.0,
        }
