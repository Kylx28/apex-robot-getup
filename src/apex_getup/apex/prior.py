"""Demonstration action prior and APEX action composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any

from apex_getup.demonstrations import (
    ReferenceTrajectory,
    reference_index_for_episode_step,
)
from apex_getup.env import G1Env


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


@dataclass(frozen=True)
class ResidualControlConfig:
    """Fixed-reference physical residual scale in radians."""

    residual_scale: float = 0.25

    def validate(self) -> None:
        if not np.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative and finite")


class DemonstrationActionPrior:
    """BONES joint targets expressed in the environment's normalized action space.

    Native demonstration samples are retained. The active control path queries
    by elapsed reference time and interpolates before converting to action
    coordinates. Legacy episode-step methods remain for older diagnostics.
    """

    def __init__(
        self,
        trajectory: ReferenceTrajectory,
        env: G1Env,
        *,
        interpolate_reference: bool = True,
    ) -> None:
        actions: list[NDArray[np.float64]] = []
        violation_masks: list[NDArray[np.bool_]] = []
        for target in trajectory.q:
            action, violations = env.joint_target_to_action(target)
            actions.append(action)
            violation_masks.append(violations)
        self.trajectory = trajectory
        self.env = env
        self.interpolate_reference = bool(interpolate_reference)
        self.actions = np.asarray(actions)
        self.joint_limits = env.joint_limits.copy()
        self.physical_limit_violations = np.asarray(violation_masks)
        if self.actions.shape != (len(trajectory.time), env.action_size):
            raise ValueError("prior action trajectory has an invalid shape")
        if not np.all(np.isfinite(self.actions)):
            raise ValueError("prior action trajectory contains NaN or Inf")

    @property
    def duration(self) -> float:
        return self.trajectory.duration

    def action_for_episode_step(
        self, episode_step: int, *, start_index: int = 0
    ) -> NDArray[np.float64]:
        sample_index = reference_index_for_episode_step(
            len(self.actions), episode_step, start_index=start_index
        )
        return self.actions[sample_index].copy()

    def action_at_time(self, reference_time: float) -> NDArray[np.float64]:
        """Return an interpolated reference target in normalized coordinates."""
        return self.env.joint_target_to_action(
            self.trajectory.sample(
                reference_time, interpolate=self.interpolate_reference
            ).q
        )[0]

    def joint_target_at_time(self, reference_time: float) -> NDArray[np.float64]:
        """Return an interpolated reference joint target in physical radians."""
        return self.trajectory.sample(
            reference_time, interpolate=self.interpolate_reference
        ).q

    def joint_target_for_episode_step(
        self, episode_step: int, *, start_index: int = 0
    ) -> NDArray[np.float64]:
        """Return the synchronized BONES joint target in radians."""
        sample_index = reference_index_for_episode_step(
            len(self.actions), episode_step, start_index=start_index
        )
        return self.trajectory.q[sample_index].copy()


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


def compose_residual_joint_target(
    policy_action: ArrayLike,
    reference_joint_target: ArrayLike,
    residual_scale: float,
    joint_limits: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compose ``q_ref + residual_scale * policy_action`` in radians.

    The bounded PPO action has equal radian authority at every joint. Physical
    joint limits are applied only after the residual has been added.
    """
    policy = np.asarray(policy_action, dtype=np.float64)
    reference = np.asarray(reference_joint_target, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if policy.ndim != 1 or policy.shape != reference.shape:
        raise ValueError(
            "policy_action and reference_joint_target must be equal-sized vectors"
        )
    if limits.shape != (policy.size, 2):
        raise ValueError("joint_limits must have shape (action_size, 2)")
    if not all(
        np.all(np.isfinite(value)) for value in (policy, reference, limits)
    ):
        raise ValueError("residual composition inputs must be finite")
    if np.any(policy < -1.0) or np.any(policy > 1.0):
        raise ValueError("policy_action must lie in [-1, 1]")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("joint limits must have positive width")
    if not np.isfinite(residual_scale) or residual_scale < 0:
        raise ValueError("residual_scale must be non-negative and finite")
    unbounded = reference + residual_scale * policy
    clipped = (unbounded < limits[:, 0]) | (unbounded > limits[:, 1])
    return np.clip(unbounded, limits[:, 0], limits[:, 1]), clipped


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


class ResidualActionAdapter:
    """Evaluation adapter for fixed-reference physical residual control."""

    def __init__(self, prior: DemonstrationActionPrior, residual_scale: float):
        if not np.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be non-negative and finite")
        self.prior = prior
        self.residual_scale = float(residual_scale)

    def __call__(
        self, task: Any, policy_action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], dict[str, object]]:
        prior_action = self.prior.action_at_time(task.current_reference_time)
        reference_joint_target = self.prior.joint_target_at_time(
            task.current_reference_time
        )
        physical_joint_target, clipping = compose_residual_joint_target(
            policy_action,
            reference_joint_target,
            self.residual_scale,
            task.env.joint_limits,
        )
        executed_action, _ = task.env.joint_target_to_action(
            physical_joint_target
        )
        return executed_action, {
            "prior_action": prior_action,
            "prior_coefficient": 1.0,
            "executed_action_clipped": clipping,
            "policy_action_scale": self.residual_scale,
            "physical_joint_target": physical_joint_target,
        }
