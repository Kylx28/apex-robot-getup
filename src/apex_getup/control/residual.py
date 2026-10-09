"""Fixed-reference residual joint-target composition."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.demonstrations import DemonstrationActionPrior


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
    if not all(np.all(np.isfinite(value)) for value in (policy, reference, limits)):
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


class ResidualActionAdapter:
    """Adapt a residual policy action to a physical reference-relative target."""

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
        executed_action, _ = task.env.joint_target_to_action(physical_joint_target)
        return executed_action, {
            "prior_action": prior_action,
            "prior_coefficient": 1.0,
            "executed_action_clipped": clipping,
            "policy_action_scale": self.residual_scale,
            "physical_joint_target": physical_joint_target,
        }
