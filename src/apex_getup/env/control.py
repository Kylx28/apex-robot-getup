"""Normalized action and physical joint-target mappings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def action_parameters_from_joint_limits(
    joint_limits: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-joint action centers and half-range scales."""
    limits = np.asarray(joint_limits, dtype=np.float64)
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError("joint_limits must have shape (N, 2)")
    if not np.all(np.isfinite(limits)):
        raise ValueError("joint limits must be finite")
    widths = limits[:, 1] - limits[:, 0]
    if np.any(widths <= 0):
        raise ValueError("every joint range must have positive width")
    return np.mean(limits, axis=1), widths / 2.0


def action_to_joint_target(
    action: ArrayLike, joint_limits: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map a clipped normalized action across each full physical joint range."""
    action_array = np.asarray(action, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    center, scale = action_parameters_from_joint_limits(limits)
    if action_array.shape != center.shape:
        raise ValueError(f"action must have shape {center.shape}, got {action_array.shape}")
    if not np.all(np.isfinite(action_array)):
        raise ValueError("action must contain only finite values")
    clipped_action = np.clip(action_array, -1.0, 1.0)
    target = center + scale * clipped_action
    return clipped_action, np.clip(target, limits[:, 0], limits[:, 1])


# Compatibility name for callers written before Milestone 2.5. Its arguments
# intentionally match the new joint-limit mapping rather than the old nominal
# pose mapping.
normalized_action_to_target = action_to_joint_target


def joint_target_to_action(
    target: ArrayLike,
    joint_limits: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Convert physical angles to normalized actions, explicitly reporting clipping.

    Returns the clipped action and a per-joint mask indicating targets outside
    the representable ``[-1, 1]`` action range.
    """
    target_array = np.asarray(target, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    center, scale = action_parameters_from_joint_limits(limits)
    if target_array.shape != center.shape:
        raise ValueError(f"target must have shape {center.shape}, got {target_array.shape}")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target conversion inputs must be finite")
    violations = (target_array < limits[:, 0]) | (target_array > limits[:, 1])
    clipped_target = np.clip(target_array, limits[:, 0], limits[:, 1])
    action = (clipped_target - center) / scale
    return np.clip(action, -1.0, 1.0), violations


physical_target_to_normalized_action = joint_target_to_action
