"""Utilities for combining independent APEX critic advantages."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.apex.rewards import APEX_REWARD_GROUPS


def combine_group_advantages(
    advantages: ArrayLike,
    weights: ArrayLike = (0.5, 0.5),
    *,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> NDArray[np.float64]:
    """Combine ``[..., style/task]`` advantages into one actor advantage.

    Each reward stream is normalized independently before weighting when
    ``normalize`` is enabled. Critic targets themselves are not mixed.
    """
    values = np.asarray(advantages, dtype=np.float64)
    group_weights = np.asarray(weights, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != len(APEX_REWARD_GROUPS):
        raise ValueError("advantages must end with a two-element critic axis")
    if group_weights.shape != (len(APEX_REWARD_GROUPS),):
        raise ValueError("weights must have shape (2,) in style/task order")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(group_weights)):
        raise ValueError("advantages and weights must be finite")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    combined_inputs = values
    if normalize:
        reduction_axes = tuple(range(values.ndim - 1))
        mean = values.mean(axis=reduction_axes, keepdims=True)
        standard_deviation = values.std(axis=reduction_axes, keepdims=True)
        combined_inputs = (values - mean) / (standard_deviation + epsilon)
    return np.sum(combined_inputs * group_weights, axis=-1)
