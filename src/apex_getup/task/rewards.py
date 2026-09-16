"""Task-only get-up reward components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from apex_getup.task.config import RewardConfig


@dataclass(frozen=True)
class RewardComponents:
    height_reward: float
    uprightness_reward: float
    height_progress_reward: float
    standing_bonus: float
    action_penalty: float
    action_smoothness_penalty: float
    torque_penalty: float
    total_reward: float

    def as_dict(self) -> dict[str, float]:
        return {
            "height_reward": self.height_reward,
            "uprightness_reward": self.uprightness_reward,
            "height_progress_reward": self.height_progress_reward,
            "standing_bonus": self.standing_bonus,
            "action_penalty": self.action_penalty,
            "action_smoothness_penalty": self.action_smoothness_penalty,
            "torque_penalty": self.torque_penalty,
            "total_reward": self.total_reward,
        }


def compute_task_reward(
    *,
    pelvis_height: float,
    previous_pelvis_height: float,
    uprightness: float,
    standing_posture: bool,
    action: ArrayLike,
    previous_action: ArrayLike,
    torque: ArrayLike,
    torque_limits: ArrayLike,
    config: RewardConfig,
) -> RewardComponents:
    action_array = np.asarray(action, dtype=np.float64)
    previous_array = np.asarray(previous_action, dtype=np.float64)
    torque_array = np.asarray(torque, dtype=np.float64)
    limits = np.asarray(torque_limits, dtype=np.float64)
    if not all(
        item.shape == action_array.shape
        for item in (previous_array, torque_array, limits)
    ):
        raise ValueError("action and torque reward inputs must have equal shapes")
    upright = float(np.clip((uprightness + 1.0) / 2.0, 0.0, 1.0))
    normalized_height = float(
        np.clip(
            (pelvis_height - config.low_height)
            / (config.standing_height - config.low_height),
            0.0,
            1.0,
        )
    )
    # Height alone admits an inverted bridge exploit. This smooth product still
    # gives dense feedback while requiring progress toward the correct side up.
    height = normalized_height * upright
    progress = float(pelvis_height - previous_pelvis_height)
    standing = float(standing_posture)
    action_penalty = float(np.mean(np.square(action_array)))
    smoothness = float(np.mean(np.square(action_array - previous_array)))
    torque_penalty = float(np.mean(np.square(torque_array / limits)))
    total = (
        config.height_weight * height
        + config.uprightness_weight * upright
        + config.height_progress_weight * progress
        + config.standing_bonus_weight * standing
        - config.action_penalty_weight * action_penalty
        - config.action_smoothness_weight * smoothness
        - config.torque_penalty_weight * torque_penalty
    )
    return RewardComponents(
        height_reward=height,
        uprightness_reward=upright,
        height_progress_reward=progress,
        standing_bonus=standing,
        action_penalty=action_penalty,
        action_smoothness_penalty=smoothness,
        torque_penalty=torque_penalty,
        total_reward=float(total),
    )
