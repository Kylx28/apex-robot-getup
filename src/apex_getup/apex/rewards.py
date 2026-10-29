"""Style and task-reward grouping for APEX's separate critics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Mapping

import numpy as np


APEX_REWARD_GROUPS = ("style", "task")

# Weighted fields produced by the current Table-I reward. The split is kept in
# one place so the environment reward remains unchanged while each APEX critic
# receives only its own return stream.
STYLE_REWARD_FIELDS = (
    "weighted_pose_tracking",
    "weighted_velocity_tracking",
    "weighted_root_xy_tracking",
    "weighted_pelvis_height_tracking",
    "weighted_orientation_tracking",
)
TASK_REWARD_FIELDS = (
    "weighted_standing_pose",
    "weighted_uprightness",
    "weighted_standing_height",
    "reward_constant",
    "weighted_foot_slip_penalty",
    "weighted_action_penalty",
    "weighted_action_smoothness_penalty",
)


@dataclass(frozen=True)
class ApexRewardGroups:
    """Per-step reward streams in canonical ``(style, task)`` order."""

    style: float
    task: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.style, self.task])):
            raise ValueError("APEX reward groups must be finite")

    def as_array(self) -> np.ndarray:
        return np.asarray([self.style, self.task], dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.style + self.task)


def split_reward_components(
    components: object | Mapping[str, float],
) -> ApexRewardGroups:
    """Split the current reward result without recomputing reward formulas."""
    if isinstance(components, Mapping):
        values = components
    elif is_dataclass(components) and not isinstance(components, type):
        values = asdict(components)
    else:
        raise TypeError("components must be a dataclass instance or mapping")
    required = (*STYLE_REWARD_FIELDS, *TASK_REWARD_FIELDS)
    missing = [name for name in required if name not in values]
    if missing:
        raise KeyError(f"reward components are missing fields: {missing}")
    style = sum(float(values[name]) for name in STYLE_REWARD_FIELDS)
    task = sum(float(values[name]) for name in TASK_REWARD_FIELDS)
    return ApexRewardGroups(style=style, task=task)
