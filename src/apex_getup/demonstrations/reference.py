"""Time-synchronized demonstration reference lookup."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from apex_getup.demonstrations.trajectory import (
    ReferenceTrajectory,
    reference_index_for_episode_step,
)
from apex_getup.env import G1Env


class DemonstrationActionPrior:
    """Expose BONES joint targets in normalized and physical coordinates.

    Native demonstration samples are retained. The active control path queries
    by elapsed reference time and optionally interpolates before converting to
    the environment's normalized action coordinates. Legacy episode-step
    methods remain available for diagnostics.
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
        """Return the reference target in normalized action coordinates."""
        return self.env.joint_target_to_action(
            self.trajectory.sample(
                reference_time, interpolate=self.interpolate_reference
            ).q
        )[0]

    def joint_target_at_time(self, reference_time: float) -> NDArray[np.float64]:
        """Return the reference joint target in physical radians."""
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
