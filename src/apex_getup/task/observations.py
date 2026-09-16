"""Actor-only proprioceptive observation construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.env.config import NUM_ACTUATORS
from apex_getup.env.observations import RobotState
from apex_getup.task.config import ObservationConfig


OBSERVATION_SIZE = 97
OBSERVATION_FEATURES = (
    "torso-frame projected gravity [3]",
    "joint position normalized by physical center/half-range [29]",
    "joint velocity / joint_velocity_scale [29]",
    "root linear velocity in torso frame / linear_velocity_scale [3]",
    "root angular velocity in torso frame / angular_velocity_scale [3]",
    "pelvis height / pelvis_height_scale [1]",
    "previous normalized executed action [29]",
)


def build_actor_observation(
    state: RobotState,
    previous_action: ArrayLike,
    joint_center: ArrayLike,
    joint_scale: ArrayLike,
    config: ObservationConfig,
) -> NDArray[np.float64]:
    """Build the 97-vector actor observation with no reference information.

    The torso rotation matrix maps torso-frame vectors into world coordinates,
    so its transpose converts the public world-frame root velocities to the
    torso frame.
    """
    action = np.asarray(previous_action, dtype=np.float64)
    center = np.asarray(joint_center, dtype=np.float64)
    scale = np.asarray(joint_scale, dtype=np.float64)
    expected = (NUM_ACTUATORS,)
    if action.shape != expected or center.shape != expected or scale.shape != expected:
        raise ValueError("previous_action, joint_center, and joint_scale must have shape (29,)")
    if np.any(scale <= 0):
        raise ValueError("joint_scale must be positive")
    rotation_world_from_torso = state.torso_rotation_matrix
    linear_body = rotation_world_from_torso.T @ state.root_linear_velocity
    angular_body = rotation_world_from_torso.T @ state.root_angular_velocity
    observation = np.concatenate(
        (
            state.projected_gravity,
            (state.joint_positions - center) / scale,
            state.joint_velocities / config.joint_velocity_scale,
            linear_body / config.linear_velocity_scale,
            angular_body / config.angular_velocity_scale,
            np.array([state.pelvis_height / config.pelvis_height_scale]),
            action,
        )
    )
    if observation.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(observation)):
        raise FloatingPointError("actor observation has an invalid shape or non-finite value")
    return np.clip(observation, -config.clip, config.clip)
