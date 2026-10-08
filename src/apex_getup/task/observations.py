"""Actor-only proprioceptive observation construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.env.config import NUM_ACTUATORS
from apex_getup.env.observations import RobotState
from apex_getup.task.config import ObservationConfig


STATE_ONLY_OBSERVATION_SIZE = 97
RESIDUAL_REFERENCE_OBSERVATION_SIZE = 214
PAPER_REFERENCE_OBSERVATION_SIZE = 255
# Backward-compatible name for the original actor observation.
OBSERVATION_SIZE = STATE_ONLY_OBSERVATION_SIZE
OBSERVATION_FEATURES = (
    "torso-frame projected gravity [3]",
    "joint position normalized by physical center/half-range [29]",
    "joint velocity / joint_velocity_scale [29]",
    "root linear velocity in torso frame / linear_velocity_scale [3]",
    "root angular velocity in torso frame / angular_velocity_scale [3]",
    "pelvis height / pelvis_height_scale [1]",
    "previous normalized executed action [29]",
)
RESIDUAL_REFERENCE_FEATURES = OBSERVATION_FEATURES + (
    "normalized reference phase [1]",
    "current normalized reference action beta [29]",
    "normalized reference-minus-current joint position error [29]",
    "normalized reference-minus-current joint velocity error [29]",
    "normalized final-standing-minus-current joint position error [29]",
)
PAPER_REFERENCE_FEATURES = (
    "simulated generalized qpos excluding root world x/y [34]",
    "simulated generalized qvel [35]",
    "reference generalized qpos excluding root world x/y [34]",
    "reference generalized qvel [35]",
    "simulated-minus-reference actuated joint position error [29]",
    "simulated-minus-reference actuated joint velocity error [29]",
    "simulated-minus-XML-stand joint position error [29]",
    "normalized reference phase [1]",
    "previous normalized policy/residual action [29]",
)


def observation_size_for_mode(mode: str) -> int:
    if mode == "state_only":
        return STATE_ONLY_OBSERVATION_SIZE
    if mode == "residual_reference":
        return RESIDUAL_REFERENCE_OBSERVATION_SIZE
    if mode == "paper_reference":
        return PAPER_REFERENCE_OBSERVATION_SIZE
    raise ValueError(f"unknown observation mode: {mode}")


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
        raise ValueError(
            "previous_action, joint_center, and joint_scale must have shape (29,)"
        )
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
    if observation.shape != (STATE_ONLY_OBSERVATION_SIZE,) or not np.all(
        np.isfinite(observation)
    ):
        raise FloatingPointError(
            "actor observation has an invalid shape or non-finite value"
        )
    return np.clip(observation, -config.clip, config.clip)


def build_paper_reference_observation(
    simulated_qpos: ArrayLike,
    simulated_qvel: ArrayLike,
    reference_qpos: ArrayLike,
    reference_qvel: ArrayLike,
    simulated_joint_positions: ArrayLike,
    simulated_joint_velocities: ArrayLike,
    reference_joint_positions: ArrayLike,
    reference_joint_velocities: ArrayLike,
    standing_joint_positions: ArrayLike,
    reference_phase: float,
    previous_policy_action: ArrayLike,
) -> NDArray[np.float64]:
    """Build the paper-style raw 255-D reference-conditioned observation.

    Generalized qpos uses MuJoCo ordering with only root world ``x/y`` removed.
    Joint errors consistently use ``simulated - target``.
    """
    sim_qpos = np.asarray(simulated_qpos, dtype=np.float64)
    sim_qvel = np.asarray(simulated_qvel, dtype=np.float64)
    ref_qpos = np.asarray(reference_qpos, dtype=np.float64)
    ref_qvel = np.asarray(reference_qvel, dtype=np.float64)
    vectors = {
        "simulated_qpos": (sim_qpos, (36,)),
        "simulated_qvel": (sim_qvel, (35,)),
        "reference_qpos": (ref_qpos, (36,)),
        "reference_qvel": (ref_qvel, (35,)),
        "simulated_joint_positions": (
            np.asarray(simulated_joint_positions, dtype=np.float64), (29,)
        ),
        "simulated_joint_velocities": (
            np.asarray(simulated_joint_velocities, dtype=np.float64), (29,)
        ),
        "reference_joint_positions": (
            np.asarray(reference_joint_positions, dtype=np.float64), (29,)
        ),
        "reference_joint_velocities": (
            np.asarray(reference_joint_velocities, dtype=np.float64), (29,)
        ),
        "standing_joint_positions": (
            np.asarray(standing_joint_positions, dtype=np.float64), (29,)
        ),
        "previous_policy_action": (
            np.asarray(previous_policy_action, dtype=np.float64), (29,)
        ),
    }
    for name, (value, expected) in vectors.items():
        if value.shape != expected or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {expected}")
    if not np.isfinite(reference_phase) or not 0.0 <= reference_phase <= 1.0:
        raise ValueError("reference_phase must lie in [0, 1]")
    q = vectors["simulated_joint_positions"][0]
    qd = vectors["simulated_joint_velocities"][0]
    q_ref = vectors["reference_joint_positions"][0]
    qd_ref = vectors["reference_joint_velocities"][0]
    q_stand = vectors["standing_joint_positions"][0]
    observation = np.concatenate(
        (
            sim_qpos[2:],
            sim_qvel,
            ref_qpos[2:],
            ref_qvel,
            q - q_ref,
            qd - qd_ref,
            q - q_stand,
            np.array([reference_phase]),
            vectors["previous_policy_action"][0],
        )
    )
    if observation.shape != (PAPER_REFERENCE_OBSERVATION_SIZE,) or not np.all(
        np.isfinite(observation)
    ):
        raise FloatingPointError("paper reference observation is invalid")
    return observation


def append_residual_reference_observation(
    state_observation: ArrayLike,
    state: RobotState,
    reference_phase: float,
    reference_action: ArrayLike,
    reference_joint_positions: ArrayLike,
    reference_joint_velocities: ArrayLike,
    standing_joint_positions: ArrayLike,
    joint_position_scale: ArrayLike,
    config: ObservationConfig,
) -> NDArray[np.float64]:
    """Append the synchronized BONES target to the unchanged 97-D state vector."""
    base = np.asarray(state_observation, dtype=np.float64)
    beta = np.asarray(reference_action, dtype=np.float64)
    q_ref = np.asarray(reference_joint_positions, dtype=np.float64)
    qd_ref = np.asarray(reference_joint_velocities, dtype=np.float64)
    q_stand = np.asarray(standing_joint_positions, dtype=np.float64)
    q_scale = np.asarray(joint_position_scale, dtype=np.float64)
    if base.shape != (STATE_ONLY_OBSERVATION_SIZE,):
        raise ValueError("state_observation must have shape (97,)")
    if any(
        value.shape != (NUM_ACTUATORS,)
        for value in (beta, q_ref, qd_ref, q_stand, q_scale)
    ):
        raise ValueError("reference joint features must have shape (29,)")
    if not np.isfinite(reference_phase) or not 0.0 <= reference_phase <= 1.0:
        raise ValueError("reference_phase must lie in [0, 1]")
    if np.any(q_scale <= 0):
        raise ValueError("joint_position_scale must be positive")
    reference_features = np.concatenate(
        (
            np.array([reference_phase]),
            beta,
            (q_ref - state.joint_positions) / q_scale,
            (qd_ref - state.joint_velocities) / config.joint_velocity_scale,
            (q_stand - state.joint_positions) / q_scale,
        )
    )
    observation = np.concatenate((base, reference_features))
    if observation.shape != (RESIDUAL_REFERENCE_OBSERVATION_SIZE,) or not np.all(
        np.isfinite(observation)
    ):
        raise FloatingPointError(
            "residual-reference observation has an invalid shape or non-finite value"
        )
    # The first 97 values have already been clipped by build_actor_observation;
    # applying the same bound to the appended values preserves them exactly.
    return np.clip(observation, -config.clip, config.clip)
