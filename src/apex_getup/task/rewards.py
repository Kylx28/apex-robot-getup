"""Table-I-inspired reward for demonstration-guided humanoid stand-up."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from apex_getup.task.config import RewardConfig


def quaternion_angular_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Shortest angular distance in radians between two ``wxyz`` quaternions."""
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != (4,) or second_array.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    first_norm = np.linalg.norm(first_array)
    second_norm = np.linalg.norm(second_array)
    if first_norm < 1e-12 or second_norm < 1e-12:
        raise ValueError("quaternions must have nonzero norm")
    dot = abs(float(np.dot(first_array / first_norm, second_array / second_norm)))
    return float(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


@dataclass(frozen=True)
class RewardComponents:
    pose_tracking: float
    velocity_tracking: float
    root_xy_tracking: float
    pelvis_height_tracking: float
    orientation_tracking: float
    standing_pose: float
    uprightness: float
    standing_height: float
    foot_slip_penalty: float
    action_penalty: float
    action_smoothness_penalty: float
    weighted_pose_tracking: float
    weighted_velocity_tracking: float
    weighted_root_xy_tracking: float
    weighted_pelvis_height_tracking: float
    weighted_orientation_tracking: float
    weighted_standing_pose: float
    weighted_uprightness: float
    weighted_standing_height: float
    weighted_foot_slip_penalty: float
    weighted_action_penalty: float
    weighted_action_smoothness_penalty: float
    w_track: float
    w_stand: float
    standing_phase_weight: float
    weighted_tracking_total: float
    weighted_final_standing_total: float
    reward_constant: float
    total_reward: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


def _vector(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def compute_task_reward(
    *,
    phase: float,
    joint_positions: ArrayLike,
    joint_velocities: ArrayLike,
    root_position: ArrayLike,
    root_quaternion: ArrayLike,
    pelvis_height: float,
    uprightness: float,
    reference_joint_positions: ArrayLike,
    reference_joint_velocities: ArrayLike,
    reference_root_position: ArrayLike,
    reference_root_quaternion: ArrayLike,
    standing_joint_positions: ArrayLike,
    standing_height: float,
    left_foot_contact: bool,
    right_foot_contact: bool,
    left_foot_planar_velocity: ArrayLike,
    right_foot_planar_velocity: ArrayLike,
    policy_action: ArrayLike,
    previous_policy_action: ArrayLike,
    config: RewardConfig,
) -> RewardComponents:
    """Compute the reward from the synchronized reference state.

    Control regularization uses the sampled PPO ``policy_action``. In residual
    mode this is the learned residual before scaling/composition, never the
    BONES prior or the executed action.
    """
    if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
        raise ValueError("phase must lie in [0, 1]")
    q = np.asarray(joint_positions, dtype=np.float64)
    if q.ndim != 1 or not np.all(np.isfinite(q)):
        raise ValueError("joint_positions must be a finite vector")
    joint_shape = q.shape
    qd = _vector(joint_velocities, joint_shape, "joint_velocities")
    q_ref = _vector(reference_joint_positions, joint_shape, "reference_joint_positions")
    qd_ref = _vector(reference_joint_velocities, joint_shape, "reference_joint_velocities")
    q_stand = _vector(standing_joint_positions, joint_shape, "standing_joint_positions")
    root = _vector(root_position, (3,), "root_position")
    root_ref = _vector(reference_root_position, (3,), "reference_root_position")
    root_quat = _vector(root_quaternion, (4,), "root_quaternion")
    root_quat_ref = _vector(reference_root_quaternion, (4,), "reference_root_quaternion")
    left_velocity = _vector(left_foot_planar_velocity, (2,), "left_foot_planar_velocity")
    right_velocity = _vector(right_foot_planar_velocity, (2,), "right_foot_planar_velocity")
    action = np.asarray(policy_action, dtype=np.float64)
    if action.ndim != 1 or not np.all(np.isfinite(action)):
        raise ValueError("policy_action must be a finite vector")
    previous_action = _vector(previous_policy_action, action.shape, "previous_policy_action")
    if not all(np.isfinite(value) for value in (pelvis_height, uprightness, standing_height)):
        raise ValueError("height and uprightness values must be finite")

    pose = float(np.exp(-config.pose_tracking_scale * np.mean((q - q_ref) ** 2)))
    velocity = float(
        np.exp(-config.velocity_tracking_scale * np.mean((qd - qd_ref) ** 2))
    )
    root_xy = float(
        np.exp(-config.root_xy_tracking_scale * np.mean((root[:2] - root_ref[:2]) ** 2))
    )
    height = float(
        np.exp(-config.pelvis_height_tracking_scale * (pelvis_height - root_ref[2]) ** 2)
    )
    orientation_error = quaternion_angular_distance(root_quat, root_quat_ref)
    orientation = float(np.exp(-config.orientation_tracking_scale * orientation_error**2))

    w_stand = float(
        np.clip(
            (phase - config.standing_phase_threshold)
            / (1.0 - config.standing_phase_threshold),
            0.0,
            1.0,
        )
    )
    w_track = 1.0 - 0.5 * w_stand
    standing_pose = float(
        np.exp(-config.standing_pose_scale * np.mean((q - q_stand) ** 2))
    )
    upright = float(np.clip((uprightness + 1.0) / 2.0, 0.0, 1.0))
    standing_height_reward = float(
        np.exp(-config.standing_height_scale * (pelvis_height - standing_height) ** 2)
    )

    slip = float(
        (np.dot(left_velocity, left_velocity) if left_foot_contact else 0.0)
        + (np.dot(right_velocity, right_velocity) if right_foot_contact else 0.0)
    )
    control = float(np.mean(np.square(action)))
    smoothness = float(np.mean(np.square(action - previous_action)))

    weighted_pose = w_track * config.pose_tracking_weight * pose
    weighted_velocity = w_track * config.velocity_tracking_weight * velocity
    weighted_root_xy = w_track * config.root_xy_tracking_weight * root_xy
    weighted_height = w_track * config.pelvis_height_tracking_weight * height
    weighted_orientation = w_track * config.orientation_tracking_weight * orientation
    weighted_standing_pose = (
        w_stand * config.standing_pose_weight * standing_pose
    )
    weighted_upright = w_stand * config.uprightness_weight * upright
    weighted_standing_height = (
        w_stand
        * config.standing_height_weight
        * standing_height_reward
    )
    weighted_tracking_total = (
        weighted_pose
        + weighted_velocity
        + weighted_root_xy
        + weighted_height
        + weighted_orientation
    )
    weighted_final_standing_total = (
        weighted_standing_pose + weighted_upright + weighted_standing_height
    )
    weighted_slip = -config.foot_slip_penalty_weight * slip
    weighted_control = -config.action_penalty_weight * control
    weighted_smoothness = -config.action_smoothness_penalty_weight * smoothness
    reward_constant = 0.2
    total = (
        weighted_tracking_total
        + weighted_final_standing_total
        + reward_constant
        + weighted_slip
        + weighted_control
        + weighted_smoothness
    )
    return RewardComponents(
        pose_tracking=pose,
        velocity_tracking=velocity,
        root_xy_tracking=root_xy,
        pelvis_height_tracking=height,
        orientation_tracking=orientation,
        standing_pose=standing_pose,
        uprightness=upright,
        standing_height=standing_height_reward,
        foot_slip_penalty=slip,
        action_penalty=control,
        action_smoothness_penalty=smoothness,
        weighted_pose_tracking=weighted_pose,
        weighted_velocity_tracking=weighted_velocity,
        weighted_root_xy_tracking=weighted_root_xy,
        weighted_pelvis_height_tracking=weighted_height,
        weighted_orientation_tracking=weighted_orientation,
        weighted_standing_pose=weighted_standing_pose,
        weighted_uprightness=weighted_upright,
        weighted_standing_height=weighted_standing_height,
        weighted_foot_slip_penalty=weighted_slip,
        weighted_action_penalty=weighted_control,
        weighted_action_smoothness_penalty=weighted_smoothness,
        w_track=w_track,
        w_stand=w_stand,
        standing_phase_weight=w_stand,
        weighted_tracking_total=weighted_tracking_total,
        weighted_final_standing_total=weighted_final_standing_total,
        reward_constant=reward_constant,
        total_reward=float(total),
    )
