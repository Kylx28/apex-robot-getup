from __future__ import annotations

import numpy as np
import pytest

from apex_getup.env.observations import RobotState
from apex_getup.task import (
    GetUpTask,
    GetUpTaskConfig,
    ObservationConfig,
    RewardConfig,
    SuccessConfig,
    build_actor_observation,
    compute_task_reward,
)


def _state(rotation: np.ndarray | None = None) -> RobotState:
    matrix = np.eye(3) if rotation is None else rotation
    return RobotState(
        joint_positions=np.zeros(29),
        joint_velocities=np.zeros(29),
        root_position=np.zeros(3),
        root_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        root_linear_velocity=np.array([1.0, 2.0, 3.0]),
        root_angular_velocity=np.array([4.0, 5.0, 6.0]),
        pelvis_height=0.5,
        torso_rotation_matrix=matrix,
        projected_gravity=matrix.T @ np.array([0.0, 0.0, -1.0]),
        contacts=(),
        left_foot_contact=False,
        right_foot_contact=False,
        non_foot_contact=False,
    )


def test_actor_observation_shape_finite_and_body_frame() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    observation = build_actor_observation(
        _state(rotation),
        np.zeros(29),
        np.zeros(29),
        np.ones(29),
        ObservationConfig(
            joint_velocity_scale=1.0,
            linear_velocity_scale=1.0,
            angular_velocity_scale=1.0,
        ),
    )
    assert observation.shape == (97,)
    assert np.all(np.isfinite(observation))
    np.testing.assert_allclose(observation[61:64], rotation.T @ [1.0, 2.0, 3.0])
    np.testing.assert_allclose(observation[64:67], rotation.T @ [4.0, 5.0, 6.0])


def test_fixed_reset_is_deterministic() -> None:
    task = GetUpTask()
    first = task.reset(seed=11)
    first_state = task.env.get_state().as_vector()
    second = task.reset(seed=11)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_state, task.env.get_state().as_vector())
    np.testing.assert_array_equal(task.env.get_state().joint_velocities, np.zeros(29))


def test_reward_components_and_total() -> None:
    config = RewardConfig()
    components = compute_task_reward(
        pelvis_height=config.standing_height,
        previous_pelvis_height=config.standing_height - 0.01,
        uprightness=1.0,
        standing_posture=True,
        action=np.ones(29),
        previous_action=np.zeros(29),
        torque=np.full(29, 5.0),
        torque_limits=np.full(29, 10.0),
        config=config,
    )
    expected = 2.0 + 1.0 + 0.1 + 5.0 - 0.01 - 0.02 - 0.005 * 0.25
    assert components.total_reward == pytest.approx(expected)
    assert components.height_reward == 1.0
    assert components.standing_bonus == 1.0


def test_sustained_success_and_timeout() -> None:
    success = SuccessConfig(
        pelvis_height=-1.0,
        torso_uprightness=-1.0,
        maximum_linear_speed=1e6,
        maximum_angular_speed=1e6,
        hold_duration=0.04,
    )
    task = GetUpTask(GetUpTaskConfig(episode_duration=0.04, success=success))
    task.reset()
    first = task.step(np.zeros(29))
    assert not first.info["episode_success"]
    second = task.step(np.zeros(29))
    assert second.info["episode_success"]
    assert second.info["stable_success_duration"] == pytest.approx(0.04)
    assert second.info["first_success_time"] == pytest.approx(0.04)
    assert second.truncated
