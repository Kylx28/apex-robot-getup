import numpy as np
import pytest

from apex_getup.env.control import (
    action_parameters_from_joint_limits,
    action_to_joint_target,
    joint_target_to_action,
    pd_torque,
)


def test_normalized_action_and_joint_target_clipping() -> None:
    action, target = action_to_joint_target(
        action=[-2.0, 0.5, 3.0],
        joint_limits=[[-0.25, 0.25], [-2.0, 0.75], [0.0, 1.0]],
    )
    np.testing.assert_allclose(action, [-1.0, 0.5, 1.0])
    np.testing.assert_allclose(target, [-0.25, 0.0625, 1.0])


def test_pd_torque_computation_and_clipping() -> None:
    torque = pd_torque(
        target=[1.0, -1.0],
        position=[0.5, 0.0],
        velocity=[0.25, -0.5],
        kp=[10.0, 10.0],
        kd=[2.0, 2.0],
        torque_limits=[4.0, 20.0],
    )
    np.testing.assert_allclose(torque, [4.0, -9.0])


def test_nonfinite_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        action_to_joint_target([np.nan], [[-1.0, 1.0]])


def test_physical_target_to_normalized_action() -> None:
    action, saturated = joint_target_to_action(
        target=[0.25, 2.5, -2.0],
        joint_limits=[[-1.0, 1.0], [0.0, 2.0], [-3.0, 1.0]],
    )
    np.testing.assert_allclose(action, [0.25, 1.0, -0.5])
    np.testing.assert_array_equal(saturated, [False, True, False])
    _, clipped_target = action_to_joint_target(
        action,
        joint_limits=[[-1.0, 1.0], [0.0, 2.0], [-3.0, 1.0]],
    )
    np.testing.assert_allclose(clipped_target, [0.25, 2.0, -2.0])


def test_invalid_joint_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive width"):
        action_parameters_from_joint_limits([[1.0, 1.0]])
