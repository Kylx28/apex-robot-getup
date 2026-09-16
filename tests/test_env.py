import numpy as np
import pytest

from apex_getup.env import G1Env, G1EnvConfig, InitialState


def test_reset_is_reproducible() -> None:
    env = G1Env(seed=17)
    initial = InitialState(
        root_position=[0.1, -0.2, 0.6],
        root_quaternion=[2.0, 0.0, 0.0, 0.0],
        joint_velocities=np.linspace(-0.1, 0.1, 29),
    )
    first = env.reset(initial, seed=123)
    first_random = env.rng.standard_normal(4)
    second = env.reset(initial, seed=123)
    second_random = env.rng.standard_normal(4)
    np.testing.assert_array_equal(first.as_vector(), second.as_vector())
    np.testing.assert_array_equal(first_random, second_random)


def test_state_and_action_dimensions() -> None:
    env = G1Env()
    state = env.reset()
    assert env.action_size == 29
    assert state.joint_positions.shape == (29,)
    assert state.joint_velocities.shape == (29,)
    assert state.root_position.shape == (3,)
    assert state.root_quaternion.shape == (4,)
    assert state.root_linear_velocity.shape == (3,)
    assert state.root_angular_velocity.shape == (3,)
    assert state.torso_rotation_matrix.shape == (3, 3)
    assert state.projected_gravity.shape == (3,)
    assert state.as_vector().shape == (env.observation_size,)
    assert np.linalg.norm(state.root_quaternion) == pytest.approx(1.0)


def test_action_clipping_is_applied_by_environment() -> None:
    env = G1Env()
    env.reset()
    result = env.step(np.full(29, 5.0))
    np.testing.assert_array_equal(env.last_action, np.ones(29))
    assert result.info["action_was_clipped"] is True
    assert np.all(env.last_joint_target <= env.joint_limits[:, 1])
    assert np.all(env.last_joint_target >= env.joint_limits[:, 0])
    assert np.all(np.abs(env.last_torque) <= env.torque_limits)
    assert 0.0 <= result.info["torque_saturation_fraction"] <= 1.0


def test_actions_span_each_physical_joint_range_and_round_trip() -> None:
    env = G1Env()
    low_action, low_target = env.action_to_joint_target(np.full(29, -1.0))
    high_action, high_target = env.action_to_joint_target(np.full(29, 1.0))
    np.testing.assert_array_equal(low_action, -np.ones(29))
    np.testing.assert_array_equal(high_action, np.ones(29))
    np.testing.assert_allclose(low_target, env.joint_limits[:, 0])
    np.testing.assert_allclose(high_target, env.joint_limits[:, 1])
    np.testing.assert_allclose(env.action_center, np.mean(env.joint_limits, axis=1))
    np.testing.assert_allclose(
        env.action_scale, (env.joint_limits[:, 1] - env.joint_limits[:, 0]) / 2
    )
    probe = np.linspace(-1.0, 1.0, 29)
    _, target = env.action_to_joint_target(probe)
    recovered, violations = env.joint_target_to_action(target)
    np.testing.assert_allclose(recovered, probe, atol=1e-12)
    assert not np.any(violations)


def test_control_and_physics_timestep_consistency() -> None:
    config = G1EnvConfig(simulation_timestep=0.002, control_frequency=50.0)
    assert config.physics_steps_per_control_step == 10
    env = G1Env(config)
    env.reset()
    env.step(np.zeros(29))
    assert env.simulation_time == pytest.approx(config.control_timestep)

    with pytest.raises(ValueError, match="integer multiple"):
        G1EnvConfig(simulation_timestep=0.003, control_frequency=50.0).validate()


def test_custom_reset_state_and_finite_stepping() -> None:
    env = G1Env(G1EnvConfig(episode_duration=0.1), seed=5)
    initial = InitialState(
        joint_positions=env.nominal_pose,
        joint_velocities=np.zeros(29),
        root_position=[0.0, 0.0, 0.35],
        root_quaternion=[np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        root_linear_velocity=[0.0, 0.0, 0.0],
        root_angular_velocity=[0.1, -0.2, 0.3],
    )
    state = env.reset(initial)
    np.testing.assert_allclose(state.root_position, [0.0, 0.0, 0.35])
    np.testing.assert_allclose(state.root_angular_velocity, [0.1, -0.2, 0.3])
    for _ in range(5):
        result = env.step(np.zeros(29))
        assert np.all(np.isfinite(result.state.as_vector()))
        assert np.all(np.isfinite(env.data.qpos))
        assert np.all(np.isfinite(env.data.qvel))
    assert result.truncated
