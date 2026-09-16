from __future__ import annotations

import numpy as np

from apex_getup.rl import PPOAgent, PPOConfig, RolloutBuffer, compute_gae


def test_rollout_buffer_shapes_and_gae() -> None:
    buffer = RolloutBuffer.create(4, 97, 29)
    assert buffer.observations.shape == (4, 97)
    assert buffer.policy_actions.shape == (4, 29)
    assert buffer.executed_actions.shape == (4, 29)
    rewards = np.array([1.0, 1.0])
    values = np.array([0.5, 0.25])
    dones = np.array([False, True])
    advantages, returns = compute_gae(rewards, values, dones, 7.0, 1.0, 1.0)
    np.testing.assert_allclose(advantages, [1.5, 0.75])
    np.testing.assert_allclose(returns, [2.0, 1.0])


def test_action_dimensions_and_deterministic_evaluation() -> None:
    agent = PPOAgent(97, 29, PPOConfig(seed=3))
    observation = np.linspace(-1.0, 1.0, 97)
    first = agent.deterministic_action(observation)
    second = agent.deterministic_action(observation)
    assert first.shape == (29,)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.abs(first) <= 1.0)
    sampled, latent, log_probability, value = agent.sample_action(observation)
    assert sampled.shape == latent.shape == (29,)
    assert np.isfinite(log_probability)
    assert np.isfinite(value)


def test_checkpoint_round_trip(tmp_path) -> None:
    agent = PPOAgent(97, 29, PPOConfig(seed=4, actor_hidden_dimensions=(16,)))
    observation = np.linspace(-0.5, 0.5, 97)
    expected = agent.deterministic_action(observation)
    path = agent.save(tmp_path / "checkpoint.npz", metadata={"steps": 12})
    restored, metadata = PPOAgent.load(path)
    np.testing.assert_array_equal(restored.deterministic_action(observation), expected)
    assert metadata == {"steps": 12}
