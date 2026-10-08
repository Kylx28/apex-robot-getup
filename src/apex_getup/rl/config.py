"""Vanilla PPO configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    backend: str = "native"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.0
    value_coefficient: float = 0.5
    maximum_gradient_norm: float = 0.5
    rollout_steps: int = 1024
    minibatch_size: int = 256
    update_epochs: int = 5
    target_kl: float | None = None
    actor_hidden_dimensions: tuple[int, ...] = (128, 128)
    critic_hidden_dimensions: tuple[int, ...] = (128, 128)
    # Full-range G1 targets make conventional 0.5--0.6 action noise produce
    # large target jumps and pervasive torque clipping.
    initial_action_std: float = 0.1
    seed: int = 0
    device: str = "cpu"

    def validate(self) -> None:
        if self.backend not in {"native", "sb3"}:
            raise ValueError("PPO backend must be 'native' or 'sb3'")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("learning_rate and gamma must be positive")
        if not 0 <= self.gae_lambda <= 1 or self.clip_range <= 0:
            raise ValueError("gae_lambda or clip_range is invalid")
        if self.rollout_steps <= 0 or self.minibatch_size <= 0 or self.update_epochs <= 0:
            raise ValueError("rollout and update sizes must be positive")
        if self.initial_action_std <= 0 or self.maximum_gradient_norm <= 0:
            raise ValueError("action std and gradient norm must be positive")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be None or positive")
        if any(width <= 0 for width in (*self.actor_hidden_dimensions, *self.critic_hidden_dimensions)):
            raise ValueError("hidden dimensions must be positive")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
