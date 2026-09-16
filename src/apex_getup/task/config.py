"""Configuration for the fixed-state getting-up task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RewardConfig:
    height_weight: float = 2.0
    uprightness_weight: float = 1.0
    height_progress_weight: float = 10.0
    standing_bonus_weight: float = 5.0
    action_penalty_weight: float = 0.01
    action_smoothness_weight: float = 0.02
    torque_penalty_weight: float = 0.005
    low_height: float = 0.05
    standing_height: float = 0.72


@dataclass(frozen=True)
class SuccessConfig:
    pelvis_height: float = 0.65
    torso_uprightness: float = 0.85
    maximum_linear_speed: float = 0.5
    maximum_angular_speed: float = 1.0
    hold_duration: float = 0.5
    terminate_on_success: bool = False


@dataclass(frozen=True)
class ObservationConfig:
    joint_velocity_scale: float = 10.0
    linear_velocity_scale: float = 3.0
    angular_velocity_scale: float = 5.0
    pelvis_height_scale: float = 1.0
    clip: float = 10.0


@dataclass(frozen=True)
class GetUpTaskConfig:
    demonstration_path: Path = Path(
        "datasets/processed/getup/stand_up_lying_R_002__A473.npz"
    )
    episode_duration: float = 10.0
    initial_velocity_mode: str = "zero"
    reward: RewardConfig = field(default_factory=RewardConfig)
    success: SuccessConfig = field(default_factory=SuccessConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def validate(self) -> None:
        if self.episode_duration <= 0:
            raise ValueError("episode_duration must be positive")
        if self.initial_velocity_mode not in {"zero", "reference"}:
            raise ValueError("initial_velocity_mode must be 'zero' or 'reference'")
        if self.success.hold_duration <= 0:
            raise ValueError("success hold_duration must be positive")
        if not self.success.torso_uprightness <= 1.0:
            raise ValueError("torso uprightness threshold cannot exceed one")
        if self.success.maximum_linear_speed <= 0 or self.success.maximum_angular_speed <= 0:
            raise ValueError("success speed thresholds must be positive")
        if self.reward.standing_height <= self.reward.low_height:
            raise ValueError("standing reward height must exceed low height")
        scales = (
            self.observation.joint_velocity_scale,
            self.observation.linear_velocity_scale,
            self.observation.angular_velocity_scale,
            self.observation.pelvis_height_scale,
            self.observation.clip,
        )
        if any(value <= 0 for value in scales):
            raise ValueError("observation scales and clip must be positive")
