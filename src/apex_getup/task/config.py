"""Configuration for the fixed-state getting-up task."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from apex_getup.env.config import NUM_ACTUATORS


@dataclass(frozen=True)
class ResetPerturbationConfig:
    """Gaussian perturbations applied around the frame-zero reset state."""

    joint_position_std: float = 0.0
    joint_velocity_std: float = 0.0
    root_orientation_std: float = 0.0
    root_linear_velocity_std: float = 0.0
    root_angular_velocity_std: float = 0.0

    def validate(self) -> None:
        values = (
            self.joint_position_std,
            self.joint_velocity_std,
            self.root_orientation_std,
            self.root_linear_velocity_std,
            self.root_angular_velocity_std,
        )
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("reset perturbation standard deviations must be non-negative")


@dataclass(frozen=True)
class RewardConfig:
    """Table-I-inspired reference, standing, and regularization reward."""

    pose_tracking_weight: float = 2.0
    velocity_tracking_weight: float = 0.7
    root_xy_tracking_weight: float = 0.2
    pelvis_height_tracking_weight: float = 1.0
    orientation_tracking_weight: float = 0.5
    standing_pose_weight: float = 2.0
    uprightness_weight: float = 2.0
    standing_height_weight: float = 1.5
    foot_slip_penalty_weight: float = 0.15
    action_penalty_weight: float = 0.002
    action_smoothness_penalty_weight: float = 0.005
    pose_tracking_scale: float = 4.0
    velocity_tracking_scale: float = 0.03
    root_xy_tracking_scale: float = 2.0
    pelvis_height_tracking_scale: float = 10.0
    orientation_tracking_scale: float = 1.0
    standing_pose_scale: float = 5.0
    standing_height_scale: float = 20.0
    standing_phase_threshold: float = 0.65

    def validate(self) -> None:
        weights = (
            self.pose_tracking_weight,
            self.velocity_tracking_weight,
            self.root_xy_tracking_weight,
            self.pelvis_height_tracking_weight,
            self.orientation_tracking_weight,
            self.standing_pose_weight,
            self.uprightness_weight,
            self.standing_height_weight,
            self.foot_slip_penalty_weight,
            self.action_penalty_weight,
            self.action_smoothness_penalty_weight,
        )
        scales = (
            self.pose_tracking_scale,
            self.velocity_tracking_scale,
            self.root_xy_tracking_scale,
            self.pelvis_height_tracking_scale,
            self.orientation_tracking_scale,
            self.standing_pose_scale,
            self.standing_height_scale,
        )
        if any(not np.isfinite(value) or value < 0 for value in weights):
            raise ValueError("reward weights must be finite and non-negative")
        if any(not np.isfinite(value) or value <= 0 for value in scales):
            raise ValueError("reward scales must be finite and positive")
        if not 0.0 <= self.standing_phase_threshold < 1.0:
            raise ValueError("standing phase threshold must lie in [0, 1)")


@dataclass(frozen=True)
class SuccessConfig:
    pelvis_height: float = 0.65
    torso_uprightness: float = 0.85
    maximum_linear_speed: float = 0.5
    maximum_angular_speed: float = 1.0
    hold_duration: float = 1.5
    terminate_on_success: bool = False


@dataclass(frozen=True)
class ObservationConfig:
    mode: str = "state_only"
    joint_velocity_scale: float = 10.0
    linear_velocity_scale: float = 3.0
    angular_velocity_scale: float = 5.0
    pelvis_height_scale: float = 1.0
    clip: float = 10.0

    def validate(self) -> None:
        if self.mode not in {"state_only", "residual_reference", "paper_reference"}:
            raise ValueError(
                "observation mode must be state_only, residual_reference, or "
                "paper_reference"
            )


@dataclass(frozen=True)
class PositionControllerConfig:
    """MuJoCo position-servo gains in canonical G1 joint order."""

    kp: float | tuple[float, ...] = 60.0
    kd: float | tuple[float, ...] = 3.0

    def validate(self) -> None:
        for name, value in (("kp", self.kp), ("kd", self.kd)):
            array = np.asarray(value, dtype=np.float64)
            if array.ndim == 0:
                array = np.full(NUM_ACTUATORS, float(array), dtype=np.float64)
            if array.shape != (NUM_ACTUATORS,):
                raise ValueError(
                    f"controller {name} must be scalar or shape ({NUM_ACTUATORS},)"
                )
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"controller {name} must be finite and non-negative")


@dataclass(frozen=True)
class GetUpTaskConfig:
    demonstration_path: Path = Path(
        "datasets/processed/getup/stand_up_lying_R_002__A473.npz"
    )
    # ``None`` derives the horizon from the native reference duration plus
    # ``reference_hold_duration``. Explicit durations remain available for
    # short diagnostics and legacy ablations.
    episode_duration: float | None = 10.0
    reference_hold_duration: float = 2.0
    simulation_timestep: float = 0.002
    control_frequency: float = 50.0
    initial_velocity_mode: str = "auto"
    interpolate_reference: bool = True
    align_reference_root_to_fallen: bool = False
    controller: PositionControllerConfig = field(
        default_factory=PositionControllerConfig
    )
    reward: RewardConfig = field(default_factory=RewardConfig)
    reset_perturbation: ResetPerturbationConfig = field(
        default_factory=ResetPerturbationConfig
    )
    success: SuccessConfig = field(default_factory=SuccessConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def validate(self) -> None:
        if self.episode_duration is not None and (
            not np.isfinite(self.episode_duration) or self.episode_duration <= 0
        ):
            raise ValueError("episode_duration must be None or positive and finite")
        if (
            not np.isfinite(self.reference_hold_duration)
            or self.reference_hold_duration < 0
        ):
            raise ValueError("reference_hold_duration must be non-negative and finite")
        if not np.isfinite(self.simulation_timestep) or self.simulation_timestep <= 0:
            raise ValueError("simulation_timestep must be positive and finite")
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive and finite")
        ratio = (1.0 / self.control_frequency) / self.simulation_timestep
        if round(ratio) < 1 or not np.isclose(ratio, round(ratio), atol=1e-10):
            raise ValueError(
                "control period must be an integer multiple of simulation_timestep"
            )
        if self.initial_velocity_mode not in {"auto", "zero", "reference"}:
            raise ValueError("initial_velocity_mode must be auto, zero, or reference")
        if not isinstance(self.interpolate_reference, bool):
            raise ValueError("interpolate_reference must be boolean")
        if not isinstance(self.align_reference_root_to_fallen, bool):
            raise ValueError("align_reference_root_to_fallen must be boolean")
        self.controller.validate()
        self.observation.validate()
        self.reward.validate()
        self.reset_perturbation.validate()
        if self.success.hold_duration <= 0:
            raise ValueError("success hold_duration must be positive")
        if not self.success.torso_uprightness <= 1.0:
            raise ValueError("torso uprightness threshold cannot exceed one")
        if self.success.maximum_linear_speed <= 0 or self.success.maximum_angular_speed <= 0:
            raise ValueError("success speed thresholds must be positive")
        scales = (
            self.observation.joint_velocity_scale,
            self.observation.linear_velocity_scale,
            self.observation.angular_velocity_scale,
            self.observation.pelvis_height_scale,
            self.observation.clip,
        )
        if any(value <= 0 for value in scales):
            raise ValueError("observation scales and clip must be positive")
