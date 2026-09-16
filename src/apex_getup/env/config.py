"""Configuration for the G1 MuJoCo environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


NUM_ACTUATORS = 29

# Order matches the actuator order in the vendored MuJoCo Menagerie model.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# The Menagerie "stand" keyframe. Angles are radians.
DEFAULT_NOMINAL_POSE: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
)

# Unitree actuator force limits from the Menagerie model, in N m.
DEFAULT_TORQUE_LIMITS: tuple[float, ...] = (
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
)


def _vector(value: float | Sequence[float], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(NUM_ACTUATORS, float(array), dtype=np.float64)
    if array.shape != (NUM_ACTUATORS,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or shape ({NUM_ACTUATORS},)")
    return array


@dataclass(frozen=True)
class G1EnvConfig:
    """Numerical and controller settings.

    Scalar gains are broadcast to all 29 joints; per-joint sequences are also
    accepted. Action centers/scales come from the loaded model's joint limits.
    ``control_frequency`` must divide the physics rate exactly.
    """

    model_path: Path | None = None
    simulation_timestep: float = 0.002
    control_frequency: float = 50.0
    # Selected by the bounded Milestone 2.5 replay sweep. The sweep's historical
    # baseline was kp=40, kd=2 and scaled both gains together.
    kp: float | tuple[float, ...] = 60.0
    kd: float | tuple[float, ...] = 3.0
    nominal_joint_pose: tuple[float, ...] = DEFAULT_NOMINAL_POSE
    torque_limits: tuple[float, ...] = DEFAULT_TORQUE_LIMITS
    episode_duration: float = 10.0

    @property
    def control_timestep(self) -> float:
        return 1.0 / self.control_frequency

    @property
    def physics_steps_per_control_step(self) -> int:
        ratio = self.control_timestep / self.simulation_timestep
        rounded = round(ratio)
        if rounded < 1 or not np.isclose(ratio, rounded, atol=1e-10):
            raise ValueError(
                "control timestep must be an integer multiple of simulation_timestep"
            )
        return int(rounded)

    def validate(self) -> None:
        if not np.isfinite(self.simulation_timestep) or self.simulation_timestep <= 0:
            raise ValueError("simulation_timestep must be positive and finite")
        if not np.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive and finite")
        if not np.isfinite(self.episode_duration) or self.episode_duration <= 0:
            raise ValueError("episode_duration must be positive and finite")
        _ = self.physics_steps_per_control_step
        for name, value in (
            ("kp", self.kp),
            ("kd", self.kd),
            ("nominal_joint_pose", self.nominal_joint_pose),
            ("torque_limits", self.torque_limits),
        ):
            array = _vector(value, name)
            if name != "nominal_joint_pose" and np.any(array < 0):
                raise ValueError(f"{name} must be non-negative")

    def arrays(self) -> tuple[NDArray[np.float64], ...]:
        return (
            _vector(self.kp, "kp"),
            _vector(self.kd, "kd"),
            _vector(self.nominal_joint_pose, "nominal_joint_pose"),
            _vector(self.torque_limits, "torque_limits"),
        )
