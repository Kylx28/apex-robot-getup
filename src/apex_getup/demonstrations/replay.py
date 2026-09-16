"""Shared replay initialization and diagnostic helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex_getup.demonstrations.trajectory import ReferenceTrajectory
from apex_getup.env.config import JOINT_NAMES
from apex_getup.env.g1_env import InitialState


@dataclass(frozen=True)
class ReferenceRepresentability:
    old_envelope_fraction: float
    physical_limit_fraction: float
    old_envelope_counts: NDArray[np.int64]
    physical_limit_counts: NDArray[np.int64]
    max_physical_violation: float
    max_physical_violation_per_joint: NDArray[np.float64]
    clipped_q: NDArray[np.float64]


def analyze_reference_representability(
    q: NDArray[np.float64],
    joint_limits: NDArray[np.float64],
    nominal_pose: NDArray[np.float64],
    *,
    old_action_scale: float = 0.5,
) -> ReferenceRepresentability:
    """Separate historical action-envelope violations from physical violations."""
    values = np.asarray(q, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    nominal = np.asarray(nominal_pose, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(JOINT_NAMES):
        raise ValueError("q must have shape (T, 29)")
    if limits.shape != (len(JOINT_NAMES), 2) or nominal.shape != (len(JOINT_NAMES),):
        raise ValueError("joint_limits and nominal_pose must match the 29-joint order")
    if not all(np.all(np.isfinite(item)) for item in (values, limits, nominal)):
        raise ValueError("reference representability inputs must be finite")
    old_violations = np.abs(values - nominal[None, :]) > old_action_scale
    below = np.maximum(limits[None, :, 0] - values, 0.0)
    above = np.maximum(values - limits[None, :, 1], 0.0)
    magnitude = np.maximum(below, above)
    physical_violations = magnitude > 0.0
    return ReferenceRepresentability(
        old_envelope_fraction=float(np.mean(old_violations)),
        physical_limit_fraction=float(np.mean(physical_violations)),
        old_envelope_counts=np.count_nonzero(old_violations, axis=0),
        physical_limit_counts=np.count_nonzero(physical_violations, axis=0),
        max_physical_violation=float(np.max(magnitude)),
        max_physical_violation_per_joint=np.max(magnitude, axis=0),
        clipped_q=np.clip(values, limits[None, :, 0], limits[None, :, 1]),
    )


def initial_state_from_trajectory(trajectory: ReferenceTrajectory) -> InitialState:
    """Construct an environment reset state exactly from reference sample zero."""
    return InitialState(
        joint_positions=trajectory.q[0],
        joint_velocities=trajectory.qd[0],
        root_position=None if trajectory.root_pos is None else trajectory.root_pos[0],
        root_quaternion=None if trajectory.root_quat is None else trajectory.root_quat[0],
    )


def quaternion_angle_error(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
    dot = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


@dataclass(frozen=True)
class ReplayDiagnostics:
    time: NDArray[np.float64]
    joint_error: NDArray[np.float64]
    reference_q: NDArray[np.float64]
    actual_q: NDArray[np.float64]
    pelvis_height: NDArray[np.float64]
    root_position_error: NDArray[np.float64] | None
    root_orientation_error: NDArray[np.float64] | None
    torque_saturation_fraction: float
    action_saturation_fraction: float
    foot_contact_fraction: tuple[float, float]
    non_foot_contact_fraction: float
    mean_abs_torque: NDArray[np.float64]
    max_abs_torque: NDArray[np.float64]
    torque_over_80_fraction: NDArray[np.float64]
    torque_limits: NDArray[np.float64]
    contact_region_fractions: dict[str, float]
    contact_region_first_time: dict[str, float]
    contact_region_last_time: dict[str, float]

    @property
    def per_joint_rmse(self) -> NDArray[np.float64]:
        return np.sqrt(np.mean(np.square(self.joint_error), axis=0))

    @property
    def mean_joint_position_error(self) -> float:
        return float(np.mean(np.abs(self.joint_error)))

    @property
    def max_joint_position_error(self) -> float:
        return float(np.max(np.abs(self.joint_error)))

    @property
    def final_pelvis_height(self) -> float:
        return float(self.pelvis_height[-1])

    @property
    def mean_torque_utilization(self) -> NDArray[np.float64]:
        return self.mean_abs_torque / self.torque_limits

    @property
    def peak_torque_utilization(self) -> NDArray[np.float64]:
        return self.max_abs_torque / self.torque_limits


def save_diagnostic_plot(diagnostics: ReplayDiagnostics, path: Path | str) -> Path:
    """Save aggregate and detailed joint-tracking diagnostics."""
    cache_path = Path("/tmp/apex-getup-matplotlib-cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_path))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10))
    axes[0].plot(diagnostics.time, np.abs(diagnostics.joint_error), alpha=0.45, linewidth=0.8)
    axes[0].plot(
        diagnostics.time,
        np.mean(np.abs(diagnostics.joint_error), axis=1),
        color="black",
        linewidth=2,
        label="mean",
    )
    axes[0].set_ylabel("absolute joint error [rad]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].bar(np.arange(len(JOINT_NAMES)), diagnostics.per_joint_rmse)
    axes[1].set_ylabel("joint RMSE [rad]")
    axes[1].set_xticks(np.arange(len(JOINT_NAMES)), JOINT_NAMES, rotation=90, fontsize=7)
    axes[1].grid(alpha=0.25)
    axes[2].plot(diagnostics.time, diagnostics.pelvis_height)
    axes[2].set_ylabel("pelvis height [m]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)

    tracking_names = (
        "left_knee_joint", "right_knee_joint", "left_hip_pitch_joint",
        "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    )
    detail_path = destination.with_name(f"{destination.stem}_joint_tracking{destination.suffix}")
    detail_figure, detail_axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for axis, name in zip(detail_axes.flat, tracking_names):
        joint_index = JOINT_NAMES.index(name)
        axis.plot(diagnostics.time, diagnostics.reference_q[:, joint_index], label="reference")
        axis.plot(diagnostics.time, diagnostics.actual_q[:, joint_index], label="actual")
        axis.set_title(name)
        axis.set_ylabel("q [rad]")
        axis.grid(alpha=0.25)
    detail_axes.flat[0].legend()
    detail_axes[-1, 0].set_xlabel("time [s]")
    detail_axes[-1, 1].set_xlabel("time [s]")
    detail_figure.tight_layout()
    detail_figure.savefig(detail_path, dpi=140)
    plt.close(detail_figure)
    return destination
