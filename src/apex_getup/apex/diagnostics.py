"""Plots and summaries for fixed-reference residual control."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex_getup.env.config import JOINT_NAMES


def per_joint_residual_magnitude(
    traces: dict[str, NDArray[np.float64]],
) -> dict[str, float]:
    """Return mean absolute physical residual for each joint, in radians."""
    values = np.mean(np.abs(traces["scaled_policy_action"]), axis=0)
    return {name: float(value) for name, value in zip(JOINT_NAMES, values)}


def save_residual_diagnostics(
    traces: dict[str, NDArray[np.float64]], path: Path, control_dt: float
) -> None:
    """Plot residuals, selected leg targets, pelvis height, and uprightness."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/apex-getup-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(len(traces["pelvis_height"])) * control_dt
    figure, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True)
    axes[0].plot(time, traces["scaled_policy_action"], alpha=0.45, linewidth=0.8)
    axes[0].set_ylabel("residual (rad)")
    axes[0].set_title("Per-joint physical residual")

    key_joints = (
        "left_hip_pitch_joint", "left_knee_joint",
        "right_hip_pitch_joint", "right_knee_joint",
    )
    for name in key_joints:
        index = JOINT_NAMES.index(name)
        axes[1].plot(
            time, traces["prior_action"][:, index], linestyle="--",
            label=f"{name} reference",
        )
        axes[1].plot(
            time, traces["executed_action"][:, index],
            label=f"{name} executed",
        )
    axes[1].set_ylabel("normalized action")
    axes[1].set_title("Reference vs executed hip/knee targets")
    axes[1].legend(ncol=2, fontsize=7)
    axes[2].plot(time, traces["pelvis_height"])
    axes[2].set_ylabel("pelvis height (m)")
    axes[3].plot(time, traces["uprightness"])
    axes[3].set_ylabel("uprightness")
    axes[3].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=150)
    plt.close(figure)
