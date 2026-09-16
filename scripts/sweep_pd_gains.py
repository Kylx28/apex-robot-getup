#!/usr/bin/env python3
"""Run the bounded Milestone 2.5 PD gain calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np

from apex_getup.demonstrations import discover_bones_dataset, load_motion
from apex_getup.demonstrations.replay import (
    analyze_reference_representability,
    save_diagnostic_plot,
)
from apex_getup.env import G1Env, G1EnvConfig
from apex_getup.env.config import JOINT_NAMES
from replay_demo import run_dynamic


BASELINE_KP = 40.0
BASELINE_KD = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", default="stand_up_lying_R_002__A473")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--control-frequency", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = discover_bones_dataset(args.dataset_root)
    raw = load_motion(args.motion, paths)
    # Keep the calibration baseline explicit so changing the selected defaults
    # does not silently change the meaning of the recorded multipliers.
    baseline = G1EnvConfig(
        control_frequency=args.control_frequency,
        kp=BASELINE_KP,
        kd=BASELINE_KD,
    )
    trajectory = raw.resample(baseline.control_timestep)
    multipliers = (0.5, 1.0, 1.5, 2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for multiplier in multipliers:
        logging.info("running gain multiplier %.1fx", multiplier)
        config = G1EnvConfig(
            control_frequency=args.control_frequency,
            kp=np.asarray(baseline.kp) * multiplier,
            kd=np.asarray(baseline.kd) * multiplier,
            episode_duration=trajectory.duration + baseline.control_timestep,
        )
        env = G1Env(config, seed=0)
        representability = analyze_reference_representability(
            trajectory.q, env.joint_limits, env.nominal_pose
        )
        try:
            diagnostics = run_dynamic(env, trajectory, headless=True, speed=1.0)
            stable = True
            plot_path = args.output_dir / f"gain_{multiplier:.1f}x.png"
            save_diagnostic_plot(diagnostics, plot_path)
            highest_index = int(np.argmax(diagnostics.per_joint_rmse))
            row: dict[str, object] = {
                "multiplier": multiplier,
                "kp": float(np.asarray(config.kp)),
                "kd": float(np.asarray(config.kd)),
                "stable": stable,
                "physical_clip_fraction": representability.physical_limit_fraction,
                "mean_joint_error_rad": diagnostics.mean_joint_position_error,
                "max_joint_error_rad": diagnostics.max_joint_position_error,
                "per_joint_rmse_rad": diagnostics.per_joint_rmse.tolist(),
                "pelvis_max_m": float(diagnostics.pelvis_height.max()),
                "pelvis_final_m": diagnostics.final_pelvis_height,
                "root_orientation_error_deg": None
                if diagnostics.root_orientation_error is None
                else float(np.rad2deg(diagnostics.root_orientation_error).mean()),
                "torque_saturation_fraction": diagnostics.torque_saturation_fraction,
                "mean_torque_utilization": float(diagnostics.mean_torque_utilization.mean()),
                "peak_torque_utilization": float(diagnostics.peak_torque_utilization.max()),
                "torque_over_80_fraction": float(diagnostics.torque_over_80_fraction.mean()),
                "per_joint_peak_torque_utilization": (
                    diagnostics.peak_torque_utilization.tolist()
                ),
                "per_joint_torque_over_80_fraction": (
                    diagnostics.torque_over_80_fraction.tolist()
                ),
                "highest_error_joint": JOINT_NAMES[highest_index],
                "highest_joint_rmse_rad": float(diagnostics.per_joint_rmse[highest_index]),
            }
        except (FloatingPointError, RuntimeError) as error:
            stable = False
            logging.exception("gain %.1fx was unstable", multiplier)
            row = {
                "multiplier": multiplier,
                "kp": float(np.asarray(config.kp)),
                "kd": float(np.asarray(config.kd)),
                "stable": stable,
                "error": str(error),
            }
        rows.append(row)

    stable_rows = [row for row in rows if row["stable"]]
    if not stable_rows:
        raise RuntimeError("all gain settings were unstable")
    best = min(
        stable_rows,
        key=lambda row: (
            float(row["mean_joint_error_rad"]),
            -float(row["pelvis_final_m"]),
            float(row["torque_saturation_fraction"]),
        ),
    )
    csv_path = args.output_dir / "gain_sweep.csv"
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "gain_sweep.json"
    summary_path.write_text(
        json.dumps(
            {
                "motion": trajectory.motion_id,
                "historical_baseline": {"kp": BASELINE_KP, "kd": BASELINE_KD},
                "results": rows,
                "selected": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/apex-getup-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stable_x = np.asarray([float(row["multiplier"]) for row in stable_rows])
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(
        stable_x,
        [float(row["mean_joint_error_rad"]) for row in stable_rows],
        marker="o",
        label="mean",
    )
    axes[0, 0].plot(
        stable_x,
        [float(row["max_joint_error_rad"]) for row in stable_rows],
        marker="o",
        label="max",
    )
    axes[0, 0].set_ylabel("joint error (rad)")
    axes[0, 0].legend()
    axes[0, 1].plot(
        stable_x,
        [float(row["pelvis_max_m"]) for row in stable_rows],
        marker="o",
        label="maximum",
    )
    axes[0, 1].plot(
        stable_x,
        [float(row["pelvis_final_m"]) for row in stable_rows],
        marker="o",
        label="final",
    )
    axes[0, 1].set_ylabel("pelvis height (m)")
    axes[0, 1].legend()
    axes[1, 0].plot(
        stable_x,
        [float(row["root_orientation_error_deg"]) for row in stable_rows],
        marker="o",
    )
    axes[1, 0].set_ylabel("mean root orientation error (deg)")
    axes[1, 1].plot(
        stable_x,
        [float(row["torque_saturation_fraction"]) for row in stable_rows],
        marker="o",
        label="saturation",
    )
    axes[1, 1].plot(
        stable_x,
        [float(row["mean_torque_utilization"]) for row in stable_rows],
        marker="o",
        label="mean utilization",
    )
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("gain multiplier")
        axis.grid(alpha=0.25)
    figure.suptitle(f"PD calibration: {trajectory.motion_id}")
    comparison_path = args.output_dir / "gain_sweep.png"
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)

    header = (
        "gain  mean_err  max_err  pelvis_max/final  root_deg  torque_sat  "
        "mean/peak_util  stable"
    )
    print(header)
    for row in rows:
        if not row["stable"]:
            print(f"{row['multiplier']:>3.1f}x  unstable")
            continue
        print(
            f"{row['multiplier']:>3.1f}x  {row['mean_joint_error_rad']:.4f}    "
            f"{row['max_joint_error_rad']:.4f}   "
            f"{row['pelvis_max_m']:.3f}/{row['pelvis_final_m']:.3f}       "
            f"{row['root_orientation_error_deg']:.2f}     "
            f"{row['torque_saturation_fraction']:.3f}       "
            f"{row['mean_torque_utilization']:.3f}/{row['peak_torque_utilization']:.3f}      yes"
        )
    print(f"selected: {best['multiplier']:.1f}x (kp={best['kp']:g}, kd={best['kd']:g})")
    print(f"saved: {csv_path}, {summary_path}, and {comparison_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
