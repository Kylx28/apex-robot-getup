#!/usr/bin/env python3
"""Replay a BONES-SEED G1 motion kinematically or through MuJoCo position actuators."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import numpy as np

from apex_getup.demonstrations import discover_bones_dataset, load_motion
from apex_getup.demonstrations.replay import (
    ReplayDiagnostics,
    analyze_reference_representability,
    initial_state_from_trajectory,
    quaternion_angle_error,
    save_diagnostic_plot,
)
from apex_getup.env import G1Env, G1EnvConfig
from apex_getup.env.config import JOINT_NAMES
from apex_getup.env.observations import CONTACT_REGIONS, contact_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="exact metadata ID or CSV/NPZ path")
    parser.add_argument("--mode", choices=("kinematic", "dynamic"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--control-frequency", type=float, required=True)
    parser.add_argument("--speed", type=float, default=1.0, help="viewer playback speed multiplier")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-processed", type=Path, default=None)
    return parser.parse_args()


def _viewer(env: G1Env, headless: bool):
    if headless:
        return None
    import mujoco.viewer

    return mujoco.viewer.launch_passive(env.model, env.data)


def _sync(viewer: object | None, wall_start: float, sim_time: float, speed: float) -> None:
    if viewer is None:
        return
    viewer.sync()
    time.sleep(max(0.0, wall_start + sim_time / speed - time.monotonic()))


def run_kinematic(env: G1Env, trajectory, headless: bool, speed: float) -> None:
    env.reset(initial_state_from_trajectory(trajectory), seed=0)
    viewer = _viewer(env, headless)
    wall_start = time.monotonic()
    first_height = None
    final_state = None
    try:
        for index, timestamp in enumerate(trajectory.time):
            if viewer is not None and not viewer.is_running():
                break
            final_state = env.set_kinematic_state(
                trajectory.q[index],
                joint_velocities=trajectory.qd[index],
                root_position=None if trajectory.root_pos is None else trajectory.root_pos[index],
                root_quaternion=None if trajectory.root_quat is None else trajectory.root_quat[index],
                time=float(timestamp),
            )
            if first_height is None:
                first_height = final_state.pelvis_height
            _sync(viewer, wall_start, float(timestamp), speed)
    finally:
        if viewer is not None:
            viewer.close()
    assert final_state is not None
    logging.info(
        "kinematic replay complete: pelvis %.3f -> %.3f m; final gravity=%s; contacts=%d",
        first_height,
        final_state.pelvis_height,
        np.array2string(final_state.projected_gravity, precision=3),
        len(final_state.contacts),
    )


def run_dynamic(env: G1Env, trajectory, headless: bool, speed: float) -> ReplayDiagnostics:
    env.reset(initial_state_from_trajectory(trajectory), seed=0)
    viewer = _viewer(env, headless)
    wall_start = time.monotonic()
    times: list[float] = []
    errors: list[np.ndarray] = []
    references: list[np.ndarray] = []
    actual_positions: list[np.ndarray] = []
    heights: list[float] = []
    root_errors: list[float] = []
    orientation_errors: list[float] = []
    torque_saturated = 0
    torque_values = 0
    action_saturated = 0
    action_values = 0
    left_contacts = right_contacts = non_foot_contacts = 0
    mean_abs_torques: list[np.ndarray] = []
    max_abs_torques: list[np.ndarray] = []
    over_80_fractions: list[np.ndarray] = []
    region_counts = {region: 0 for region in CONTACT_REGIONS}
    region_first_time: dict[str, float] = {}
    region_last_time: dict[str, float] = {}
    try:
        for index in range(1, len(trajectory.time)):
            if viewer is not None and not viewer.is_running():
                break
            action, saturation = env.joint_target_to_action(trajectory.q[index])
            _, effective_reference = env.action_to_joint_target(action)
            result = env.step(action)
            state = result.state
            errors.append(state.joint_positions - effective_reference)
            references.append(effective_reference)
            actual_positions.append(state.joint_positions)
            times.append(float(trajectory.time[index]))
            heights.append(state.pelvis_height)
            action_saturated += int(np.count_nonzero(saturation))
            action_values += saturation.size
            torque_saturated += int(
                round(
                    float(result.info["torque_saturation_fraction"])
                    * env.last_torque.size
                    * env.config.physics_steps_per_control_step
                )
            )
            torque_values += env.last_torque.size * env.config.physics_steps_per_control_step
            mean_abs_torques.append(np.asarray(result.info["mean_abs_torque"]))
            max_abs_torques.append(np.asarray(result.info["max_abs_torque"]))
            over_80_fractions.append(np.asarray(result.info["torque_over_80_fraction"]))
            left_contacts += int(state.left_foot_contact)
            right_contacts += int(state.right_foot_contact)
            non_foot_contacts += int(state.non_foot_contact)
            for region in contact_regions(state.contacts):
                region_counts[region] += 1
                region_first_time.setdefault(region, float(trajectory.time[index]))
                region_last_time[region] = float(trajectory.time[index])
            if trajectory.root_pos is not None:
                root_errors.append(float(np.linalg.norm(state.root_position - trajectory.root_pos[index])))
            if trajectory.root_quat is not None:
                orientation_errors.append(
                    quaternion_angle_error(state.root_quaternion, trajectory.root_quat[index])
                )
            _sync(viewer, wall_start, env.simulation_time, speed)
    finally:
        if viewer is not None:
            viewer.close()
    if not errors:
        raise RuntimeError("dynamic replay produced no samples")
    sample_count = len(errors)
    diagnostics = ReplayDiagnostics(
        time=np.asarray(times),
        joint_error=np.asarray(errors),
        reference_q=np.asarray(references),
        actual_q=np.asarray(actual_positions),
        pelvis_height=np.asarray(heights),
        root_position_error=None if not root_errors else np.asarray(root_errors),
        root_orientation_error=None if not orientation_errors else np.asarray(orientation_errors),
        torque_saturation_fraction=torque_saturated / torque_values,
        action_saturation_fraction=action_saturated / action_values,
        foot_contact_fraction=(left_contacts / sample_count, right_contacts / sample_count),
        non_foot_contact_fraction=non_foot_contacts / sample_count,
        mean_abs_torque=np.mean(np.asarray(mean_abs_torques), axis=0),
        max_abs_torque=np.max(np.asarray(max_abs_torques), axis=0),
        torque_over_80_fraction=np.mean(np.asarray(over_80_fractions), axis=0),
        torque_limits=env.torque_limits.copy(),
        contact_region_fractions={
            region: count / sample_count for region, count in region_counts.items()
        },
        contact_region_first_time=region_first_time,
        contact_region_last_time=region_last_time,
    )
    logging.info("mean joint error: %.4f rad", diagnostics.mean_joint_position_error)
    logging.info("max joint error: %.4f rad", diagnostics.max_joint_position_error)
    logging.info("per-joint RMSE [rad]: %s", np.array2string(diagnostics.per_joint_rmse, precision=3))
    logging.info(
        "pelvis height range: %.3f..%.3f m", diagnostics.pelvis_height.min(), diagnostics.pelvis_height.max()
    )
    if diagnostics.root_position_error is not None:
        logging.info("mean root position error: %.3f m", diagnostics.root_position_error.mean())
    if diagnostics.root_orientation_error is not None:
        logging.info(
            "mean root orientation error: %.2f deg",
            np.rad2deg(diagnostics.root_orientation_error).mean(),
        )
    logging.info("torque saturation fraction: %.3f", diagnostics.torque_saturation_fraction)
    logging.info("unrepresentable action-target fraction: %.3f", diagnostics.action_saturation_fraction)
    logging.info(
        "contact fractions: left_foot=%.3f right_foot=%.3f non_foot=%.3f",
        *diagnostics.foot_contact_fraction,
        diagnostics.non_foot_contact_fraction,
    )
    logging.info(
        "mean/peak commanded torque: %.2f / %.2f N m",
        diagnostics.mean_abs_torque.mean(),
        diagnostics.max_abs_torque.max(),
    )
    logging.info(
        "peak torque utilization by joint: %s",
        np.array2string(diagnostics.peak_torque_utilization, precision=2),
    )
    logging.info(
        "torque limits by joint [N m]: %s",
        np.array2string(diagnostics.torque_limits, precision=1),
    )
    logging.info(
        "fraction above 80%% torque by joint: %s",
        np.array2string(diagnostics.torque_over_80_fraction, precision=3),
    )
    active_regions = {
        region: fraction
        for region, fraction in diagnostics.contact_region_fractions.items()
        if fraction > 0
    }
    logging.info("contact region fractions: %s", active_regions)
    logging.info("first contact times by region: %s", diagnostics.contact_region_first_time)
    logging.info("last contact times by region: %s", diagnostics.contact_region_last_time)
    return diagnostics


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    paths = discover_bones_dataset(args.dataset_root)
    raw = load_motion(args.motion, paths)
    config = G1EnvConfig(
        control_frequency=args.control_frequency,
        episode_duration=max(raw.duration + 1.0, 1.0),
    )
    trajectory = raw.resample(config.control_timestep)
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            raise ValueError("--max-seconds must be positive")
        keep = max(2, int(np.searchsorted(trajectory.time, args.max_seconds, side="right")))
        keep = min(keep, len(trajectory.time))
        trajectory = type(trajectory)(
            time=trajectory.time[:keep], q=trajectory.q[:keep], qd=trajectory.qd[:keep],
            root_pos=None if trajectory.root_pos is None else trajectory.root_pos[:keep],
            root_quat=None if trajectory.root_quat is None else trajectory.root_quat[:keep],
            motion_id=trajectory.motion_id, original_fps=trajectory.original_fps,
            source_path=trajectory.source_path, joint_names=trajectory.joint_names,
            source_metadata=trajectory.source_metadata,
        )
    if args.save_processed is not None:
        trajectory.save(args.save_processed, control_frequency=args.control_frequency)
        logging.info("saved processed trajectory to %s", args.save_processed)
    logging.info(
        "%s: %d input samples (original source %.1f Hz) -> %d samples at %.1f Hz (%.3f s)",
        trajectory.motion_id,
        len(raw.time),
        raw.original_fps,
        len(trajectory.time),
        args.control_frequency,
        trajectory.duration,
    )
    env = G1Env(
        G1EnvConfig(
            control_frequency=args.control_frequency,
            episode_duration=trajectory.duration + config.control_timestep,
        ),
        seed=0,
    )
    representability = analyze_reference_representability(
        trajectory.q, env.joint_limits, env.nominal_pose
    )
    logging.info(
        "reference outside old +/-0.5 rad envelope: %.3f; outside physical limits: %.6f",
        representability.old_envelope_fraction,
        representability.physical_limit_fraction,
    )
    logging.info(
        "old-envelope violation counts by joint: %s",
        dict(zip(JOINT_NAMES, representability.old_envelope_counts.tolist())),
    )
    logging.info(
        "physical-limit violation counts by joint: %s; maximum violation: %.6g rad; "
        "per-joint maxima: %s",
        dict(zip(JOINT_NAMES, representability.physical_limit_counts.tolist())),
        representability.max_physical_violation,
        dict(
            zip(
                JOINT_NAMES,
                representability.max_physical_violation_per_joint.tolist(),
            )
        ),
    )
    if args.mode == "kinematic":
        run_kinematic(env, trajectory, args.headless, args.speed)
        return
    diagnostics = run_dynamic(env, trajectory, args.headless, args.speed)
    if not args.no_plot:
        plot_path = args.plot or Path("artifacts/replay") / trajectory.motion_id / "dynamic_diagnostics.png"
        save_diagnostic_plot(diagnostics, plot_path)
        logging.info("saved diagnostics plot to %s", plot_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
