from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from apex_getup.demonstrations.bones_loader import (
    BONES_JOINT_COLUMNS,
    euler_zyx_degrees_to_wxyz,
    load_bones_csv,
    validate_joint_mapping,
)
from apex_getup.demonstrations.paths import discover_bones_dataset
from apex_getup.demonstrations.replay import (
    analyze_reference_representability,
    initial_state_from_trajectory,
)
from apex_getup.demonstrations.trajectory import ReferenceTrajectory
from apex_getup.env import G1Env
from apex_getup.env.config import JOINT_NAMES


def _write_csv(path: Path, frame_count: int = 13) -> None:
    columns = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        *(BONES_JOINT_COLUMNS[name] for name in reversed(JOINT_NAMES)),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for frame in range(frame_count):
            row = {
                "Frame": frame,
                "root_translateX": frame,
                "root_translateY": 0,
                "root_translateZ": 50 + frame,
                "root_rotateX": 0,
                "root_rotateY": 0,
                "root_rotateZ": frame * 2,
            }
            row.update(
                {
                    BONES_JOINT_COLUMNS[name]: index + frame
                    for index, name in enumerate(JOINT_NAMES)
                }
            )
            writer.writerow(row)


def test_actual_dataset_paths_are_discovered() -> None:
    paths = discover_bones_dataset()
    assert paths.metadata.name == "seed_metadata_v004.parquet"
    assert paths.g1_csv_root.is_dir()


def test_explicit_joint_mapping_uses_environment_order() -> None:
    shuffled = [BONES_JOINT_COLUMNS[name] for name in reversed(JOINT_NAMES)]
    ordered = validate_joint_mapping(shuffled)
    assert ordered == tuple(f"{name}_dof" for name in JOINT_NAMES)
    with pytest.raises(ValueError, match="missing"):
        validate_joint_mapping(shuffled[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_joint_mapping([*shuffled, shuffled[0]])


def test_loading_is_finite_mapped_and_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "motion.csv"
    _write_csv(csv_path)
    first = load_bones_csv(csv_path)
    second = load_bones_csv(csv_path)
    assert first.q.shape == (13, 29)
    assert first.qd.shape == (13, 29)
    assert first.root_pos is not None and first.root_pos.shape == (13, 3)
    assert first.root_quat is not None and first.root_quat.shape == (13, 4)
    np.testing.assert_array_equal(first.q, second.q)
    np.testing.assert_array_equal(first.qd, second.qd)
    np.testing.assert_allclose(first.q[0], np.deg2rad(np.arange(29)))
    assert np.all(np.isfinite(first.q))
    np.testing.assert_allclose(np.linalg.norm(first.root_quat, axis=1), 1.0)
    np.testing.assert_allclose(first.root_pos[0], [0.0, 0.0, 0.5])


def test_euler_zyx_conversion() -> None:
    quaternions = euler_zyx_degrees_to_wxyz(np.array([[0, 0, 0], [0, 0, 90]]))
    np.testing.assert_allclose(quaternions[0], [1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        quaternions[1], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-12
    )


def test_resampling_and_processed_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "motion.csv"
    _write_csv(csv_path)  # 0.1 seconds at 120 Hz
    raw = load_bones_csv(csv_path)
    sampled = raw.resample(0.02)
    np.testing.assert_allclose(np.diff(sampled.time), 0.02)
    assert np.all(np.diff(sampled.time) > 0)
    np.testing.assert_allclose(sampled.q[0], raw.q[0])
    np.testing.assert_allclose(sampled.q[-1], raw.q[-1])
    assert np.all(np.isfinite(sampled.qd))
    assert sampled.root_quat is not None
    np.testing.assert_allclose(np.linalg.norm(sampled.root_quat, axis=1), 1.0)

    saved = sampled.save(tmp_path / "processed.npz", control_frequency=50.0)
    restored = ReferenceTrajectory.load(saved)
    np.testing.assert_array_equal(restored.time, sampled.time)
    np.testing.assert_array_equal(restored.q, sampled.q)
    np.testing.assert_array_equal(restored.root_quat, sampled.root_quat)
    assert restored.source_metadata["control_frequency"] == 50.0


def test_replay_initialization_matches_first_reference() -> None:
    env = G1Env()
    time = np.array([0.0, 0.02])
    trajectory = ReferenceTrajectory(
        time=time,
        q=np.repeat(env.nominal_pose[None, :], 2, axis=0),
        qd=np.zeros((2, 29)),
        root_pos=np.array([[0, 0, 0.5], [0, 0, 0.6]]),
        root_quat=np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
    )
    state = env.reset(initial_state_from_trajectory(trajectory))
    np.testing.assert_allclose(state.joint_positions, trajectory.q[0])
    np.testing.assert_allclose(state.joint_velocities, trajectory.qd[0])
    np.testing.assert_allclose(state.root_position, trajectory.root_pos[0])
    np.testing.assert_allclose(state.root_quaternion, trajectory.root_quat[0])


def test_reference_representability_distinguishes_envelopes() -> None:
    q = np.zeros((2, 29))
    q[0, 0] = 0.75
    q[1, 1] = 1.2
    limits = np.column_stack((-np.ones(29), np.ones(29)))
    report = analyze_reference_representability(q, limits, np.zeros(29))
    assert report.old_envelope_counts[0] == 1
    assert report.old_envelope_counts[1] == 1
    assert report.physical_limit_counts[0] == 0
    assert report.physical_limit_counts[1] == 1
    assert report.max_physical_violation == pytest.approx(0.2)
    assert report.clipped_q[1, 1] == 1.0
