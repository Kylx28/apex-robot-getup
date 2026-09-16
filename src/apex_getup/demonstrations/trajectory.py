"""Validated, raw-format-independent demonstration trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex_getup.env.config import JOINT_NAMES, NUM_ACTUATORS


def _readonly_float_array(value: object, shape: tuple[int | None, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def normalize_quaternions(quaternions: object) -> NDArray[np.float64]:
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.all(np.isfinite(q)):
        raise ValueError("root_quat must be finite with shape (T, 4)")
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("root_quat contains a zero quaternion")
    q = q / norms[:, None]
    # q and -q are the same rotation; choose a continuous representation.
    for index in range(1, len(q)):
        if np.dot(q[index - 1], q[index]) < 0:
            q[index] *= -1
    return q


def _slerp_pair(q0: NDArray[np.float64], q1: NDArray[np.float64], amount: float) -> NDArray[np.float64]:
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + amount * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    return (
        np.sin((1.0 - amount) * angle) * q0 + np.sin(amount * angle) * q1
    ) / np.sin(angle)


def slerp_series(
    source_time: NDArray[np.float64],
    quaternions: NDArray[np.float64],
    target_time: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = np.empty((len(target_time), 4), dtype=np.float64)
    for output_index, timestamp in enumerate(target_time):
        if timestamp <= source_time[0]:
            result[output_index] = quaternions[0]
            continue
        if timestamp >= source_time[-1]:
            result[output_index] = quaternions[-1]
            continue
        upper = int(np.searchsorted(source_time, timestamp, side="right"))
        lower = upper - 1
        amount = float(
            (timestamp - source_time[lower]) / (source_time[upper] - source_time[lower])
        )
        result[output_index] = _slerp_pair(quaternions[lower], quaternions[upper], amount)
    return normalize_quaternions(result)


@dataclass(frozen=True)
class ReferenceTrajectory:
    time: NDArray[np.float64]
    q: NDArray[np.float64]
    qd: NDArray[np.float64]
    root_pos: NDArray[np.float64] | None = None
    root_quat: NDArray[np.float64] | None = None
    motion_id: str = "unknown"
    original_fps: float = 0.0
    source_path: Path | None = None
    joint_names: tuple[str, ...] = JOINT_NAMES
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _readonly_float_array(self.time, (None,), "time")
        if len(time) < 2 or not np.all(np.diff(time) > 0):
            raise ValueError("time must contain at least two strictly increasing samples")
        if not np.isclose(time[0], 0.0, atol=1e-12):
            raise ValueError("time must start at zero")
        q = _readonly_float_array(self.q, (len(time), NUM_ACTUATORS), "q")
        qd = _readonly_float_array(self.qd, (len(time), NUM_ACTUATORS), "qd")
        if tuple(self.joint_names) != JOINT_NAMES:
            raise ValueError("joint_names must exactly match the environment actuator order")
        root_pos = None
        if self.root_pos is not None:
            root_pos = _readonly_float_array(self.root_pos, (len(time), 3), "root_pos")
        root_quat = None
        if self.root_quat is not None:
            root_quat = _readonly_float_array(
                normalize_quaternions(self.root_quat), (len(time), 4), "root_quat"
            )
        if self.original_fps < 0 or not np.isfinite(self.original_fps):
            raise ValueError("original_fps must be non-negative and finite")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", qd)
        object.__setattr__(self, "root_pos", root_pos)
        object.__setattr__(self, "root_quat", root_quat)
        object.__setattr__(self, "source_path", None if self.source_path is None else Path(self.source_path))
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata))

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    @property
    def original_timestep(self) -> float | None:
        return None if self.original_fps == 0 else 1.0 / self.original_fps

    def resample(self, control_dt: float) -> "ReferenceTrajectory":
        """Resample onto exact ``k * control_dt`` timestamps using SLERP for orientation."""
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")
        intervals = max(1, int(round(self.duration / control_dt)))
        target_time = np.arange(intervals + 1, dtype=np.float64) * control_dt
        q = np.column_stack(
            [np.interp(target_time, self.time, self.q[:, index]) for index in range(NUM_ACTUATORS)]
        )
        root_pos = None
        if self.root_pos is not None:
            root_pos = np.column_stack(
                [np.interp(target_time, self.time, self.root_pos[:, index]) for index in range(3)]
            )
        root_quat = None
        if self.root_quat is not None:
            root_quat = slerp_series(self.time, self.root_quat, target_time)
        edge_order = 2 if len(target_time) >= 3 else 1
        qd = np.gradient(q, target_time, axis=0, edge_order=edge_order)
        metadata = dict(self.source_metadata)
        metadata["resampled_from_samples"] = len(self.time)
        return ReferenceTrajectory(
            time=target_time,
            q=q,
            qd=qd,
            root_pos=root_pos,
            root_quat=root_quat,
            motion_id=self.motion_id,
            original_fps=self.original_fps,
            source_path=self.source_path,
            joint_names=self.joint_names,
            source_metadata=metadata,
        )

    def save(self, path: Path | str, control_frequency: float | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.source_metadata)
        metadata.update(
            {
                "motion_id": self.motion_id,
                "original_fps": self.original_fps,
                "source_path": None if self.source_path is None else str(self.source_path),
                "joint_names": list(self.joint_names),
                "control_frequency": control_frequency,
            }
        )
        np.savez_compressed(
            destination,
            time=self.time,
            q=self.q,
            qd=self.qd,
            root_pos=np.empty((0, 3)) if self.root_pos is None else self.root_pos,
            root_quat=np.empty((0, 4)) if self.root_quat is None else self.root_quat,
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> "ReferenceTrajectory":
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            root_pos_data = archive["root_pos"]
            root_quat_data = archive["root_quat"]
            known = {"motion_id", "original_fps", "source_path", "joint_names", "control_frequency"}
            source_metadata = {key: value for key, value in metadata.items() if key not in known}
            if metadata.get("control_frequency") is not None:
                source_metadata["control_frequency"] = metadata["control_frequency"]
            return cls(
                time=archive["time"],
                q=archive["q"],
                qd=archive["qd"],
                root_pos=None if len(root_pos_data) == 0 else root_pos_data,
                root_quat=None if len(root_quat_data) == 0 else root_quat_data,
                motion_id=metadata["motion_id"],
                original_fps=float(metadata["original_fps"]),
                source_path=None if metadata["source_path"] is None else Path(metadata["source_path"]),
                joint_names=tuple(metadata["joint_names"]),
                source_metadata=source_metadata,
            )

