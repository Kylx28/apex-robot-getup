"""Validated, raw-format-independent demonstration trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex_getup.env.config import JOINT_NAMES, NUM_ACTUATORS


def reference_index_for_episode_step(
    frame_count: int, episode_step: int, *, start_index: int = 0
) -> int:
    """Return the clamped reference target index for one control transition.

    Control step zero advances the simulation from the reset state at
    ``start_index`` toward the next reference sample.  Keeping this indexing in
    one function prevents the action prior and reference-conditioned
    observation from drifting apart.
    """
    if frame_count < 2:
        raise ValueError("a reference trajectory requires at least two frames")
    if episode_step < 0 or not 0 <= start_index < frame_count:
        raise ValueError("episode_step/start_index is outside the reference")
    return min(start_index + episode_step + 1, frame_count - 1)


@dataclass(frozen=True)
class ReferenceSample:
    """One trajectory state interpolated at an absolute reference time."""

    time: float
    phase: float
    q: NDArray[np.float64]
    qd: NDArray[np.float64]
    root_pos: NDArray[np.float64] | None
    root_quat: NDArray[np.float64] | None
    lower_index: int
    upper_index: int
    interpolation: float


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


def _quaternion_multiply(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Multiply two ``wxyz`` quaternions."""
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


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

    def sample(
        self, query_time: float, *, interpolate: bool = True
    ) -> ReferenceSample:
        """Query the native trajectory at ``query_time`` in seconds.

        When ``interpolate`` is true, vectors are linearly interpolated and
        root orientation uses SLERP. When false, the query uses the paper's
        nearest-frame rule, ``round(t / reference_dt)``. Times outside the
        motion are clamped. The final sample retains its backward-difference
        joint velocity, matching the released environment.
        """
        if not np.isfinite(query_time):
            raise ValueError("reference query time must be finite")
        clamped_time = float(np.clip(query_time, self.time[0], self.time[-1]))
        if not interpolate:
            reference_dt = float(np.median(np.diff(self.time)))
            index = int(round(clamped_time / reference_dt))
            index = int(np.clip(index, 0, len(self.time) - 1))
            return ReferenceSample(
                time=float(self.time[index]),
                phase=index / max(1, len(self.time) - 1),
                q=self.q[index].copy(),
                qd=self.qd[index].copy(),
                root_pos=(
                    None if self.root_pos is None else self.root_pos[index].copy()
                ),
                root_quat=(
                    None if self.root_quat is None else self.root_quat[index].copy()
                ),
                lower_index=index,
                upper_index=index,
                interpolation=0.0,
            )
        if clamped_time <= self.time[0]:
            lower = upper = 0
            amount = 0.0
        elif clamped_time >= self.time[-1]:
            lower = upper = len(self.time) - 1
            amount = 0.0
        else:
            upper = int(np.searchsorted(self.time, clamped_time, side="right"))
            lower = upper - 1
            amount = float(
                (clamped_time - self.time[lower])
                / (self.time[upper] - self.time[lower])
            )
            if amount <= 1e-12:
                upper = lower
                amount = 0.0
            elif 1.0 - amount <= 1e-12:
                lower = upper
                amount = 0.0

        def interpolate(values: NDArray[np.float64]) -> NDArray[np.float64]:
            if lower == upper:
                return values[lower].copy()
            return ((1.0 - amount) * values[lower] + amount * values[upper]).copy()

        q = interpolate(self.q)
        qd = interpolate(self.qd)
        root_pos = None if self.root_pos is None else interpolate(self.root_pos)
        root_quat = None
        if self.root_quat is not None:
            root_quat = (
                self.root_quat[lower].copy()
                if lower == upper
                else _slerp_pair(self.root_quat[lower], self.root_quat[upper], amount)
            )
        return ReferenceSample(
            time=clamped_time,
            phase=clamped_time / self.duration,
            q=q,
            qd=qd,
            root_pos=root_pos,
            root_quat=root_quat,
            lower_index=lower,
            upper_index=upper,
            interpolation=amount,
        )

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
        qd = np.zeros_like(q)
        qd[1:-1] = (q[2:] - q[:-2]) / (2.0 * control_dt)
        qd[0] = (q[1] - q[0]) / control_dt
        qd[-1] = (q[-1] - q[-2]) / control_dt
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

    def with_stationary_terminal_velocity(self) -> "ReferenceTrajectory":
        """Return a copy whose final joint velocity is exactly zero.

        Reference consumers clamp to the final sample after a demonstration
        ends.  A finite-difference endpoint velocity is inappropriate during
        that hold: the target pose is stationary, so its target velocity must
        be stationary as well.
        """
        qd = self.qd.copy()
        qd[-1] = 0.0
        metadata = dict(self.source_metadata)
        metadata["stationary_terminal_velocity"] = True
        return ReferenceTrajectory(
            time=self.time,
            q=self.q,
            qd=qd,
            root_pos=self.root_pos,
            root_quat=self.root_quat,
            motion_id=self.motion_id,
            original_fps=self.original_fps,
            source_path=self.source_path,
            joint_names=self.joint_names,
            source_metadata=metadata,
        )

    def align_first_root_pose(
        self, target_position: object, target_quaternion: object
    ) -> "ReferenceTrajectory":
        """Rigidly align the root trajectory's first pose to a target pose.

        Joint motion is left untouched. Translation differences and one fixed
        world-frame quaternion offset are applied to every root sample.
        """
        if self.root_pos is None or self.root_quat is None:
            raise ValueError("root alignment requires root position and orientation")
        position = np.asarray(target_position, dtype=np.float64)
        quaternion = np.asarray(target_quaternion, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("target_position must be finite with shape (3,)")
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("target_quaternion must be finite with shape (4,)")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-12:
            raise ValueError("target_quaternion must be nonzero")
        quaternion = quaternion / norm
        first_inverse = self.root_quat[0] * np.array([1.0, -1.0, -1.0, -1.0])
        offset = _quaternion_multiply(quaternion, first_inverse)
        root_quat = np.stack(
            [_quaternion_multiply(offset, value) for value in self.root_quat]
        )
        root_pos = position + (self.root_pos - self.root_pos[0])
        metadata = dict(self.source_metadata)
        metadata.update(
            {
                "root_aligned_to_fallen": True,
                "root_alignment_target_position": position.tolist(),
                "root_alignment_target_quaternion": quaternion.tolist(),
            }
        )
        return ReferenceTrajectory(
            time=self.time,
            q=self.q,
            qd=self.qd,
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
