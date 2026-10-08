"""Loader for BONES-SEED's retargeted Unitree G1 CSV format.

The official BONES viewer defines this format as 120 Hz, Z-up, centimeters for
root translation, and degrees for root ZYX Euler angles and joint DoFs.
"""

from __future__ import annotations

import csv
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import NDArray
import pyarrow.compute as pc
import pyarrow.parquet as pq

from apex_getup.demonstrations.paths import BonesDatasetPaths, discover_bones_dataset
from apex_getup.demonstrations.trajectory import ReferenceTrajectory, normalize_quaternions
from apex_getup.env.config import JOINT_NAMES, NUM_ACTUATORS


BONES_G1_FPS = 120.0
BONES_JOINT_COLUMNS: dict[str, str] = {name: f"{name}_dof" for name in JOINT_NAMES}
ROOT_POSITION_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_EULER_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")


def validate_joint_mapping(columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return BONES columns in environment order, failing on missing/duplicate mappings."""
    if len(BONES_JOINT_COLUMNS) != NUM_ACTUATORS or set(BONES_JOINT_COLUMNS) != set(JOINT_NAMES):
        raise RuntimeError("BONES mapping must contain each of the 29 environment joints once")
    if len(set(BONES_JOINT_COLUMNS.values())) != NUM_ACTUATORS:
        raise RuntimeError("BONES mapping contains duplicate source columns")
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(f"CSV contains duplicate columns: {duplicates}")
    missing = [source for source in BONES_JOINT_COLUMNS.values() if source not in columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} required G1 joint columns: {missing}")
    ordered = tuple(BONES_JOINT_COLUMNS[name] for name in JOINT_NAMES)
    if len(ordered) != NUM_ACTUATORS:
        raise RuntimeError("ordered BONES mapping is not exactly 29-dimensional")
    return ordered


def euler_zyx_degrees_to_wxyz(euler_xyz_degrees: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert degrees `(x,y,z)` to `wxyz` for `R = Rz(z) Ry(y) Rx(x)`."""
    angles = np.deg2rad(np.asarray(euler_xyz_degrees, dtype=np.float64)) * 0.5
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    result = np.column_stack(
        (
            cz * cy * cx + sz * sy * sx,
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
        )
    )
    return normalize_quaternions(result)


def _numeric_column(rows: list[dict[str, str]], column: str) -> NDArray[np.float64]:
    try:
        values = np.array([float(row[column]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"column {column!r} contains missing or non-numeric data") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} contains NaN or Inf")
    return values


def finite_difference_velocity(
    positions: NDArray[np.float64], timestep: float
) -> NDArray[np.float64]:
    """Match the released environment's trajectory velocity convention.

    Interior samples use a centered difference. The first and last samples
    use forward and backward differences respectively, so frame zero has the
    motion's initial velocity and the terminal velocity remains defined while
    the final reference frame is held.
    """
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("positions must be a finite 2-D array with at least two rows")
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("timestep must be positive and finite")
    velocity = np.zeros_like(values)
    velocity[1:-1] = (values[2:] - values[:-2]) / (2.0 * timestep)
    velocity[0] = (values[1] - values[0]) / timestep
    velocity[-1] = (values[-1] - values[-2]) / timestep
    return velocity


def load_bones_csv(
    path: Path | str,
    *,
    motion_id: str | None = None,
    original_fps: float = BONES_G1_FPS,
    metadata: dict[str, object] | None = None,
) -> ReferenceTrajectory:
    """Load a BONES G1 CSV into MuJoCo joint order and SI/radian units."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"BONES G1 CSV not found: {source}")
    if not np.isfinite(original_fps) or original_fps <= 0:
        raise ValueError("original_fps must be positive and finite")
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or ())
        ordered_joint_columns = validate_joint_mapping(columns)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("BONES trajectory must contain at least two frames")

    relevant = {"Frame", *ROOT_POSITION_COLUMNS, *ROOT_EULER_COLUMNS, *ordered_joint_columns}
    unknown_columns = tuple(column for column in columns if column not in relevant)
    if unknown_columns:
        warnings.warn(
            f"preserving {len(unknown_columns)} unrecognized CSV column names in metadata: "
            f"{unknown_columns}",
            stacklevel=2,
        )
    if "Frame" not in columns:
        raise ValueError("CSV is missing required Frame column")
    frames = _numeric_column(rows, "Frame")
    if not np.all(np.diff(frames) > 0):
        raise ValueError("Frame values must be strictly increasing")
    time = (frames - frames[0]) / original_fps

    # BONES G1 joint DoFs are degrees; environment joints are radians.
    q = np.deg2rad(np.column_stack([_numeric_column(rows, name) for name in ordered_joint_columns]))
    # Match the released fixed-rate forward/central/backward convention. In
    # particular, frame zero is not NumPy's second-order endpoint estimate.
    qd = finite_difference_velocity(q, 1.0 / original_fps)

    root_position_presence = [name in columns for name in ROOT_POSITION_COLUMNS]
    if any(root_position_presence) and not all(root_position_presence):
        raise ValueError("root translation requires all X/Y/Z columns")
    root_pos = None
    if all(root_position_presence):
        # BONES roots are Z-up centimeters; MuJoCo is Z-up meters.
        root_pos = 0.01 * np.column_stack(
            [_numeric_column(rows, name) for name in ROOT_POSITION_COLUMNS]
        )
    root_euler_presence = [name in columns for name in ROOT_EULER_COLUMNS]
    if any(root_euler_presence) and not all(root_euler_presence):
        raise ValueError("root orientation requires all rotateX/Y/Z columns")
    root_quat = None
    if all(root_euler_presence):
        root_euler = np.column_stack([_numeric_column(rows, name) for name in ROOT_EULER_COLUMNS])
        root_quat = euler_zyx_degrees_to_wxyz(root_euler)

    source_metadata = dict(metadata or {})
    source_metadata.update(
        {
            "raw_columns": columns,
            "unknown_columns": list(unknown_columns),
            "frame_start": float(frames[0]),
            "frame_end": float(frames[-1]),
            "joint_angle_unit": "degrees",
            "root_translation_unit": "centimeters",
            "root_euler_unit": "degrees",
            "root_euler_order": "ZYX (Rz*Ry*Rx)",
            "coordinate_system": "right-handed Z-up",
        }
    )
    return ReferenceTrajectory(
        time=time,
        q=q,
        qd=qd,
        root_pos=root_pos,
        root_quat=root_quat,
        motion_id=motion_id or source.stem,
        original_fps=float(original_fps),
        source_path=source,
        joint_names=JOINT_NAMES,
        source_metadata=source_metadata,
    )


def _metadata_match(motion: str, paths: BonesDatasetPaths) -> dict[str, object]:
    columns = [
        "move_name", "filename", "move_g1_path", "move_duration_frames",
        "category", "package", "is_mirror", "content_short_description",
        "content_natural_desc_1", "content_type_of_movement", "content_body_position",
    ]
    table = pq.read_table(paths.metadata, columns=columns)
    mask = pc.or_(pc.equal(table["filename"], motion), pc.equal(table["move_name"], motion))
    matches = table.filter(mask).to_pylist()
    if not matches:
        raise FileNotFoundError(
            f"no BONES metadata row with filename or move_name {motion!r}; "
            "run scripts/find_getup_demos.py"
        )
    unique_paths = {row["move_g1_path"] for row in matches if row["move_g1_path"]}
    if len(unique_paths) != 1:
        raise ValueError(f"motion {motion!r} maps ambiguously to G1 paths: {sorted(unique_paths)}")
    return matches[0]


def resolve_motion_path(
    motion: Path | str,
    dataset: BonesDatasetPaths | None = None,
) -> tuple[Path, dict[str, object]]:
    """Resolve an explicit CSV/NPZ path or an exact metadata motion identifier."""
    candidate = Path(motion).expanduser()
    if candidate.is_file():
        return candidate.resolve(), {}
    paths = dataset or discover_bones_dataset()
    row = _metadata_match(str(motion), paths)
    resolved = paths.root / str(row["move_g1_path"])
    if not resolved.is_file():
        raise FileNotFoundError(f"metadata points to missing G1 CSV: {resolved}")
    return resolved.resolve(), row


def load_motion(
    motion: Path | str,
    dataset: BonesDatasetPaths | None = None,
    *,
    original_fps: float = BONES_G1_FPS,
) -> ReferenceTrajectory:
    path, metadata = resolve_motion_path(motion, dataset)
    if path.suffix.lower() == ".npz":
        return ReferenceTrajectory.load(path)
    return load_bones_csv(
        path,
        motion_id=str(metadata.get("filename") or path.stem),
        original_fps=original_fps,
        metadata=metadata,
    )

