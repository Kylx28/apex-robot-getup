"""Teacher-policy action recordings usable as an APEX action prior."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

@runtime_checkable
class ActionPrior(Protocol):
    """Minimal interface shared by demonstration and teacher action priors."""

    def action_at_time(self, elapsed_seconds: float) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class RecordedActionPrior:
    """Time-indexed normalized actions recorded from a successful teacher."""

    times: NDArray[np.float64]
    actions: NDArray[np.float64]
    joint_targets: NDArray[np.float64]
    metadata: dict[str, object]

    @classmethod
    def from_arrays(
        cls,
        times: ArrayLike,
        actions: ArrayLike,
        joint_targets: ArrayLike,
        metadata: Mapping[str, object],
    ) -> "RecordedActionPrior":
        time_array = np.asarray(times, dtype=np.float64)
        action_array = np.asarray(actions, dtype=np.float64)
        target_array = np.asarray(joint_targets, dtype=np.float64)
        if time_array.ndim != 1 or time_array.size == 0:
            raise ValueError("times must be a non-empty vector")
        if action_array.ndim != 2 or action_array.shape != (time_array.size, 29):
            raise ValueError("actions must have shape [frames, 29]")
        if target_array.shape != action_array.shape:
            raise ValueError("joint_targets must have shape [frames, 29]")
        if not np.all(np.isfinite(time_array)) or not np.all(
            np.isfinite(action_array)
        ) or not np.all(np.isfinite(target_array)):
            raise ValueError("recorded prior data must be finite")
        if time_array[0] < 0.0 or np.any(np.diff(time_array) <= 0.0):
            raise ValueError("times must be non-negative and strictly increasing")
        if np.any(np.abs(action_array) > 1.0 + 1e-9):
            raise ValueError("teacher actions must lie in normalized [-1, 1] space")
        metadata_dict = dict(metadata)
        required = {
            "joint_names",
            "model_path",
            "simulation_timestep",
            "control_period",
            "source_checkpoint",
            "action_semantics",
        }
        if missing := required - set(metadata_dict):
            raise ValueError(f"teacher metadata is missing {sorted(missing)}")
        if len(metadata_dict["joint_names"]) != 29:
            raise ValueError("teacher metadata must contain 29 joint names")
        return cls(
            time_array.copy(),
            action_array.copy(),
            target_array.copy(),
            metadata_dict,
        )

    @property
    def duration_seconds(self) -> float:
        return float(self.times[-1])

    def action_at_time(self, elapsed_seconds: float) -> NDArray[np.float64]:
        if not np.isfinite(elapsed_seconds):
            raise ValueError("elapsed_seconds must be finite")
        query = float(np.clip(elapsed_seconds, self.times[0], self.times[-1]))
        upper = int(np.searchsorted(self.times, query, side="right"))
        if upper == 0:
            return self.actions[0].copy()
        if upper >= self.times.size:
            return self.actions[-1].copy()
        lower = upper - 1
        interval = self.times[upper] - self.times[lower]
        alpha = (query - self.times[lower]) / interval
        return (
            (1.0 - alpha) * self.actions[lower] + alpha * self.actions[upper]
        ).copy()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            times=self.times,
            actions=self.actions,
            joint_targets=self.joint_targets,
            metadata_json=np.asarray(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RecordedActionPrior":
        with np.load(Path(path), allow_pickle=False) as data:
            expected = {"times", "actions", "joint_targets", "metadata_json"}
            if set(data.files) != expected:
                raise ValueError(f"teacher prior must contain {sorted(expected)}")
            return cls.from_arrays(
                data["times"],
                data["actions"],
                data["joint_targets"],
                json.loads(str(data["metadata_json"].item())),
            )


def load_apex_expert(
    path: str | Path,
    task: Any,
    *,
    require_recorded_success: bool = False,
) -> ActionPrior:
    """Load a recorded policy prior or a BONES CSV expert."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"APEX expert not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npz":
        prior = RecordedActionPrior.load(source)
        validate_recorded_prior(
            prior,
            task,
            require_success=require_recorded_success,
        )
        return prior
    if suffix == ".csv":
        from apex_getup.demonstrations import (
            DemonstrationActionPrior,
            load_motion,
        )

        configured = Path(task.config.demonstration_path)
        trajectory = (
            task.reset_trajectory
            if source.resolve() == configured.resolve()
            else load_motion(source)
        )
        prior = DemonstrationActionPrior(
            trajectory,
            task.env,
            interpolate_reference=task.config.interpolate_reference,
        )
        if np.any(prior.physical_limit_violations):
            raise ValueError("BONES expert contains targets outside joint limits")
        return prior
    raise ValueError("APEX expert must be a recorded .npz prior or BONES .csv")


def export_teacher_action_prior(
    task: object,
    policy: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    residual_scale: float,
    source_checkpoint: str | Path,
) -> tuple[RecordedActionPrior, dict[str, float]]:
    """Record physical commands from a deterministic residual teacher."""
    from apex_getup.control import compose_residual_joint_target
    from apex_getup.demonstrations import DemonstrationActionPrior
    from apex_getup.env.config import JOINT_NAMES
    from apex_getup.env.g1_env import default_model_path

    observation = task.reset(seed=0)
    reference = DemonstrationActionPrior(
        task.reset_trajectory,
        task.env,
        interpolate_reference=task.config.interpolate_reference,
    )
    times: list[float] = []
    actions: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    episode_return = 0.0
    last_info: dict[str, object] = {}
    while True:
        policy_action = np.asarray(policy(observation), dtype=np.float64)
        target, _ = compose_residual_joint_target(
            policy_action,
            reference.joint_target_at_time(task.current_reference_time),
            residual_scale,
            task.env.joint_limits,
        )
        normalized, clipped = task.env.joint_target_to_action(target)
        if np.any(clipped):
            raise RuntimeError("teacher target exceeds physical joint limits")
        times.append(task.current_reference_time)
        actions.append(normalized)
        targets.append(target)
        result = task.step(
            normalized,
            policy_action=policy_action,
            physical_joint_target=target,
        )
        episode_return += result.reward
        last_info = result.info
        observation = result.observation
        if result.terminated or result.truncated:
            break
    model_path = Path(
        task.env.config.model_path or default_model_path()
    ).resolve()
    prior = RecordedActionPrior.from_arrays(
        times,
        actions,
        targets,
        {
            "joint_names": list(JOINT_NAMES),
            "model_path": str(model_path),
            "simulation_timestep": task.config.simulation_timestep,
            "control_period": task.control_timestep,
            "source_checkpoint": str(source_checkpoint),
            "action_semantics": "normalized_physical_joint_target",
        },
    )
    return prior, {
        "episode_return": float(episode_return),
        "success": float(last_info.get("episode_success", False)),
    }


def validate_recorded_prior(
    prior: RecordedActionPrior,
    task: object,
    *,
    require_success: bool = False,
) -> None:
    """Validate teacher metadata against a task instance."""
    from apex_getup.env.config import JOINT_NAMES
    from apex_getup.env.g1_env import default_model_path

    metadata = prior.metadata
    model_path = Path(task.env.config.model_path or default_model_path()).resolve()
    if tuple(metadata["joint_names"]) != JOINT_NAMES:
        raise ValueError("teacher joint order does not match the G1 model")
    if metadata["action_semantics"] != "normalized_physical_joint_target":
        raise ValueError("teacher action semantics are incompatible")
    if Path(str(metadata["model_path"])).resolve() != model_path:
        raise ValueError("teacher model does not match the task model")
    if not np.isclose(
        float(metadata["simulation_timestep"]), task.config.simulation_timestep
    ):
        raise ValueError("teacher simulation timestep does not match the task")
    if not np.isclose(float(metadata["control_period"]), task.control_timestep):
        raise ValueError("teacher control period does not match the task")
    if require_success:
        for name in ("teacher_metrics", "replay_metrics"):
            metrics = metadata.get(name)
            if not isinstance(metrics, Mapping) or not metrics.get("success"):
                raise ValueError(f"teacher prior has no successful {name}")
