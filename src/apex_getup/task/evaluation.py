"""Common deterministic policy and static-baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from apex_getup.task.getup_task import GetUpTask


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    traces: dict[str, NDArray[np.float64]]


def evaluate_episode(
    task: GetUpTask,
    policy: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    seed: int = 0,
    step_callback: Callable[[GetUpTask], None] | None = None,
) -> EvaluationResult:
    observation = task.reset(seed=seed)
    trace_names = (
        "total_reward", "height_reward", "uprightness_reward",
        "height_progress_reward", "standing_bonus", "action_penalty",
        "action_smoothness_penalty", "torque_penalty", "pelvis_height",
        "uprightness", "action_magnitude", "action_delta_magnitude", "mean_torque",
        "torque_saturation_fraction",
    )
    traces: dict[str, list[float]] = {name: [] for name in trace_names}
    episode_return = 0.0
    maximum_height = -np.inf
    maximum_uprightness = -np.inf
    last_info: dict[str, object] | None = None
    while True:
        action = np.asarray(policy(observation), dtype=np.float64)
        if action.shape != (task.action_size,) or not np.all(np.isfinite(action)):
            raise ValueError("evaluation policy must return a finite shape-(29,) action")
        result = task.step(action)
        observation = result.observation
        last_info = result.info
        episode_return += result.reward
        maximum_height = max(maximum_height, float(result.info["pelvis_height"]))
        maximum_uprightness = max(
            maximum_uprightness, float(result.info["uprightness"])
        )
        for name in trace_names:
            traces[name].append(float(result.info[name]))
        if step_callback is not None:
            step_callback(task)
        if result.terminated or result.truncated:
            break
    assert last_info is not None
    metrics = {
        "episode_return": episode_return,
        "success": float(last_info["episode_success"]),
        "time_to_stand": np.nan
        if last_info["first_success_time"] is None
        else float(last_info["first_success_time"]),
        "final_pelvis_height": float(last_info["pelvis_height"]),
        "maximum_pelvis_height": maximum_height,
        "final_uprightness": float(last_info["uprightness"]),
        "maximum_uprightness": maximum_uprightness,
        "action_magnitude": float(np.mean(traces["action_magnitude"])),
        "action_smoothness": float(np.mean(traces["action_delta_magnitude"])),
        "mean_torque": float(np.mean(traces["mean_torque"])),
        "torque_saturation": float(np.mean(traces["torque_saturation_fraction"])),
    }
    return EvaluationResult(
        metrics=metrics,
        traces={name: np.asarray(values) for name, values in traces.items()},
    )
