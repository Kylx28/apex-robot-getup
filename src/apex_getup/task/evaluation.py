"""Common deterministic policy and static-baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from apex_getup.task.getup_task import GetUpTask


ActionAdapter = Callable[
    [GetUpTask, NDArray[np.float64]],
    tuple[NDArray[np.float64], dict[str, object]],
]


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
    action_adapter: ActionAdapter | None = None,
    observation_transform: Callable[
        [NDArray[np.float64]], NDArray[np.float64]
    ] | None = None,
) -> EvaluationResult:
    observation = task.reset(seed=seed)
    if observation_transform is not None:
        observation = observation_transform(observation)
    initial_pelvis_height = task.initial_pelvis_height
    initial_uprightness = task.initial_uprightness
    trace_names = (
        "total_reward", "pose_tracking", "velocity_tracking",
        "root_xy_tracking", "pelvis_height_tracking", "orientation_tracking",
        "standing_pose", "uprightness_reward", "standing_height",
        "foot_slip_penalty", "action_penalty", "action_smoothness_penalty",
        "weighted_pose_tracking", "weighted_velocity_tracking",
        "weighted_root_xy_tracking", "weighted_pelvis_height_tracking",
        "weighted_orientation_tracking", "weighted_standing_pose",
        "weighted_uprightness", "weighted_standing_height",
        "weighted_foot_slip_penalty", "weighted_action_penalty",
        "weighted_action_smoothness_penalty", "w_track", "w_stand",
        "standing_phase_weight", "weighted_tracking_total",
        "weighted_final_standing_total", "reward_constant",
        "stable_standing_indicator", "consecutive_standing_duration",
        "pelvis_height", "uprightness", "root_linear_speed",
        "root_angular_speed", "action_magnitude",
        "action_delta_magnitude", "mean_torque",
        "torque_saturation_fraction",
        "prior_coefficient", "mean_absolute_policy_action",
        "mean_absolute_prior_action", "mean_absolute_executed_action",
        "executed_action_clipping_fraction",
        "mean_absolute_scaled_policy_action", "scaled_policy_to_prior_ratio",
    )
    traces: dict[str, list[float]] = {name: [] for name in trace_names}
    action_traces: dict[str, list[NDArray[np.float64]]] = {
        "policy_action": [],
        "prior_action": [],
        "executed_action": [],
        "scaled_policy_action": [],
    }
    episode_return = 0.0
    maximum_height = -np.inf
    maximum_uprightness = -np.inf
    last_info: dict[str, object] | None = None
    while True:
        policy_action = np.asarray(policy(observation), dtype=np.float64)
        if policy_action.shape != (task.action_size,) or not np.all(
            np.isfinite(policy_action)
        ):
            raise ValueError("evaluation policy must return a finite shape-(29,) action")
        if action_adapter is None:
            executed_action = policy_action.copy()
            prior_action = np.zeros_like(policy_action)
            prior_coefficient = 0.0
            policy_action_scale = 1.0
            clipping_mask = np.zeros_like(policy_action, dtype=np.bool_)
            physical_joint_target = None
        else:
            executed_action, action_info = action_adapter(task, policy_action)
            executed_action = np.asarray(executed_action, dtype=np.float64)
            prior_action = np.asarray(action_info["prior_action"], dtype=np.float64)
            prior_coefficient = float(action_info["prior_coefficient"])
            policy_action_scale = float(action_info.get("policy_action_scale", 1.0))
            clipping_mask = np.asarray(
                action_info["executed_action_clipped"], dtype=np.bool_
            )
            physical_joint_target = action_info.get("physical_joint_target")
        if executed_action.shape != (task.action_size,) or not np.all(
            np.isfinite(executed_action)
        ):
            raise ValueError("executed evaluation action must be finite with shape (29,)")
        result = task.step(
            executed_action,
            policy_action=policy_action,
            physical_joint_target=physical_joint_target,
        )
        observation = result.observation
        if observation_transform is not None:
            observation = observation_transform(observation)
        last_info = result.info
        episode_return += result.reward
        maximum_height = max(maximum_height, float(result.info["pelvis_height"]))
        maximum_uprightness = max(
            maximum_uprightness, float(result.info["uprightness"])
        )
        action_values = {
            "prior_coefficient": prior_coefficient,
            "mean_absolute_policy_action": float(np.mean(np.abs(policy_action))),
            "mean_absolute_prior_action": float(np.mean(np.abs(prior_action))),
            "mean_absolute_executed_action": float(np.mean(np.abs(executed_action))),
            "executed_action_clipping_fraction": float(np.mean(clipping_mask)),
            "mean_absolute_scaled_policy_action": float(
                np.mean(np.abs(policy_action_scale * policy_action))
            ),
            "scaled_policy_to_prior_ratio": float(
                np.mean(np.abs(policy_action_scale * policy_action))
                / float(np.mean(np.abs(prior_action)))
                if float(np.mean(np.abs(prior_action))) > 1e-12
                else 0.0
            ),
        }
        action_traces["policy_action"].append(policy_action.copy())
        action_traces["prior_action"].append(prior_action.copy())
        action_traces["executed_action"].append(executed_action.copy())
        action_traces["scaled_policy_action"].append(
            policy_action_scale * policy_action
        )
        for name in trace_names:
            traces[name].append(
                action_values[name] if name in action_values else float(result.info[name])
            )
        if step_callback is not None:
            step_callback(task)
        if result.terminated or result.truncated:
            break
    assert last_info is not None
    metrics = {
        "episode_return": episode_return,
        "initial_pelvis_height": initial_pelvis_height,
        "initial_uprightness": initial_uprightness,
        "success": float(last_info["episode_success"]),
        "time_to_stand": np.nan
        if last_info["first_success_time"] is None
        else float(last_info["first_success_time"]),
        "final_pelvis_height": float(last_info["pelvis_height"]),
        "maximum_pelvis_height": maximum_height,
        "final_uprightness": float(last_info["uprightness"]),
        "maximum_uprightness": maximum_uprightness,
        "final_root_linear_speed": float(traces["root_linear_speed"][-1]),
        "final_root_angular_speed": float(traces["root_angular_speed"][-1]),
        "stable_standing_fraction": float(
            np.mean(traces["stable_standing_indicator"])
        ),
        "maximum_consecutive_standing_duration": float(
            np.max(traces["consecutive_standing_duration"])
        ),
        "action_magnitude": float(np.mean(traces["action_magnitude"])),
        "action_smoothness": float(np.mean(traces["action_delta_magnitude"])),
        "mean_torque": float(np.mean(traces["mean_torque"])),
        "torque_saturation": float(np.mean(traces["torque_saturation_fraction"])),
        "prior_coefficient": float(np.mean(traces["prior_coefficient"])),
        "mean_absolute_policy_action": float(
            np.mean(traces["mean_absolute_policy_action"])
        ),
        "mean_absolute_prior_action": float(
            np.mean(traces["mean_absolute_prior_action"])
        ),
        "mean_absolute_executed_action": float(
            np.mean(traces["mean_absolute_executed_action"])
        ),
        "executed_action_clipping_fraction": float(
            np.mean(traces["executed_action_clipping_fraction"])
        ),
        "mean_absolute_scaled_policy_action": float(
            np.mean(traces["mean_absolute_scaled_policy_action"])
        ),
        "scaled_policy_to_prior_ratio": float(
            np.mean(traces["scaled_policy_to_prior_ratio"])
        ),
    }
    for name in (
        "pose_tracking", "velocity_tracking", "root_xy_tracking",
        "pelvis_height_tracking", "orientation_tracking", "standing_pose",
        "uprightness_reward", "standing_height", "foot_slip_penalty",
        "action_penalty", "action_smoothness_penalty", "w_track", "w_stand",
        "standing_phase_weight", "weighted_tracking_total",
        "weighted_final_standing_total", "reward_constant",
    ):
        metrics[f"mean_{name}"] = float(np.mean(traces[name]))
    return EvaluationResult(
        metrics=metrics,
        traces={
            **{name: np.asarray(values) for name, values in traces.items()},
            **{name: np.asarray(values) for name, values in action_traces.items()},
        },
    )
