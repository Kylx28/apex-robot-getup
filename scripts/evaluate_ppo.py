#!/usr/bin/env python3
"""Deterministically evaluate a task-only PPO checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from apex_getup.rl import PPOAgent
from apex_getup.task import GetUpTask, GetUpTaskConfig, evaluate_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-duration", type=float, default=10.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-gif", type=Path, default=None)
    parser.add_argument("--reward-plot", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def save_reward_plot(traces: dict[str, np.ndarray], path: Path, control_dt: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/apex-getup-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(len(traces["total_reward"])) * control_dt
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for name in (
        "total_reward", "height_reward", "uprightness_reward",
        "height_progress_reward", "standing_bonus",
    ):
        axes[0].plot(time_axis, traces[name], label=name)
    axes[0].legend(ncol=2)
    for name in ("action_penalty", "action_smoothness_penalty", "torque_penalty"):
        axes[1].plot(time_axis, traces[name], label=name)
    axes[1].legend()
    axes[2].plot(time_axis, traces["pelvis_height"], label="pelvis height")
    axes[2].plot(time_axis, traces["uprightness"], label="uprightness")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlabel("time (s)")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    agent, checkpoint_metadata = PPOAgent.load(args.checkpoint)
    results: list[dict[str, float]] = []
    representative = None
    for episode in range(args.episodes):
        task = GetUpTask(
            GetUpTaskConfig(episode_duration=args.episode_duration), seed=episode
        )
        viewer = None
        renderer = None
        camera = None
        frames = []
        callback = None
        if args.render and episode == 0:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(task.env.model, task.env.data)
            wall_start = time.monotonic()

            def viewer_callback(current_task: GetUpTask) -> None:
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep(
                        max(
                            0.0,
                            wall_start
                            + current_task.env.simulation_time
                            - time.monotonic(),
                        )
                    )

            callback = viewer_callback

        if args.record_gif is not None and episode == 0:
            import mujoco

            renderer = mujoco.Renderer(task.env.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = 2.5
            camera.azimuth = 135
            camera.elevation = -20
            prior_callback = callback

            def record_callback(current_task: GetUpTask) -> None:
                if prior_callback is not None:
                    prior_callback(current_task)
                if current_task.env.elapsed_control_steps % 2 == 0:
                    camera.lookat[:] = current_task.env.data.xpos[
                        current_task.env._pelvis_body_id
                    ]
                    renderer.update_scene(current_task.env.data, camera=camera)
                    frames.append(renderer.render().copy())

            callback = record_callback

        try:
            result = evaluate_episode(
                task, agent.deterministic_action, seed=episode, step_callback=callback
            )
        finally:
            if viewer is not None:
                viewer.close()
            if renderer is not None:
                renderer.close()
            if frames and episode == 0:
                from PIL import Image

                args.record_gif.parent.mkdir(parents=True, exist_ok=True)
                images = [Image.fromarray(frame) for frame in frames]
                images[0].save(
                    args.record_gif,
                    save_all=True,
                    append_images=images[1:],
                    duration=40,
                    loop=0,
                )
        representative = representative or result
        results.append(result.metrics)
    keys = results[0].keys()
    summary = {
        key: float(np.nanmean([result[key] for result in results]))
        if not all(np.isnan(result[key]) for result in results)
        else None
        for key in keys
    }
    summary["success_rate"] = summary.pop("success")
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "episodes": results,
        "mean": summary,
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.reward_plot is not None and representative is not None:
        save_reward_plot(
            representative.traces, args.reward_plot, 1.0 / 50.0
        )


if __name__ == "__main__":
    main()
