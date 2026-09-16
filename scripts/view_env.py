#!/usr/bin/env python3
"""View or headlessly smoke-test a fixed-target G1 rollout."""

from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np

from apex_getup.env import G1Env, G1EnvConfig, InitialState


FALLEN_POSES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "supine": (
        np.array([0.0, 0.0, 0.32]),
        np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]),
    ),
    "prone": (
        np.array([0.0, 0.0, 0.32]),
        np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]),
    ),
    "left_side": (
        np.array([0.0, 0.0, 0.28]),
        np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]),
    ),
    "right_side": (
        np.array([0.0, 0.0, 0.28]),
        np.array([math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", choices=FALLEN_POSES, default="supine")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without opening an interactive MuJoCo window",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    config = G1EnvConfig(episode_duration=args.duration)
    env = G1Env(config, seed=args.seed)
    root_position, root_quaternion = FALLEN_POSES[args.pose]
    state = env.reset(
        InitialState(root_position=root_position, root_quaternion=root_quaternion),
        seed=args.seed,
    )
    action, violations = env.joint_target_to_action(env.nominal_pose)
    assert not np.any(violations)
    next_log = 0.0

    logging.info(
        "pose=%s control=%.1f Hz physics=%.1f Hz substeps=%d",
        args.pose,
        config.control_frequency,
        1.0 / config.simulation_timestep,
        config.physics_steps_per_control_step,
    )

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    wall_start = time.monotonic()
    try:
        while env.simulation_time < args.duration and (viewer is None or viewer.is_running()):
            result = env.step(action)
            state = result.state
            if env.simulation_time + 1e-12 >= next_log:
                logging.info(
                    "t=%5.2f pelvis_z=% .3f gravity_body=%s contacts=%d feet=(%s,%s) nonfoot=%s",
                    env.simulation_time,
                    state.pelvis_height,
                    np.array2string(state.projected_gravity, precision=2),
                    len(state.contacts),
                    state.left_foot_contact,
                    state.right_foot_contact,
                    state.non_foot_contact,
                )
                next_log += 1.0
            if viewer is not None:
                viewer.sync()
                target_wall_time = wall_start + env.simulation_time
                time.sleep(max(0.0, target_wall_time - time.monotonic()))
    finally:
        if viewer is not None:
            viewer.close()

    if not np.all(np.isfinite(state.as_vector())):
        raise RuntimeError("rollout finished with a non-finite state")
    logging.info("rollout complete: %d control steps", env.elapsed_control_steps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(parse_args())
