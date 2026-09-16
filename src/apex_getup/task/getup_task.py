"""Fixed fallen-state getting-up MDP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.demonstrations import ReferenceTrajectory
from apex_getup.env import G1Env, G1EnvConfig, InitialState
from apex_getup.env.g1_env import StepResult
from apex_getup.env.observations import RobotState
from apex_getup.task.config import GetUpTaskConfig
from apex_getup.task.observations import OBSERVATION_SIZE, build_actor_observation
from apex_getup.task.rewards import RewardComponents, compute_task_reward


@dataclass(frozen=True)
class TaskStep:
    observation: NDArray[np.float64]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class GetUpTask:
    """Task wrapper around :class:`G1Env`; BONES is used only at reset."""

    observation_size = OBSERVATION_SIZE
    action_size = 29

    def __init__(self, config: GetUpTaskConfig | None = None, seed: int = 0):
        self.config = config or GetUpTaskConfig()
        self.config.validate()
        demonstration_path = Path(self.config.demonstration_path)
        if not demonstration_path.is_file():
            raise FileNotFoundError(f"processed reset trajectory not found: {demonstration_path}")
        self.reset_trajectory = ReferenceTrajectory.load(demonstration_path)
        self.env = G1Env(
            G1EnvConfig(episode_duration=self.config.episode_duration), seed=seed
        )
        self.previous_action = np.zeros(self.action_size, dtype=np.float64)
        self.previous_pelvis_height = 0.0
        self.stable_success_duration = 0.0
        self.first_success_time: float | None = None
        self.episode_success = False
        self._last_state: RobotState | None = None

    @property
    def control_timestep(self) -> float:
        return self.env.config.control_timestep

    @staticmethod
    def torso_uprightness(state: RobotState) -> float:
        """Cosine between the torso local +Z axis and world +Z."""
        return float(np.clip(state.torso_rotation_matrix[2, 2], -1.0, 1.0))

    def standing_posture(self, state: RobotState) -> bool:
        success = self.config.success
        return bool(
            state.pelvis_height > success.pelvis_height
            and self.torso_uprightness(state) > success.torso_uprightness
            and np.linalg.norm(state.root_linear_velocity) < success.maximum_linear_speed
            and np.linalg.norm(state.root_angular_velocity) < success.maximum_angular_speed
        )

    def _observation(self, state: RobotState) -> NDArray[np.float64]:
        return build_actor_observation(
            state,
            self.previous_action,
            self.env.action_center,
            self.env.action_scale,
            self.config.observation,
        )

    def reset(self, *, seed: int | None = None) -> NDArray[np.float64]:
        velocity = (
            np.zeros(self.action_size)
            if self.config.initial_velocity_mode == "zero"
            else self.reset_trajectory.qd[0]
        )
        state = self.env.reset(
            InitialState(
                joint_positions=self.reset_trajectory.q[0],
                joint_velocities=velocity,
                root_position=None
                if self.reset_trajectory.root_pos is None
                else self.reset_trajectory.root_pos[0],
                root_quaternion=None
                if self.reset_trajectory.root_quat is None
                else self.reset_trajectory.root_quat[0],
            ),
            seed=seed,
        )
        self.previous_action = np.zeros(self.action_size, dtype=np.float64)
        self.previous_pelvis_height = state.pelvis_height
        self.stable_success_duration = 0.0
        self.first_success_time = None
        self.episode_success = False
        self._last_state = state
        return self._observation(state)

    def step(self, executed_action: ArrayLike) -> TaskStep:
        old_action = self.previous_action.copy()
        try:
            env_result: StepResult = self.env.step(executed_action)
        except FloatingPointError as error:
            if self._last_state is None:
                raise
            zeros = np.zeros(self.action_size)
            invalid_info: dict[str, object] = {
                "joint_target": self.env.last_joint_target.copy(),
                "torque": zeros.copy(),
                "mean_abs_torque": zeros.copy(),
                "max_abs_torque": zeros.copy(),
                "torque_over_80_fraction": zeros.copy(),
                "torque_saturation_fraction": 0.0,
                "action_was_clipped": False,
                "height_reward": 0.0,
                "uprightness_reward": 0.0,
                "height_progress_reward": 0.0,
                "standing_bonus": 0.0,
                "action_penalty": 0.0,
                "action_smoothness_penalty": 0.0,
                "torque_penalty": 0.0,
                "total_reward": 0.0,
                "uprightness": self.torso_uprightness(self._last_state),
                "standing_posture": False,
                "stable_success_duration": self.stable_success_duration,
                "first_success_time": self.first_success_time,
                "episode_success": False,
                "pelvis_height": self._last_state.pelvis_height,
                "action_magnitude": 0.0,
                "action_delta_magnitude": 0.0,
                "mean_torque": 0.0,
                "invalid_state": True,
                "invalid_state_error": str(error),
            }
            return TaskStep(
                observation=self._observation(self._last_state),
                reward=0.0,
                terminated=True,
                truncated=False,
                info=invalid_info,
            )
        state = env_result.state
        posture = self.standing_posture(state)
        if posture:
            self.stable_success_duration += self.control_timestep
        else:
            self.stable_success_duration = 0.0
        if (
            not self.episode_success
            and self.stable_success_duration + 1e-12
            >= self.config.success.hold_duration
        ):
            self.episode_success = True
            self.first_success_time = self.env.simulation_time
        components: RewardComponents = compute_task_reward(
            pelvis_height=state.pelvis_height,
            previous_pelvis_height=self.previous_pelvis_height,
            uprightness=self.torso_uprightness(state),
            standing_posture=posture,
            action=self.env.last_action,
            previous_action=old_action,
            torque=env_result.info["mean_abs_torque"],
            torque_limits=self.env.torque_limits,
            config=self.config.reward,
        )
        self.previous_action = self.env.last_action.copy()
        self.previous_pelvis_height = state.pelvis_height
        self._last_state = state
        terminated = bool(
            self.episode_success and self.config.success.terminate_on_success
        )
        info: dict[str, object] = {
            **env_result.info,
            **components.as_dict(),
            "uprightness": self.torso_uprightness(state),
            "standing_posture": posture,
            "stable_success_duration": self.stable_success_duration,
            "first_success_time": self.first_success_time,
            "episode_success": self.episode_success,
            "pelvis_height": state.pelvis_height,
            "action_magnitude": float(np.sqrt(np.mean(np.square(self.previous_action)))),
            "action_delta_magnitude": float(
                np.sqrt(np.mean(np.square(self.previous_action - old_action)))
            ),
            "mean_torque": float(np.mean(env_result.info["mean_abs_torque"])),
        }
        return TaskStep(
            observation=self._observation(state),
            reward=components.total_reward,
            terminated=terminated,
            truncated=env_result.truncated,
            info=info,
        )
