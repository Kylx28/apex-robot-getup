"""Fixed fallen-state getting-up MDP."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.demonstrations import ReferenceSample, load_motion
from apex_getup.env import G1Env, G1EnvConfig, InitialState
from apex_getup.env.g1_env import StepResult
from apex_getup.env.observations import RobotState
from apex_getup.task.config import GetUpTaskConfig
from apex_getup.task.observations import (
    append_residual_reference_observation,
    build_actor_observation,
    build_paper_reference_observation,
    observation_size_for_mode,
)
from apex_getup.task.rewards import (
    RewardComponents,
    compute_task_reward,
)


@dataclass(frozen=True)
class TaskStep:
    observation: NDArray[np.float64]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class GetUpTask:
    """G1 get-up task with state-only or BONES-conditioned observations."""

    action_size = 29

    def __init__(self, config: GetUpTaskConfig | None = None, seed: int = 0):
        self.config = config or GetUpTaskConfig()
        self.config.validate()
        demonstration_path = Path(self.config.demonstration_path)
        if not demonstration_path.is_file():
            raise FileNotFoundError(f"processed reset trajectory not found: {demonstration_path}")
        loaded_trajectory = load_motion(demonstration_path)
        resolved_episode_duration = (
            float(loaded_trajectory.duration + self.config.reference_hold_duration)
            if self.config.episode_duration is None
            else float(self.config.episode_duration)
        )
        self.config = replace(
            self.config, episode_duration=resolved_episode_duration
        )
        self.env = G1Env(
            G1EnvConfig(
                episode_duration=self.config.episode_duration,
                simulation_timestep=self.config.simulation_timestep,
                control_frequency=self.config.control_frequency,
                actuator_mode=self.config.controller.actuator_mode,
                kp=self.config.controller.kp,
                kd=self.config.controller.kd,
            ),
            seed=seed,
        )
        if self.config.align_reference_root_to_fallen:
            fallen_pose = self.env.keyframe_pose("fallen")
            loaded_trajectory = loaded_trajectory.align_first_root_pose(
                fallen_pose.root_position,
                fallen_pose.root_quaternion,
            )
        # Preserve the released environment's endpoint velocity convention.
        self.reset_trajectory = loaded_trajectory
        if (
            self.reset_trajectory.root_pos is None
            or self.reset_trajectory.root_quat is None
        ):
            raise ValueError(
                "the Table-I reward requires demonstration root_pos and root_quat"
            )
        standing_pose = self.env.keyframe_pose("stand")
        self.standing_joint_positions = standing_pose.joint_positions.copy()
        self.standing_pelvis_height = float(standing_pose.root_position[2])
        self._reference_qpos, self._reference_qvel = (
            self._build_reference_generalized_states()
        )
        self.observation_size = observation_size_for_mode(
            self.config.observation.mode
        )
        self._reference_actions = np.asarray(
            [
                self.env.joint_target_to_action(target)[0]
                for target in self.reset_trajectory.q
            ]
        )
        self._rng = np.random.default_rng(seed)
        self.initial_pelvis_height = 0.0
        self.initial_uprightness = 0.0
        self.previous_action = np.zeros(self.action_size, dtype=np.float64)
        self.previous_policy_action = np.zeros(self.action_size, dtype=np.float64)
        self.stable_success_duration = 0.0
        self.consecutive_standing_steps = 0
        self.first_success_time: float | None = None
        self.episode_success = False
        self._last_state: RobotState | None = None

    @property
    def control_timestep(self) -> float:
        return self.env.config.control_timestep

    @staticmethod
    def torso_uprightness(state: RobotState) -> float:
        """Cosine between the pelvis/root local +Z axis and world +Z.

        The historical method name is retained for API compatibility.
        """
        quaternion = state.root_quaternion / np.linalg.norm(state.root_quaternion)
        _, x, y, _ = quaternion
        return float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))

    def standing_posture(self, state: RobotState) -> bool:
        success = self.config.success
        return bool(
            state.pelvis_height > success.pelvis_height
            and self.torso_uprightness(state) > success.torso_uprightness
            and np.linalg.norm(state.root_linear_velocity) < success.maximum_linear_speed
            and np.linalg.norm(state.root_angular_velocity) < success.maximum_angular_speed
        )

    def _build_reference_generalized_states(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Construct synchronized MuJoCo-order BONES qpos/qvel arrays."""
        assert self.reset_trajectory.root_pos is not None
        assert self.reset_trajectory.root_quat is not None
        qpos = np.stack(
            [
                self.env.pack_generalized_position(root_pos, root_quat, joints)
                for root_pos, root_quat, joints in zip(
                    self.reset_trajectory.root_pos,
                    self.reset_trajectory.root_quat,
                    self.reset_trajectory.q,
                )
            ]
        )
        # Official reference root velocity is identically zero.
        qvel = np.zeros((len(qpos), self.env.model.nv), dtype=np.float64)
        qvel[:, self.env._qvel_indices] = self.reset_trajectory.qd
        return qpos, qvel

    @property
    def current_reference_time(self) -> float:
        """Reference clock time corresponding to the current simulator state."""
        return self.env.simulation_time

    def reference_sample(self, reference_time: float | None = None) -> ReferenceSample:
        """Return the configured native-rate reference on the control clock."""
        return self.reset_trajectory.sample(
            self.current_reference_time if reference_time is None else reference_time,
            interpolate=self.config.interpolate_reference,
        )

    def _reference_generalized_state(
        self, sample: ReferenceSample
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert sample.root_pos is not None
        assert sample.root_quat is not None
        qpos = self.env.pack_generalized_position(
            sample.root_pos, sample.root_quat, sample.q
        )
        if sample.lower_index == sample.upper_index:
            qvel = self._reference_qvel[sample.lower_index].copy()
        else:
            qvel = (
                (1.0 - sample.interpolation)
                * self._reference_qvel[sample.lower_index]
                + sample.interpolation * self._reference_qvel[sample.upper_index]
            )
        qvel[self.env._qvel_indices] = sample.qd
        return qpos, qvel

    def _observation(self, state: RobotState) -> NDArray[np.float64]:
        reference = self.reference_sample()
        if self.config.observation.mode == "paper_reference":
            simulated_qpos, simulated_qvel = self.env.generalized_state()
            reference_qpos, reference_qvel = self._reference_generalized_state(
                reference
            )
            return build_paper_reference_observation(
                simulated_qpos,
                simulated_qvel,
                reference_qpos,
                reference_qvel,
                state.joint_positions,
                state.joint_velocities,
                reference.q,
                reference.qd,
                self.standing_joint_positions,
                reference.phase,
                self.previous_policy_action,
            )
        state_observation = build_actor_observation(
            state,
            self.previous_action,
            self.env.action_center,
            self.env.action_scale,
            self.config.observation,
        )
        if self.config.observation.mode == "state_only":
            return state_observation
        return append_residual_reference_observation(
            state_observation,
            state,
            reference.phase,
            self.env.joint_target_to_action(reference.q)[0],
            reference.q,
            reference.qd,
            self.standing_joint_positions,
            self.env.action_scale,
            self.config.observation,
        )

    @staticmethod
    def _perturb_quaternion(
        quaternion: NDArray[np.float64], rotation_vector: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply a small axis-angle rotation to a ``wxyz`` quaternion."""
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-12:
            return quaternion.copy()
        half_angle = 0.5 * angle
        delta = np.concatenate(
            ([np.cos(half_angle)], np.sin(half_angle) * rotation_vector / angle)
        )
        w1, x1, y1, z1 = delta
        w2, x2, y2, z2 = quaternion
        result = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
        return result / np.linalg.norm(result)

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Reset to demonstration frame zero, with optional configured noise."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        reference = self.reference_sample(0.0)
        # ``auto`` follows the frame-zero reference velocity, matching the
        # released environment. ``zero`` remains an explicit legacy ablation.
        use_reference_velocity = self.config.initial_velocity_mode != "zero"
        joint_velocity = (
            reference.qd
            if use_reference_velocity
            else np.zeros(self.action_size)
        )
        perturbation = self.config.reset_perturbation
        joint_position = reference.q.copy()
        if perturbation.joint_position_std > 0:
            joint_position += self._rng.normal(
                0.0, perturbation.joint_position_std, self.action_size
            )
            joint_position = np.clip(
                joint_position,
                self.env.joint_limits[:, 0],
                self.env.joint_limits[:, 1],
            )
        joint_velocity = joint_velocity.copy()
        if perturbation.joint_velocity_std > 0:
            joint_velocity += self._rng.normal(
                0.0, perturbation.joint_velocity_std, self.action_size
            )
        assert self.reset_trajectory.root_pos is not None
        assert self.reset_trajectory.root_quat is not None
        assert reference.root_quat is not None
        assert reference.root_pos is not None
        root_quaternion = reference.root_quat.copy()
        if perturbation.root_orientation_std > 0:
            root_quaternion = self._perturb_quaternion(
                root_quaternion,
                self._rng.normal(0.0, perturbation.root_orientation_std, 3),
            )
        root_linear_velocity = np.zeros(3)
        if perturbation.root_linear_velocity_std > 0:
            root_linear_velocity = self._rng.normal(
                0.0, perturbation.root_linear_velocity_std, 3
            )
        root_angular_velocity = np.zeros(3)
        if perturbation.root_angular_velocity_std > 0:
            root_angular_velocity = self._rng.normal(
                0.0, perturbation.root_angular_velocity_std, 3
            )
        state = self.env.reset(
            InitialState(
                joint_positions=joint_position,
                joint_velocities=joint_velocity,
                root_position=reference.root_pos,
                root_quaternion=root_quaternion,
                root_linear_velocity=root_linear_velocity,
                root_angular_velocity=root_angular_velocity,
            ),
            seed=seed,
        )
        self.initial_pelvis_height = state.pelvis_height
        self.initial_uprightness = self.torso_uprightness(state)
        self.previous_action = np.zeros(self.action_size, dtype=np.float64)
        self.previous_policy_action = np.zeros(self.action_size, dtype=np.float64)
        self.stable_success_duration = 0.0
        self.consecutive_standing_steps = 0
        self.first_success_time = None
        self.episode_success = False
        self._last_state = state
        return self._observation(state)

    def step(
        self,
        executed_action: ArrayLike,
        *,
        policy_action: ArrayLike | None = None,
        physical_joint_target: ArrayLike | None = None,
    ) -> TaskStep:
        """Advance the task; reward regularization uses ``policy_action``.

        Passing no policy action preserves the direct/vanilla convention where
        the policy and executed action are identical. Residual control supplies
        ``physical_joint_target`` in radians; vanilla control continues through
        the normalized ``executed_action`` path.
        """
        old_action = self.previous_action.copy()
        old_policy_action = self.previous_policy_action.copy()
        reward_action = np.asarray(
            executed_action if policy_action is None else policy_action,
            dtype=np.float64,
        )
        if reward_action.shape != (self.action_size,) or not np.all(
            np.isfinite(reward_action)
        ):
            raise ValueError("policy_action must be finite with shape (29,)")
        # The prior, incoming observation, and reward all use this exact
        # time-interpolated sample.  The next observation advances with MuJoCo
        # time after the transition.
        tracking_reference = self.reference_sample()
        tracking_reference_index = tracking_reference.lower_index
        try:
            env_result: StepResult = (
                self.env.step(executed_action)
                if physical_joint_target is None
                else self.env.step_joint_target(physical_joint_target)
            )
        except FloatingPointError as error:
            if self._last_state is None:
                raise
            zeros = np.zeros(self.action_size)
            reward_components = {
                name: 0.0 for name in RewardComponents.__dataclass_fields__
            }
            invalid_info: dict[str, object] = {
                "joint_target": self.env.last_joint_target.copy(),
                "torque": zeros.copy(),
                "mean_abs_torque": zeros.copy(),
                "max_abs_torque": zeros.copy(),
                "torque_over_80_fraction": zeros.copy(),
                "torque_saturation_fraction": 0.0,
                "action_was_clipped": False,
                **reward_components,
                "uprightness_reward": 0.0,
                "reward_components": reward_components,
                "uprightness": self.torso_uprightness(self._last_state),
                "standing_posture": False,
                "stable_standing_indicator": 0.0,
                "consecutive_standing_steps": self.consecutive_standing_steps,
                "consecutive_standing_duration": self.stable_success_duration,
                "stable_success_duration": self.stable_success_duration,
                "first_success_time": self.first_success_time,
                "episode_success": False,
                "pelvis_height": self._last_state.pelvis_height,
                "root_linear_speed": float(
                    np.linalg.norm(self._last_state.root_linear_velocity)
                ),
                "root_angular_speed": float(
                    np.linalg.norm(self._last_state.root_angular_velocity)
                ),
                "action_magnitude": 0.0,
                "action_delta_magnitude": 0.0,
                "mean_torque": 0.0,
                "tracking_reference_index": tracking_reference_index,
                "tracking_reference_time": tracking_reference.time,
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
            self.consecutive_standing_steps += 1
        else:
            self.consecutive_standing_steps = 0
        self.stable_success_duration = (
            self.consecutive_standing_steps * self.control_timestep
        )
        if (
            not self.episode_success
            and self.stable_success_duration + 1e-12
            >= self.config.success.hold_duration
        ):
            self.episode_success = True
            self.first_success_time = self.env.simulation_time
        assert self.reset_trajectory.root_pos is not None
        assert self.reset_trajectory.root_quat is not None
        assert tracking_reference.root_pos is not None
        assert tracking_reference.root_quat is not None
        phase = tracking_reference.phase
        left_foot_velocity, right_foot_velocity = self.env.foot_planar_velocities()
        left_foot_ground_contact = any(
            contact.involves_left_foot
            and "world" in (contact.body1, contact.body2)
            for contact in state.contacts
        )
        right_foot_ground_contact = any(
            contact.involves_right_foot
            and "world" in (contact.body1, contact.body2)
            for contact in state.contacts
        )
        if self.config.observation.mode == "paper_reference":
            # Match the released slip-support heuristic exactly.
            left_height, right_height = self.env.foot_origin_heights()
            left_foot_ground_contact = left_height < 0.06
            right_foot_ground_contact = right_height < 0.06
        components: RewardComponents = compute_task_reward(
            phase=phase,
            joint_positions=state.joint_positions,
            joint_velocities=state.joint_velocities,
            root_position=state.root_position,
            root_quaternion=state.root_quaternion,
            pelvis_height=state.pelvis_height,
            uprightness=self.torso_uprightness(state),
            reference_joint_positions=tracking_reference.q,
            reference_joint_velocities=tracking_reference.qd,
            reference_root_position=tracking_reference.root_pos,
            reference_root_quaternion=tracking_reference.root_quat,
            standing_joint_positions=self.standing_joint_positions,
            standing_height=self.standing_pelvis_height,
            left_foot_contact=left_foot_ground_contact,
            right_foot_contact=right_foot_ground_contact,
            left_foot_planar_velocity=left_foot_velocity,
            right_foot_planar_velocity=right_foot_velocity,
            policy_action=reward_action,
            previous_policy_action=old_policy_action,
            config=self.config.reward,
        )
        reward_components = components.as_dict()
        total_reward = components.total_reward
        self.previous_action = self.env.last_action.copy()
        self.previous_policy_action = reward_action.copy()
        self._last_state = state
        terminated = bool(
            self.episode_success and self.config.success.terminate_on_success
        )
        info: dict[str, object] = {
            **env_result.info,
            **reward_components,
            "uprightness_reward": components.uprightness,
            "reward_components": reward_components,
            "uprightness": self.torso_uprightness(state),
            "standing_posture": posture,
            "stable_standing_indicator": float(posture),
            "consecutive_standing_steps": self.consecutive_standing_steps,
            "consecutive_standing_duration": self.stable_success_duration,
            "stable_success_duration": self.stable_success_duration,
            "first_success_time": self.first_success_time,
            "episode_success": self.episode_success,
            "pelvis_height": state.pelvis_height,
            "root_linear_speed": float(
                np.linalg.norm(state.root_linear_velocity)
            ),
            "root_angular_speed": float(
                np.linalg.norm(state.root_angular_velocity)
            ),
            "action_magnitude": float(np.sqrt(np.mean(np.square(self.previous_action)))),
            "action_delta_magnitude": float(
                np.sqrt(np.mean(np.square(self.previous_action - old_action)))
            ),
            "mean_torque": float(np.mean(env_result.info["mean_abs_torque"])),
            "tracking_reference_index": tracking_reference_index,
            "tracking_reference_time": tracking_reference.time,
        }
        return TaskStep(
            observation=self._observation(state),
            reward=total_reward,
            terminated=terminated,
            truncated=env_result.truncated,
            info=info,
        )
