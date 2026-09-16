"""Minimal dependency-light MuJoCo environment for the 29-DoF Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex_getup.env.config import G1EnvConfig, JOINT_NAMES, NUM_ACTUATORS
from apex_getup.env.control import (
    action_parameters_from_joint_limits,
    action_to_joint_target,
    joint_target_to_action,
    pd_torque,
)
from apex_getup.env.observations import Contact, RobotState, contact_flags, extract_contacts


@dataclass(frozen=True)
class InitialState:
    """Optional reset state. Omitted fields use the configured nominal reset."""

    joint_positions: ArrayLike | None = None
    joint_velocities: ArrayLike | None = None
    root_position: ArrayLike | None = None
    root_quaternion: ArrayLike | None = None
    root_linear_velocity: ArrayLike | None = None
    root_angular_velocity: ArrayLike | None = None


@dataclass(frozen=True)
class StepResult:
    state: RobotState
    terminated: bool
    truncated: bool
    info: dict[str, object]


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "unitree_g1" / "scene.xml"


def _array(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return result


class G1Env:
    """G1 simulation with normalized joint targets and an explicit PD loop.

    This is intentionally not tied to Gymnasium. A later RL adapter can wrap
    the small ``reset``/``step`` API without coupling simulation to an RL stack.
    """

    action_size = NUM_ACTUATORS
    observation_size = 78

    def __init__(self, config: G1EnvConfig | None = None, seed: int | None = None):
        self.config = config or G1EnvConfig()
        self.config.validate()
        model_path = Path(self.config.model_path or default_model_path()).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"G1 scene not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
        self.model.opt.timestep = self.config.simulation_timestep
        self.data = mujoco.MjData(self.model)
        if self.model.nu != NUM_ACTUATORS:
            raise ValueError(f"expected {NUM_ACTUATORS} actuators, model has {self.model.nu}")

        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
            dtype=np.int32,
        )
        if np.any(self._joint_ids < 0):
            raise ValueError("the model is missing one or more expected G1 joints")
        self._qpos_indices = self.model.jnt_qposadr[self._joint_ids].copy()
        self._qvel_indices = self.model.jnt_dofadr[self._joint_ids].copy()
        self.joint_limits = self.model.jnt_range[self._joint_ids].copy()
        if not np.all(self.model.jnt_limited[self._joint_ids]):
            raise ValueError("all actuated G1 joints must have limits")

        transmitted_joints = self.model.actuator_trnid[:, 0]
        if not np.array_equal(transmitted_joints, self._joint_ids):
            raise ValueError("actuator order must match JOINT_NAMES")

        root_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        self._root_qpos_index = int(self.model.jnt_qposadr[root_joint_id])
        self._root_qvel_index = int(self.model.jnt_dofadr[root_joint_id])
        self._pelvis_body_id = self._body_id("pelvis")
        self._torso_body_id = self._body_id("torso_link")
        self._left_foot_body_id = self._body_id("left_ankle_roll_link")
        self._right_foot_body_id = self._body_id("right_ankle_roll_link")

        self.kp, self.kd, self.nominal_pose, configured_limits = self.config.arrays()
        self.action_center, self.action_scale = action_parameters_from_joint_limits(
            self.joint_limits
        )
        model_limits = np.max(np.abs(self.model.jnt_actfrcrange[self._joint_ids]), axis=1)
        self.torque_limits = np.minimum(configured_limits, model_limits)
        if np.any(self.nominal_pose < self.joint_limits[:, 0]) or np.any(
            self.nominal_pose > self.joint_limits[:, 1]
        ):
            raise ValueError("nominal_joint_pose violates model joint limits")

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._control_steps = 0
        self.last_action = np.zeros(NUM_ACTUATORS)
        self.last_joint_target = self.nominal_pose.copy()
        self.last_torque = np.zeros(NUM_ACTUATORS)

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def simulation_time(self) -> float:
        return float(self.data.time)

    @property
    def elapsed_control_steps(self) -> int:
        return self._control_steps

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"model has no body named {name!r}")
        return body_id

    def reset(
        self,
        initial_state: InitialState | None = None,
        *,
        seed: int | None = None,
    ) -> RobotState:
        """Reset deterministically and optionally replace any generalized state fields."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        state = initial_state or InitialState()

        joint_positions = (
            self.nominal_pose
            if state.joint_positions is None
            else _array(state.joint_positions, (NUM_ACTUATORS,), "joint_positions")
        )
        if np.any(joint_positions < self.joint_limits[:, 0]) or np.any(
            joint_positions > self.joint_limits[:, 1]
        ):
            raise ValueError("initial joint positions violate model limits")
        joint_velocities = (
            np.zeros(NUM_ACTUATORS)
            if state.joint_velocities is None
            else _array(state.joint_velocities, (NUM_ACTUATORS,), "joint_velocities")
        )
        root_position = (
            np.array([0.0, 0.0, 0.793])
            if state.root_position is None
            else _array(state.root_position, (3,), "root_position")
        )
        root_quaternion = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if state.root_quaternion is None
            else _array(state.root_quaternion, (4,), "root_quaternion")
        )
        norm = np.linalg.norm(root_quaternion)
        if norm < 1e-12:
            raise ValueError("root_quaternion must have nonzero norm")
        root_quaternion = root_quaternion / norm
        root_linear_velocity = (
            np.zeros(3)
            if state.root_linear_velocity is None
            else _array(state.root_linear_velocity, (3,), "root_linear_velocity")
        )
        root_angular_velocity = (
            np.zeros(3)
            if state.root_angular_velocity is None
            else _array(state.root_angular_velocity, (3,), "root_angular_velocity")
        )

        self.data.qpos[self._qpos_indices] = joint_positions
        self.data.qvel[self._qvel_indices] = joint_velocities
        root_qpos = self._root_qpos_index
        root_qvel = self._root_qvel_index
        self.data.qpos[root_qpos : root_qpos + 3] = root_position
        self.data.qpos[root_qpos + 3 : root_qpos + 7] = root_quaternion
        # Free-joint translation is world-frame, while its angular qvel is in
        # the child body frame. Convert the public world-frame convention.
        self.data.qvel[root_qvel : root_qvel + 3] = root_linear_velocity
        root_rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(root_rotation, root_quaternion)
        self.data.qvel[root_qvel + 3 : root_qvel + 6] = (
            root_rotation.reshape(3, 3).T @ root_angular_velocity
        )
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._control_steps = 0
        self.last_action = np.zeros(NUM_ACTUATORS)
        self.last_joint_target = joint_positions.copy()
        self.last_torque = np.zeros(NUM_ACTUATORS)
        self._assert_finite()
        return self.get_state()

    def step(self, action: ArrayLike) -> StepResult:
        """Advance one control interval while recomputing PD torque each physics step."""
        clipped_action, target = self.action_to_joint_target(action)
        torque_saturation_count = 0
        torque_abs_sum = np.zeros(NUM_ACTUATORS)
        torque_abs_max = np.zeros(NUM_ACTUATORS)
        torque_over_80_count = np.zeros(NUM_ACTUATORS, dtype=np.int64)
        for _ in range(self.config.physics_steps_per_control_step):
            torque = pd_torque(
                target,
                self.data.qpos[self._qpos_indices],
                self.data.qvel[self._qvel_indices],
                self.kp,
                self.kd,
                self.torque_limits,
            )
            torque_saturation_count += int(
                np.count_nonzero(np.isclose(np.abs(torque), self.torque_limits, atol=1e-8))
            )
            absolute_torque = np.abs(torque)
            torque_abs_sum += absolute_torque
            torque_abs_max = np.maximum(torque_abs_max, absolute_torque)
            torque_over_80_count += absolute_torque >= 0.8 * self.torque_limits
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
            self._assert_finite()

        self._control_steps += 1
        self.last_action = clipped_action
        self.last_joint_target = target
        self.last_torque = torque
        truncated = self.data.time + 1e-12 >= self.config.episode_duration
        state = self.get_state()
        return StepResult(
            state=state,
            terminated=False,
            truncated=bool(truncated),
            info={
                "joint_target": target.copy(),
                "torque": torque.copy(),
                "mean_abs_torque": torque_abs_sum
                / self.config.physics_steps_per_control_step,
                "max_abs_torque": torque_abs_max,
                "torque_over_80_fraction": torque_over_80_count
                / self.config.physics_steps_per_control_step,
                "torque_saturation_fraction": torque_saturation_count
                / (NUM_ACTUATORS * self.config.physics_steps_per_control_step),
                "action_was_clipped": bool(np.any(clipped_action != np.asarray(action))),
            },
        )

    def action_to_joint_target(
        self, action: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map a normalized 29-vector across the model's physical joint ranges."""
        return action_to_joint_target(action, self.joint_limits)

    def joint_target_to_action(
        self, joint_target: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Inverse-map physical targets and flag actual joint-limit violations."""
        return joint_target_to_action(joint_target, self.joint_limits)

    def set_kinematic_state(
        self,
        joint_positions: ArrayLike,
        *,
        joint_velocities: ArrayLike | None = None,
        root_position: ArrayLike | None = None,
        root_quaternion: ArrayLike | None = None,
        time: float | None = None,
    ) -> RobotState:
        """Directly assign a valid state without integrating physics.

        This exists for dataset inspection/replay, not policy execution.
        Omitted root fields retain their current value.
        """
        q = _array(joint_positions, (NUM_ACTUATORS,), "joint_positions")
        if np.any(q < self.joint_limits[:, 0]) or np.any(q > self.joint_limits[:, 1]):
            raise ValueError("kinematic joint positions violate model limits")
        qd = (
            np.zeros(NUM_ACTUATORS)
            if joint_velocities is None
            else _array(joint_velocities, (NUM_ACTUATORS,), "joint_velocities")
        )
        self.data.qpos[self._qpos_indices] = q
        self.data.qvel[self._qvel_indices] = qd
        root = self._root_qpos_index
        root_velocity = self._root_qvel_index
        self.data.qvel[root_velocity : root_velocity + 6] = 0.0
        if root_position is not None:
            self.data.qpos[root : root + 3] = _array(root_position, (3,), "root_position")
        if root_quaternion is not None:
            quaternion = _array(root_quaternion, (4,), "root_quaternion")
            norm = np.linalg.norm(quaternion)
            if norm < 1e-12:
                raise ValueError("root_quaternion must have nonzero norm")
            self.data.qpos[root + 3 : root + 7] = quaternion / norm
        if time is not None:
            if not np.isfinite(time) or time < 0:
                raise ValueError("time must be non-negative and finite")
            self.data.time = float(time)
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._assert_finite()
        return self.get_state()

    def get_state(self) -> RobotState:
        contacts = extract_contacts(
            self.model,
            self.data,
            self._left_foot_body_id,
            self._right_foot_body_id,
        )
        left_foot, right_foot, non_foot = contact_flags(contacts)
        torso_rotation = self.data.xmat[self._torso_body_id].reshape(3, 3).copy()
        projected_gravity = torso_rotation.T @ np.array([0.0, 0.0, -1.0])
        root_velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._pelvis_body_id,
            root_velocity,
            0,
        )
        root = self._root_qpos_index
        return RobotState(
            joint_positions=self.data.qpos[self._qpos_indices].copy(),
            joint_velocities=self.data.qvel[self._qvel_indices].copy(),
            root_position=self.data.qpos[root : root + 3].copy(),
            root_quaternion=self.data.qpos[root + 3 : root + 7].copy(),
            root_linear_velocity=root_velocity[3:].copy(),
            root_angular_velocity=root_velocity[:3].copy(),
            pelvis_height=float(self.data.xpos[self._pelvis_body_id, 2]),
            torso_rotation_matrix=torso_rotation,
            projected_gravity=projected_gravity,
            contacts=contacts,
            left_foot_contact=left_foot,
            right_foot_contact=right_foot,
            non_foot_contact=non_foot,
        )

    def foot_contacts(self) -> tuple[bool, bool]:
        state = self.get_state()
        return state.left_foot_contact, state.right_foot_contact

    def non_foot_body_contacts(self) -> tuple[Contact, ...]:
        return tuple(
            contact
            for contact in self.get_state().contacts
            if contact.involves_non_foot_body
        )

    def _assert_finite(self) -> None:
        if not (
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.data.ctrl))
        ):
            raise FloatingPointError("MuJoCo state or control contains NaN/Inf")
