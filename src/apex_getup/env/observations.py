"""Structured state and contact extraction for the G1."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Contact:
    """One active MuJoCo contact, named by its two geoms and bodies."""

    geom1: str
    geom2: str
    body1: str
    body2: str
    distance: float
    involves_left_foot: bool
    involves_right_foot: bool
    involves_non_foot_body: bool

    @property
    def involves_foot(self) -> bool:
        return self.involves_left_foot or self.involves_right_foot


@dataclass(frozen=True)
class RobotState:
    """A snapshot of robot state.

    Positions and velocities use SI units. Root position, linear velocity, and
    angular velocity are in the world frame. Quaternions use MuJoCo's ``wxyz``
    order. ``projected_gravity`` is the world down unit vector expressed in the
    torso frame.
    """

    joint_positions: NDArray[np.float64]
    joint_velocities: NDArray[np.float64]
    root_position: NDArray[np.float64]
    root_quaternion: NDArray[np.float64]
    root_linear_velocity: NDArray[np.float64]
    root_angular_velocity: NDArray[np.float64]
    pelvis_height: float
    torso_rotation_matrix: NDArray[np.float64]
    projected_gravity: NDArray[np.float64]
    contacts: tuple[Contact, ...]
    left_foot_contact: bool
    right_foot_contact: bool
    non_foot_contact: bool

    def as_vector(self) -> NDArray[np.float64]:
        """Return a fixed-size 78-vector suitable for later policy code."""
        return np.concatenate(
            (
                self.joint_positions,
                self.joint_velocities,
                self.root_position,
                self.root_quaternion,
                self.root_linear_velocity,
                self.root_angular_velocity,
                np.array([self.pelvis_height]),
                self.projected_gravity,
                np.array(
                    [
                        self.left_foot_contact,
                        self.right_foot_contact,
                        self.non_foot_contact,
                    ],
                    dtype=np.float64,
                ),
            )
        )


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"unnamed_{object_id}"


def extract_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_foot_body_id: int,
    right_foot_body_id: int,
) -> tuple[Contact, ...]:
    """Extract active contacts, including self-collisions."""
    result: list[Contact] = []
    for index in range(data.ncon):
        raw = data.contact[index]
        geom1, geom2 = int(raw.geom1), int(raw.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        body_ids = (body1, body2)
        result.append(
            Contact(
                geom1=_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                geom2=_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                body1=_name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                body2=_name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
                distance=float(raw.dist),
                involves_left_foot=left_foot_body_id in body_ids,
                involves_right_foot=right_foot_body_id in body_ids,
                # Body 0 is the MuJoCo world/floor. Every other body in this
                # single-robot scene is part of G1.
                involves_non_foot_body=any(
                    body_id not in (0, left_foot_body_id, right_foot_body_id)
                    for body_id in body_ids
                ),
            )
        )
    return tuple(result)


def contact_flags(contacts: tuple[Contact, ...]) -> tuple[bool, bool, bool]:
    """Return left-foot, right-foot, and any non-foot contact flags."""
    left = any(contact.involves_left_foot for contact in contacts)
    right = any(contact.involves_right_foot for contact in contacts)
    non_foot = any(contact.involves_non_foot_body for contact in contacts)
    return left, right, non_foot


CONTACT_REGIONS: tuple[str, ...] = (
    "torso", "pelvis", "left_hand", "right_hand", "left_forearm",
    "right_forearm", "left_knee", "right_knee", "left_shin", "right_shin",
    "left_foot", "right_foot", "other",
)


def _body_region(body_name: str) -> str | None:
    if body_name.startswith("unnamed_0") or body_name == "world":
        return None
    if body_name == "pelvis":
        return "pelvis"
    if "torso" in body_name or "waist" in body_name or "head" in body_name:
        return "torso"
    side = "left" if body_name.startswith("left_") else "right" if body_name.startswith("right_") else None
    if side is None:
        return "other"
    if "ankle_roll" in body_name:
        return f"{side}_foot"
    if "ankle_pitch" in body_name:
        return f"{side}_shin"
    if "knee" in body_name:
        return f"{side}_knee"
    if "wrist_yaw" in body_name:
        return f"{side}_hand"
    if "elbow" in body_name or "wrist" in body_name:
        return f"{side}_forearm"
    return "other"


def contact_regions(contacts: tuple[Contact, ...]) -> set[str]:
    """Return coarse robot body regions participating in active contacts."""
    regions: set[str] = set()
    for contact in contacts:
        for body_name in (contact.body1, contact.body2):
            region = _body_region(body_name)
            if region is not None:
                regions.add(region)
    return regions
