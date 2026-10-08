"""Strict YAML configuration for PPO training runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from apex_getup.apex import PriorScheduleConfig, ResidualControlConfig
from apex_getup.env.config import JOINT_NAMES
from apex_getup.rl import NormalizationConfig, PPOConfig
from apex_getup.task import (
    GetUpTaskConfig,
    ObservationConfig,
    PDControllerConfig,
    ResetPerturbationConfig,
    RewardConfig,
    SuccessConfig,
)


T = TypeVar("T")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(value)


def _strict_dataclass_kwargs(
    value: object, cls: type[T], name: str
) -> dict[str, Any]:
    result = _mapping(value, name)
    allowed = {item.name for item in fields(cls)}
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return result


def _pd_controller(value: object) -> PDControllerConfig:
    """Parse scalar/vector gains or an explicit canonical joint-gain map."""
    values = _mapping(value, "task.controller")
    unknown = set(values) - {"actuator_mode", "kp", "kd", "joint_gains"}
    if unknown:
        raise ValueError(f"unknown task.controller fields: {sorted(unknown)}")
    if "joint_gains" not in values:
        for gain in ("kp", "kd"):
            if isinstance(values.get(gain), list):
                values[gain] = tuple(values[gain])
        return PDControllerConfig(**values)
    if "kp" in values or "kd" in values:
        raise ValueError("joint_gains cannot be combined with controller kp or kd")
    actuator_mode = str(values.get("actuator_mode", "torque_pd"))
    joint_gains = _mapping(values["joint_gains"], "task.controller.joint_gains")
    missing = set(JOINT_NAMES) - set(joint_gains)
    extra = set(joint_gains) - set(JOINT_NAMES)
    if missing or extra:
        raise ValueError(
            "joint_gains must name every canonical G1 joint; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    kp: list[float] = []
    kd: list[float] = []
    for joint_name in JOINT_NAMES:
        pair = _mapping(
            joint_gains[joint_name],
            f"task.controller.joint_gains.{joint_name}",
        )
        unknown_pair = set(pair) - {"kp", "kd"}
        if unknown_pair or set(pair) != {"kp", "kd"}:
            raise ValueError(f"{joint_name} must define exactly kp and kd")
        kp.append(float(pair["kp"]))
        kd.append(float(pair["kd"]))
    return PDControllerConfig(
        actuator_mode=actuator_mode, kp=tuple(kp), kd=tuple(kd)
    )


@dataclass(frozen=True)
class TrainingRunConfig:
    """Resolved settings shared by a PPO training entry point."""

    seed: int = 0
    total_steps: int = 100_000
    output_dir: Path = Path("artifacts/training/apex")
    checkpoint_frequency: int = 10_000
    evaluation_frequency: int = 10_000
    smoothing_window: int = 10
    warm_start_checkpoint: Path | None = None
    num_envs: int = 1
    control_mode: str = "decaying-prior"
    residual_scale: float = 0.25
    task: GetUpTaskConfig = field(default_factory=GetUpTaskConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    prior_schedule: PriorScheduleConfig = field(default_factory=PriorScheduleConfig)

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if (
            self.total_steps <= 0
            or self.checkpoint_frequency <= 0
            or self.evaluation_frequency <= 0
        ):
            raise ValueError(
                "training, checkpoint, and evaluation step counts must be positive"
            )
        if self.smoothing_window <= 0:
            raise ValueError("smoothing_window must be positive")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.control_mode not in {"decaying-prior", "residual"}:
            raise ValueError("control mode must be decaying-prior or residual")
        ResidualControlConfig(self.residual_scale).validate()
        self.task.validate()
        self.ppo.validate()
        self.normalization.validate()
        self.prior_schedule.validate()

    @classmethod
    def from_mapping(cls, value: object) -> "TrainingRunConfig":
        root = _mapping(value, "configuration")
        allowed_sections = {
            "run", "training", "control", "task", "ppo", "normalization",
            "prior_schedule"
        }
        unknown_sections = set(root) - allowed_sections
        if unknown_sections:
            raise ValueError(
                f"unknown configuration sections: {sorted(unknown_sections)}"
            )

        run = _mapping(root.get("run"), "run")
        allowed_run = {
            "seed", "total_steps", "output_dir", "checkpoint_frequency",
            "evaluation_frequency", "smoothing_window", "warm_start_checkpoint",
        }
        unknown_run = set(run) - allowed_run
        if unknown_run:
            raise ValueError(f"unknown run fields: {sorted(unknown_run)}")
        training = _mapping(root.get("training"), "training")
        unknown_training = set(training) - {"num_envs"}
        if unknown_training:
            raise ValueError(
                f"unknown training fields: {sorted(unknown_training)}"
            )
        control = _mapping(root.get("control"), "control")
        unknown_control = set(control) - {"mode", "residual_scale"}
        if unknown_control:
            raise ValueError(
                f"unknown control fields: {sorted(unknown_control)}"
            )

        task_values = _strict_dataclass_kwargs(
            root.get("task"), GetUpTaskConfig, "task"
        )
        if "demonstration_path" in task_values:
            task_values["demonstration_path"] = Path(
                task_values["demonstration_path"]
            )
        nested_task_types = {
            "reward": RewardConfig,
            "reset_perturbation": ResetPerturbationConfig,
            "success": SuccessConfig,
            "observation": ObservationConfig,
        }
        for key, nested_type in nested_task_types.items():
            if key in task_values:
                nested_values = _strict_dataclass_kwargs(
                    task_values[key], nested_type, f"task.{key}"
                )
                task_values[key] = nested_type(**nested_values)
        if "controller" in task_values:
            task_values["controller"] = _pd_controller(task_values["controller"])

        ppo_values = _strict_dataclass_kwargs(root.get("ppo"), PPOConfig, "ppo")
        for key in ("actor_hidden_dimensions", "critic_hidden_dimensions"):
            if key in ppo_values:
                ppo_values[key] = tuple(ppo_values[key])
        seed = int(run.get("seed", 0))
        ppo_values.setdefault("seed", seed)

        result = cls(
            seed=seed,
            total_steps=int(run.get("total_steps", 100_000)),
            output_dir=Path(run.get("output_dir", "artifacts/training/apex")),
            checkpoint_frequency=int(run.get("checkpoint_frequency", 10_000)),
            evaluation_frequency=int(run.get("evaluation_frequency", 10_000)),
            smoothing_window=int(run.get("smoothing_window", 10)),
            warm_start_checkpoint=(
                None
                if run.get("warm_start_checkpoint") is None
                else Path(run["warm_start_checkpoint"])
            ),
            num_envs=int(training.get("num_envs", 1)),
            control_mode=str(control.get("mode", "decaying-prior")),
            residual_scale=float(control.get("residual_scale", 0.25)),
            task=GetUpTaskConfig(**task_values),
            ppo=PPOConfig(**ppo_values),
            normalization=NormalizationConfig(
                **_strict_dataclass_kwargs(
                    root.get("normalization"),
                    NormalizationConfig,
                    "normalization",
                )
            ),
            prior_schedule=PriorScheduleConfig(
                **_strict_dataclass_kwargs(
                    root.get("prior_schedule"),
                    PriorScheduleConfig,
                    "prior_schedule",
                )
            ),
        )
        result.validate()
        return result


def load_training_config(path: Path | str) -> TrainingRunConfig:
    """Load and validate one YAML training configuration."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return TrainingRunConfig.from_mapping(payload)
