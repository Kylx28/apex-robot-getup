"""APEX components introduced incrementally after the PPO baseline."""

from apex_getup.apex.prior import (
    DemonstrationActionPrior,
    FixedCoefficientPriorAdapter,
    ResidualActionAdapter,
    PriorCoefficientSchedule,
    PriorScheduleConfig,
    ResidualControlConfig,
    compose_apex_action,
    compose_residual_joint_target,
)

__all__ = [
    "DemonstrationActionPrior",
    "FixedCoefficientPriorAdapter",
    "ResidualActionAdapter",
    "PriorCoefficientSchedule",
    "PriorScheduleConfig",
    "ResidualControlConfig",
    "compose_apex_action",
    "compose_residual_joint_target",
]
