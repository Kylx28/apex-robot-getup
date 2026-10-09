"""APEX components introduced incrementally after the PPO baseline."""

from apex_getup.apex.prior import (
    FixedCoefficientPriorAdapter,
    PriorCoefficientSchedule,
    PriorScheduleConfig,
    compose_apex_action,
)

__all__ = [
    "FixedCoefficientPriorAdapter",
    "PriorCoefficientSchedule",
    "PriorScheduleConfig",
    "compose_apex_action",
]
