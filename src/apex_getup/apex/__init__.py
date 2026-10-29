"""APEX components introduced incrementally after the PPO baseline."""

# Initialize these names before importing ApexAgent: the native RL trainer
# requests them while the agent imports the shared PPO implementation.
from apex_getup.apex.prior import (
    FixedCoefficientPriorAdapter,
    PriorCoefficientSchedule,
    PriorScheduleConfig,
    compose_apex_action,
)
from apex_getup.apex.advantages import combine_group_advantages
from apex_getup.apex.agent import ApexAgent
from apex_getup.apex.config import ApexConfig
from apex_getup.apex.critics import ApexCritic
from apex_getup.apex.evaluation import (
    evaluate_apex_episode,
    record_apex_evaluation_video,
)
from apex_getup.apex.observations import ApexObservations, build_apex_observations
from apex_getup.apex.rewards import (
    APEX_REWARD_GROUPS,
    ApexRewardGroups,
    split_reward_components,
)
from apex_getup.apex.rollout import ApexRolloutBuffer, ApexRolloutSpec
from apex_getup.apex.teacher import (
    ActionPrior,
    RecordedActionPrior,
    export_teacher_action_prior,
    load_apex_expert,
    validate_recorded_prior,
)

__all__ = [
    "APEX_REWARD_GROUPS",
    "ActionPrior",
    "ApexAgent",
    "ApexConfig",
    "ApexCritic",
    "ApexObservations",
    "ApexRewardGroups",
    "ApexRolloutBuffer",
    "ApexRolloutSpec",
    "FixedCoefficientPriorAdapter",
    "PriorCoefficientSchedule",
    "PriorScheduleConfig",
    "RecordedActionPrior",
    "build_apex_observations",
    "combine_group_advantages",
    "compose_apex_action",
    "export_teacher_action_prior",
    "load_apex_expert",
    "evaluate_apex_episode",
    "record_apex_evaluation_video",
    "split_reward_components",
    "validate_recorded_prior",
]
