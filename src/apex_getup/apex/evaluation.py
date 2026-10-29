"""Deterministic APEX evaluation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from apex_getup.apex.agent import ApexAgent
from apex_getup.apex.prior import FixedCoefficientPriorAdapter
from apex_getup.apex.teacher import ActionPrior
from apex_getup.rl.normalization import VecNormalizer
from apex_getup.task import EvaluationResult, GetUpTask, evaluate_episode
from apex_getup.video import MujocoVideoRecorder


def evaluate_apex_episode(
    task: GetUpTask,
    agent: ApexAgent,
    actor_normalizer: VecNormalizer,
    prior: ActionPrior,
    coefficient: float,
    *,
    seed: int = 0,
    teacher: bool = False,
    step_callback: Callable[[GetUpTask], None] | None = None,
) -> EvaluationResult:
    """Evaluate teacher, guided, or prior-free APEX control."""
    if actor_normalizer.training:
        raise ValueError("evaluation normalizer must be frozen")

    def policy(_: np.ndarray) -> np.ndarray:
        if teacher:
            return np.zeros(task.action_size)
        actor_observation = task.apex_observations().actor
        return agent.deterministic_action(
            actor_observation,
            actor_normalizer=actor_normalizer,
        )

    return evaluate_episode(
        task,
        policy,
        seed=seed,
        step_callback=step_callback,
        action_adapter=FixedCoefficientPriorAdapter(prior, coefficient),
    )


def record_apex_evaluation_video(
    task: GetUpTask,
    agent: ApexAgent,
    actor_normalizer: VecNormalizer,
    prior: ActionPrior,
    coefficient: float,
    path: str | Path,
    *,
    seed: int = 0,
    teacher: bool = False,
) -> EvaluationResult:
    """Record one deterministic APEX evaluation episode."""
    with MujocoVideoRecorder(task, path) as recorder:
        return evaluate_apex_episode(
            task,
            agent,
            actor_normalizer,
            prior,
            coefficient,
            seed=seed,
            teacher=teacher,
            step_callback=lambda _: recorder.capture(),
        )
