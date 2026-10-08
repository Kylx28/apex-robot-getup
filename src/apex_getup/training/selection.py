"""Deterministic evaluation ranking used for best-checkpoint selection."""

from __future__ import annotations

from collections.abc import Mapping


FRAME_ZERO_SELECTION_METRIC = (
    "sustained_success_then_standing_duration_then_final_state"
)


def frame_zero_selection_key(
    metrics: Mapping[str, float],
) -> tuple[float, ...]:
    """Return the lexicographic quality key for a fixed frame-zero rollout.

    Sustained task success dominates every shaping metric. Among checkpoints
    with equal success, prefer longer stable standing, better final height and
    uprightness, lower terminal root speeds, and finally higher raw return.
    """
    return (
        float(metrics["success"]),
        float(metrics["maximum_consecutive_standing_duration"]),
        float(metrics["final_pelvis_height"]),
        float(metrics["final_uprightness"]),
        -float(metrics["final_root_linear_speed"]),
        -float(metrics["final_root_angular_speed"]),
        float(metrics["episode_return"]),
    )



def is_better_frame_zero_evaluation(
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float] | None,
) -> bool:
    """Whether ``candidate`` should replace the current best checkpoint."""
    return incumbent is None or frame_zero_selection_key(
        candidate
    ) > frame_zero_selection_key(incumbent)
