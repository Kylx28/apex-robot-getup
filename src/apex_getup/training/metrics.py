"""Generic append-only scalar logging and matplotlib training plots."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


AXES = ("environment_step", "ppo_update", "episode")
CSV_FIELDS = (
    "axis", "index", "environment_step", "ppo_update", "episode", "metric", "value"
)


@dataclass(frozen=True)
class MetricRecord:
    axis: str
    index: int
    metric: str
    value: float
    environment_step: int | None = None
    ppo_update: int | None = None
    episode: int | None = None


def metrics_path(path: Path | str) -> Path:
    """Resolve either a run directory, metrics directory, or scalar CSV."""
    candidate = Path(path)
    if candidate.suffix.lower() == ".csv":
        return candidate
    if candidate.name == "metrics":
        return candidate / "scalars.csv"
    return candidate / "metrics" / "scalars.csv"


class ScalarLogger:
    """Append-only scalar logger with arbitrary metric names.

    Calls can be indexed by environment step, PPO update, or episode while
    retaining the other available counters as context. Non-finite and missing
    values are skipped, which makes optional metrics safe.
    """

    def __init__(self, run_directory: Path | str, *, flush_every: int = 1) -> None:
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.run_directory = Path(run_directory)
        self.path = metrics_path(self.run_directory)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._pending: list[MetricRecord] = []
        self._log_calls = 0

    def log(
        self,
        metrics: Mapping[str, object],
        *,
        axis: str,
        index: int,
        environment_step: int | None = None,
        ppo_update: int | None = None,
        episode: int | None = None,
    ) -> None:
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}")
        if index < 0:
            raise ValueError("metric index must be non-negative")
        context = {
            "environment_step": environment_step,
            "ppo_update": ppo_update,
            "episode": episode,
        }
        context[axis] = index
        for name, raw_value in metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if raw_value is None:
                continue
            array = np.asarray(raw_value)
            if array.ndim != 0:
                continue
            try:
                value = float(array)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            self._pending.append(
                MetricRecord(
                    axis=axis,
                    index=int(index),
                    metric=name,
                    value=value,
                    environment_step=context["environment_step"],
                    ppo_update=context["ppo_update"],
                    episode=context["episode"],
                )
            )
        self._log_calls += 1
        if self._log_calls % self.flush_every == 0:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for record in self._pending:
                writer.writerow(
                    {
                        "axis": record.axis,
                        "index": record.index,
                        "environment_step": "" if record.environment_step is None else record.environment_step,
                        "ppo_update": "" if record.ppo_update is None else record.ppo_update,
                        "episode": "" if record.episode is None else record.episode,
                        "metric": record.metric,
                        "value": f"{record.value:.17g}",
                    }
                )
        self._pending.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ScalarLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _optional_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def load_metrics(path: Path | str) -> list[MetricRecord]:
    """Load valid records from a run, tolerating absent or partial final rows."""
    source = metrics_path(path)
    if not source.is_file():
        return []
    records: list[MetricRecord] = []
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(CSV_FIELDS).issubset(reader.fieldnames):
            return []
        for row in reader:
            try:
                record = MetricRecord(
                    axis=str(row["axis"]),
                    index=int(row["index"]),
                    metric=str(row["metric"]),
                    value=float(row["value"]),
                    environment_step=_optional_int(row["environment_step"]),
                    ppo_update=_optional_int(row["ppo_update"]),
                    episode=_optional_int(row["episode"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if record.axis in AXES and record.metric and math.isfinite(record.value):
                records.append(record)
    return records


def metric_series(
    records: Iterable[MetricRecord], metric: str, *, axis: str = "environment_step"
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    selected = sorted(
        (record for record in records if record.axis == axis and record.metric == metric),
        key=lambda record: record.index,
    )
    return (
        np.asarray([record.index for record in selected], dtype=np.int64),
        np.asarray([record.value for record in selected], dtype=np.float64),
    )


def moving_average(values: Sequence[float], window: int) -> NDArray[np.float64]:
    """Trailing moving average with a shortened window at the beginning."""
    if window <= 0:
        raise ValueError("moving-average window must be positive")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("moving_average expects a one-dimensional sequence")
    if len(array) == 0 or window == 1:
        return array.copy()
    cumulative = np.concatenate(([0.0], np.cumsum(array)))
    result = np.empty_like(array)
    for index in range(len(array)):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


def mean_scalar_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    rename: Mapping[str, str] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, float]:
    """Mean every finite scalar found in heterogeneous metric mappings."""
    excluded = set(exclude)
    renames = dict(rename or {})
    values: dict[str, list[float]] = {}
    for row in rows:
        for name, raw_value in row.items():
            if name in excluded or raw_value is None:
                continue
            array = np.asarray(raw_value)
            if array.ndim != 0:
                continue
            try:
                value = float(array)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.setdefault(renames.get(name, name), []).append(value)
    return {name: float(np.mean(items)) for name, items in values.items()}


class TrainingMetrics:
    """Shared bridge between training loops and :class:`ScalarLogger`."""

    def __init__(self, run_directory: Path | str, *, flush_every: int = 1) -> None:
        self.logger = ScalarLogger(run_directory, flush_every=flush_every)
        self.run_directory = Path(run_directory)
        self.ppo_updates = 0
        self.episodes = 0
        self._last_episode_summary: dict[str, float] = {}

    def record_update(
        self,
        environment_step: int,
        episodes: Sequence[Mapping[str, object]],
        update_metrics: Mapping[str, object],
    ) -> dict[str, float]:
        self.ppo_updates += 1
        for episode_metrics in episodes:
            self.episodes += 1
            episode_step = int(
                float(episode_metrics.get("environment_step", environment_step))
            )
            self.logger.log(
                episode_metrics,
                axis="episode",
                index=self.episodes,
                environment_step=episode_step,
                ppo_update=self.ppo_updates,
            )
        if episodes:
            self._last_episode_summary = mean_scalar_metrics(
                episodes,
                rename={"success": "success_rate"},
                exclude={"environment_step"},
            )
        self.logger.log(
            update_metrics,
            axis="ppo_update",
            index=self.ppo_updates,
            environment_step=environment_step,
            episode=self.episodes,
        )
        combined = {**self._last_episode_summary, **mean_scalar_metrics([update_metrics])}
        self.logger.log(
            combined,
            axis="environment_step",
            index=environment_step,
            ppo_update=self.ppo_updates,
            episode=self.episodes,
        )
        self.logger.flush()
        return combined

    def record_evaluation(
        self,
        environment_step: int,
        metrics: Mapping[str, object],
        *,
        prefix: str = "frame_zero",
    ) -> None:
        """Persist an independent evaluation without mixing it into training data."""
        prefixed = {f"{prefix}/{name}": value for name, value in metrics.items()}
        self.logger.log(
            prefixed,
            axis="environment_step",
            index=environment_step,
            ppo_update=self.ppo_updates,
            episode=self.episodes,
        )
        self.logger.flush()

    def finish(self, *, smoothing_window: int = 10) -> list[Path]:
        self.logger.close()
        return generate_training_plots(
            self.run_directory, smoothing_window=smoothing_window
        )


DEFAULT_GROUPS: dict[str, tuple[str, ...]] = {
    "return_success": (
        "episode_return", "success_rate", "time_to_stand",
        "stable_standing_duration", "stable_standing_indicator",
        "consecutive_standing_duration",
    ),
    "stable_standing": (
        "stable_standing_indicator", "consecutive_standing_duration",
        "reward/standing_pose", "reward/uprightness",
        "reward/standing_height", "reward/w_stand", "reward/w_track",
        "reward/weighted_final_standing_total",
    ),
    "reference_tracking": (
        "reward/pose_tracking", "reward/velocity_tracking",
        "reward/root_xy_tracking", "reward/pelvis_height_tracking",
        "reward/orientation_tracking", "reward/weighted_pose_tracking",
        "reward/weighted_velocity_tracking", "reward/weighted_root_xy_tracking",
        "reward/weighted_pelvis_height_tracking",
        "reward/weighted_orientation_tracking", "reward/weighted_tracking_total",
    ),
    "pelvis_height": ("maximum_pelvis_height", "final_pelvis_height"),
    "uprightness": ("maximum_uprightness", "final_uprightness"),
    "ppo_losses": (
        "actor_loss", "critic_loss", "task_critic_loss", "style_critic_loss"
    ),
    "ppo_diagnostics": (
        "entropy", "approximate_kl", "clip_fraction", "gradient_norm",
        "action_std", "log_std",
    ),
    "throughput": (
        "environment_steps_per_second", "policy_inference_time",
        "simulation_rollout_time", "ppo_update_time",
    ),
    "actions_control": (
        "mean_absolute_policy_action", "mean_absolute_executed_action",
        "action_delta_magnitude", "mean_torque", "torque_saturation",
        "executed_action_clipping_fraction",
    ),
    "prior_residual": (
        "mean_absolute_prior_action", "mean_absolute_scaled_policy_action",
        "scaled_policy_to_prior_ratio", "prior_coefficient",
    ),
    "apex": (
        "task_return", "style_return", "task_advantage", "style_advantage",
        "combined_advantage",
    ),
    "frame_zero_evaluation": (
        "frame_zero/success", "frame_zero/episode_return",
        "frame_zero/maximum_consecutive_standing_duration",
        "frame_zero/maximum_pelvis_height", "frame_zero/final_pelvis_height",
        "frame_zero/maximum_uprightness", "frame_zero/final_uprightness",
        "frame_zero/final_root_linear_speed",
        "frame_zero/final_root_angular_speed",
    ),
}


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "metric"


def _plot_metrics(
    records: Sequence[MetricRecord],
    metric_names: Sequence[str],
    destination: Path,
    smoothing_window: int,
    title: str,
) -> bool:
    available: list[tuple[str, NDArray[np.int64], NDArray[np.float64]]] = []
    for name in metric_names:
        steps, values = metric_series(records, name)
        if len(values):
            available.append((name, steps, values))
    if not available:
        return False
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for name, steps, values in available:
        axis.plot(
            steps,
            values,
            marker=".",
            alpha=0.3 if smoothing_window > 1 else 1.0,
        )
        if smoothing_window > 1:
            axis.plot(
                steps,
                moving_average(values, smoothing_window),
                marker=".",
                label=name,
            )
        else:
            axis.lines[-1].set_label(name)
    axis.set_title(title)
    axis.set_xlabel("environment steps")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return True


def generate_training_plots(
    run_directory: Path | str,
    *,
    smoothing_window: int = 10,
    output_directory: Path | str | None = None,
) -> list[Path]:
    """Regenerate grouped and per-metric plots entirely from raw scalar CSV."""
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")
    run = Path(run_directory)
    records = load_metrics(run)
    destination = Path(output_directory) if output_directory else run / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/apex-getup-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    written: list[Path] = []
    groups = dict(DEFAULT_GROUPS)
    reward_metrics = sorted(
        {record.metric for record in records if record.metric.startswith("reward/")}
    )
    if reward_metrics:
        groups["reward_components"] = tuple(reward_metrics)
    for group, names in groups.items():
        path = destination / f"{group}.png"
        if _plot_metrics(records, names, path, smoothing_window, group.replace("_", " ")):
            written.append(path)
    individual = destination / "metrics"
    individual.mkdir(parents=True, exist_ok=True)
    for name in sorted(
        {record.metric for record in records if record.axis == "environment_step"}
    ):
        path = individual / f"{_safe_filename(name)}.png"
        if _plot_metrics(records, (name,), path, smoothing_window, name):
            written.append(path)
    return written


def plot_run_comparison(
    run_directories: Sequence[Path | str],
    output_directory: Path | str,
    *,
    metrics: Sequence[str] = (
        "episode_return", "success_rate", "maximum_pelvis_height", "maximum_uprightness"
    ),
    smoothing_window: int = 10,
) -> Path | None:
    """Compare metrics common to all readable runs, aligned by environment step."""
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")
    runs = [(Path(path), load_metrics(path)) for path in run_directories]
    runs = [(path, records) for path, records in runs if records]
    if not runs:
        return None
    common = [
        metric
        for metric in metrics
        if all(len(metric_series(records, metric)[0]) for _, records in runs)
    ]
    if not common:
        return None
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/apex-getup-matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2
    rows = int(math.ceil(len(common) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(13, 4.5 * rows), constrained_layout=True, squeeze=False
    )
    for axis, metric in zip(axes.flat, common):
        for path, records in runs:
            steps, values = metric_series(records, metric)
            axis.plot(
                steps,
                moving_average(values, smoothing_window),
                marker=".",
                label=path.name,
            )
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("environment steps")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(common):]:
        axis.set_visible(False)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "run_comparison.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
