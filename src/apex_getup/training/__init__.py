"""Algorithm-independent training metrics and visualization."""

from apex_getup.training.config import TrainingRunConfig, load_training_config
from apex_getup.rl.normalization import NormalizationConfig

from apex_getup.training.metrics import (
    MetricRecord,
    ScalarLogger,
    TrainingMetrics,
    generate_training_plots,
    load_metrics,
    mean_scalar_metrics,
    metric_series,
    moving_average,
    plot_run_comparison,
)
from apex_getup.training.selection import (
    FRAME_ZERO_SELECTION_METRIC,
    frame_zero_selection_key,
    is_better_frame_zero_evaluation,
)

__all__ = [
    "MetricRecord",
    "NormalizationConfig",
    "ScalarLogger",
    "TrainingMetrics",
    "TrainingRunConfig",
    "FRAME_ZERO_SELECTION_METRIC",
    "frame_zero_selection_key",
    "generate_training_plots",
    "load_metrics",
    "load_training_config",
    "is_better_frame_zero_evaluation",
    "mean_scalar_metrics",
    "metric_series",
    "moving_average",
    "plot_run_comparison",
]
