"""Privileged two-head value function used by full APEX training."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from apex_getup.apex.rewards import APEX_REWARD_GROUPS


class ApexCritic(nn.Module):
    """Shared privileged-state trunk with style and task value heads.

    Output order is fixed to ``(style, task)``. The actor is deliberately not
    part of this module so its reference-free observation boundary stays clear.
    """

    def __init__(
        self,
        observation_size: int,
        hidden_dimensions: tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        if observation_size <= 0:
            raise ValueError("observation_size must be positive")
        if not hidden_dimensions or any(size <= 0 for size in hidden_dimensions):
            raise ValueError("hidden_dimensions must contain positive sizes")

        layers: list[nn.Module] = []
        input_size = observation_size
        for output_size in hidden_dimensions:
            linear = nn.Linear(input_size, output_size)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.Tanh()))
            input_size = output_size
        self.trunk = nn.Sequential(*layers)
        self.style_head = nn.Linear(input_size, 1)
        self.task_head = nn.Linear(input_size, 1)
        for head in (self.style_head, self.task_head):
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)

    @property
    def value_names(self) -> tuple[str, str]:
        return APEX_REWARD_GROUPS

    def forward(self, observations: Tensor) -> Tensor:
        if observations.shape[-1] != self.trunk[0].in_features:
            raise ValueError("unexpected privileged observation size")
        features = self.trunk(observations)
        return torch.cat(
            (self.style_head(features), self.task_head(features)), dim=-1
        )
