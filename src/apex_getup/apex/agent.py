"""Asymmetric multi-critic APEX PPO agent."""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from apex_getup.apex.advantages import combine_group_advantages
from apex_getup.apex.config import ApexConfig
from apex_getup.apex.critics import ApexCritic
from apex_getup.rl.ppo import (
    SquashedGaussianActor,
    resolve_device,
    squashed_gaussian_log_probability,
)

if TYPE_CHECKING:
    from apex_getup.apex.rollout import ApexRolloutBuffer
    from apex_getup.rl.config import PPOConfig
    from apex_getup.rl.normalization import VecNormalizer


class ApexAgent:
    """Reference-free actor with privileged style/task critics."""

    def __init__(
        self,
        actor_observation_size: int,
        critic_observation_size: int,
        action_size: int,
        config: "PPOConfig | None" = None,
        apex_config: ApexConfig | None = None,
    ) -> None:
        if min(actor_observation_size, critic_observation_size, action_size) <= 0:
            raise ValueError("APEX observation and action dimensions must be positive")
        if config is None:
            from apex_getup.rl.config import PPOConfig

            config = PPOConfig()
        self.config = config
        self.config.validate()
        self.apex_config = apex_config or ApexConfig()
        self.apex_config.validate()
        if self.config.backend != "native":
            raise ValueError("ApexAgent requires the native PyTorch PPO backend")
        self.actor_observation_size = int(actor_observation_size)
        self.critic_observation_size = int(critic_observation_size)
        self.action_size = int(action_size)

        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.policy = SquashedGaussianActor(
            self.actor_observation_size,
            self.action_size,
            self.config.actor_hidden_dimensions,
            self.config.initial_action_std,
        ).to(self.device)
        self.actor = self.policy.mean_network
        self.log_std = self.policy.log_std
        self.critic = ApexCritic(
            self.critic_observation_size,
            self.config.critic_hidden_dimensions,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=self.config.learning_rate,
        )
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(self.config.seed)
        self.normalization_state: dict[str, object] | None = None
        self.global_environment_steps = 0
        self.prior_schedule_state: dict[str, object] | None = None

    @property
    def parameters(self) -> list[nn.Parameter]:
        return [*self.actor.parameters(), *self.critic.parameters(), self.log_std]

    def sample_actions(
        self,
        actor_observations: NDArray[np.float64],
        critic_observations: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Sample policy actions and return style/task values.

        Actor and critic inputs are deliberately separate. Log probabilities
        correspond to the sampled, tanh-squashed policy action and never to an
        action modified later by the demonstration prior.
        """
        actor_array = np.asarray(actor_observations, dtype=np.float64)
        critic_array = np.asarray(critic_observations, dtype=np.float64)
        if (
            actor_array.ndim != 2
            or actor_array.shape[1] != self.actor_observation_size
        ):
            raise ValueError(
                "actor_observations must have shape "
                "[batch, actor_observation_size]"
            )
        if (
            critic_array.ndim != 2
            or critic_array.shape[1] != self.critic_observation_size
        ):
            raise ValueError(
                "critic_observations must have shape "
                "[batch, critic_observation_size]"
            )
        if actor_array.shape[0] != critic_array.shape[0]:
            raise ValueError("actor and critic observation batches must match")
        if not np.all(np.isfinite(actor_array)) or not np.all(
            np.isfinite(critic_array)
        ):
            raise ValueError("APEX observations must be finite")

        with torch.no_grad():
            actor_tensor = torch.as_tensor(
                actor_array, dtype=torch.float32, device=self.device
            )
            critic_tensor = torch.as_tensor(
                critic_array, dtype=torch.float32, device=self.device
            )
            policy_action, latent, log_probability = self.policy.sample(
                actor_tensor, self.generator
            )
            values = self.critic(critic_tensor)

        return (
            policy_action.cpu().numpy().astype(np.float64),
            latent.cpu().numpy().astype(np.float64),
            log_probability.cpu().numpy().astype(np.float64),
            values.cpu().numpy().astype(np.float64),
        )

    def deterministic_action(
        self,
        actor_observation: NDArray[np.float64],
        *,
        actor_normalizer: "VecNormalizer | None" = None,
    ) -> NDArray[np.float64]:
        """Return an actor-only deterministic action for deployment.

        A supplied normalizer must be frozen. No privileged critic observation
        is accepted by this API.
        """
        array = np.asarray(actor_observation, dtype=np.float64)
        if array.shape[-1:] != (self.actor_observation_size,):
            raise ValueError(
                "actor_observation must end with actor_observation_size"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("actor_observation must be finite")
        if actor_normalizer is not None:
            if actor_normalizer.training:
                raise ValueError("deployment actor normalizer must be frozen")
            if actor_normalizer.observation_size != self.actor_observation_size:
                raise ValueError("actor normalizer has the wrong observation size")
            array = actor_normalizer.normalize_observations(array, update=False)

        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                array,
                dtype=torch.float32,
                device=self.device,
            )
            action = torch.tanh(self.policy(observation_tensor))

        return action.cpu().numpy().astype(np.float64)

    def values(
        self, critic_observations: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return ``[batch, 2]`` values in ``(style, task)`` order."""
        critic_array = np.asarray(critic_observations, dtype=np.float64)
        if (
            critic_array.ndim != 2
            or critic_array.shape[1] != self.critic_observation_size
        ):
            raise ValueError("critic_observations have the wrong shape")
        if not np.all(np.isfinite(critic_array)):
            raise ValueError("critic_observations must be finite")
        with torch.no_grad():
            critic_tensor = torch.as_tensor(
                critic_array,
                dtype=torch.float32,
                device=self.device,
            )
            values = self.critic(critic_tensor)
        return values.cpu().numpy().astype(np.float64)

    def policy_log_probability(
        self,
        actor_observations: NDArray[np.float64],
        latent_actions: NDArray[np.float64],
        policy_actions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate policy-action likelihoods for PPO bookkeeping."""
        actor_array = np.asarray(actor_observations, dtype=np.float64)
        latents = np.asarray(latent_actions, dtype=np.float64)
        actions = np.asarray(policy_actions, dtype=np.float64)
        batch = actor_array.shape[0] if actor_array.ndim == 2 else -1
        if actor_array.shape != (batch, self.actor_observation_size):
            raise ValueError("actor_observations have the wrong shape")
        if (
            latents.shape != (batch, self.action_size)
            or actions.shape != latents.shape
        ):
            raise ValueError("policy actions have the wrong shape")
        if not all(
            np.all(np.isfinite(value))
            for value in (actor_array, latents, actions)
        ):
            raise ValueError("policy likelihood inputs must be finite")

        with torch.no_grad():
            actor_tensor = torch.as_tensor(
                actor_array, dtype=torch.float32, device=self.device
            )
            latent_tensor = torch.as_tensor(
                latents, dtype=torch.float32, device=self.device
            )
            action_tensor = torch.as_tensor(
                actions, dtype=torch.float32, device=self.device
            )
            log_probability = self.policy.log_probability(
                actor_tensor, latent_tensor, action_tensor
            )
        return log_probability.cpu().numpy().astype(np.float64)

    def update(self, buffer: "ApexRolloutBuffer") -> dict[str, float]:
        """Run one asymmetric multi-critic PPO update."""
        size = buffer.transition_count
        if size <= 0:
            raise ValueError("cannot update APEX from an empty rollout")

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, dtype=torch.float32, device=self.device)

        actor_observations = tensor(buffer.flatten(buffer.actor_observations))
        critic_observations = tensor(buffer.flatten(buffer.critic_observations))
        latent_actions = tensor(buffer.flatten(buffer.latent_actions))
        policy_actions = tensor(buffer.flatten(buffer.policy_actions))
        old_log_probabilities = tensor(buffer.flatten(buffer.log_probabilities))
        group_returns = tensor(buffer.flatten(buffer.group_returns))
        group_advantages = buffer.flatten(buffer.group_advantages)
        combined = combine_group_advantages(
            group_advantages,
            self.apex_config.advantage_weights,
            normalize=self.apex_config.normalize_group_advantages,
        )
        actor_advantages = tensor(combined)
        metrics: dict[str, list[float]] = {
            name: []
            for name in (
                "actor_loss",
                "critic_loss",
                "style_critic_loss",
                "task_critic_loss",
                "entropy",
                "approximate_kl",
                "clip_fraction",
                "gradient_norm",
            )
        }
        stop_early = False
        for _ in range(self.config.update_epochs):
            order = torch.randperm(
                size, generator=self.generator, device=self.device
            )
            for start in range(0, size, self.config.minibatch_size):
                indexes = order[start : start + self.config.minibatch_size]
                mean = self.policy(actor_observations[indexes])
                values = self.critic(critic_observations[indexes])
                new_log_probability = squashed_gaussian_log_probability(
                    latent_actions[indexes],
                    policy_actions[indexes],
                    mean,
                    self.log_std,
                )
                log_ratio = torch.clamp(
                    new_log_probability - old_log_probabilities[indexes],
                    -20.0,
                    20.0,
                )
                ratio = log_ratio.exp()
                batch_advantages = actor_advantages[indexes]
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                errors = (values - group_returns[indexes]).square()
                style_critic_loss = errors[:, 0].mean()
                task_critic_loss = errors[:, 1].mean()
                critic_loss = style_critic_loss + task_critic_loss
                entropy = torch.distributions.Normal(
                    mean, self.log_std.exp()
                ).entropy().sum(dim=-1).mean()
                loss = (
                    actor_loss
                    + self.config.value_coefficient * critic_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.parameters, self.config.maximum_gradient_norm
                )
                self.optimizer.step()
                with torch.no_grad():
                    self.log_std.clamp_(-4.0, 1.0)
                    approximate_kl = (
                        old_log_probabilities[indexes] - new_log_probability
                    ).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0) > self.config.clip_range
                    ).float().mean()
                values_by_name = {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "style_critic_loss": style_critic_loss,
                    "task_critic_loss": task_critic_loss,
                    "entropy": entropy,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for name, value in values_by_name.items():
                    metrics[name].append(float(value.item()))
                if (
                    self.config.target_kl is not None
                    and approximate_kl.item() > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break
        result = {name: float(np.mean(items)) for name, items in metrics.items()}
        result.update(
            early_stop_kl=float(stop_early),
            action_std=float(self.log_std.exp().mean().item()),
            log_std=float(self.log_std.mean().item()),
            style_advantage=float(np.mean(group_advantages[:, 0])),
            task_advantage=float(np.mean(group_advantages[:, 1])),
            combined_advantage=float(np.mean(combined)),
            style_return=float(
                np.mean(buffer.flatten(buffer.group_returns)[:, 0])
            ),
            task_return=float(
                np.mean(buffer.flatten(buffer.group_returns)[:, 1])
            ),
        )
        return result

    def save(
        self,
        path: Path | str,
        *,
        metadata: dict[str, object] | None = None,
        normalization_state: dict[str, object] | None = None,
    ) -> Path:
        """Save APEX networks, RNG state, and both observation normalizers."""
        selected_normalization_state = (
            self.normalization_state
            if normalization_state is None
            else normalization_state
        )
        if selected_normalization_state is not None:
            from apex_getup.rl.normalization import (
                load_apex_observation_normalizers,
            )

            load_apex_observation_normalizers(
                selected_normalization_state,
                training=False,
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "actor_observation_size": self.actor_observation_size,
                "critic_observation_size": self.critic_observation_size,
                "action_size": self.action_size,
                "config": asdict(self.config),
                "apex_config": asdict(self.apex_config),
                "metadata": metadata or {},
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "log_std": self.log_std.detach().cpu(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "generator_state": self.generator.get_state(),
                "apex_normalization_state": selected_normalization_state,
                "global_environment_steps": self.global_environment_steps,
                "prior_schedule_state": self.prior_schedule_state,
            },
            destination,
        )
        self.normalization_state = selected_normalization_state
        return destination

    def warm_start(
        self, path: Path | str, *, load_optimizer: bool = True
    ) -> dict[str, Any]:
        """Restore compatible APEX training state into this agent."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected = (
            self.actor_observation_size,
            self.critic_observation_size,
            self.action_size,
        )
        actual = (
            int(checkpoint["actor_observation_size"]),
            int(checkpoint["critic_observation_size"]),
            int(checkpoint["action_size"]),
        )
        if actual != expected:
            raise ValueError(
                "warm-start checkpoint dimensions do not match the APEX agent"
            )
        try:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
        except RuntimeError as error:
            raise ValueError(
                "APEX checkpoint network architecture mismatch"
            ) from error
        checkpoint_log_std = torch.as_tensor(checkpoint["log_std"])
        if tuple(checkpoint_log_std.shape) != tuple(self.log_std.shape):
            raise ValueError("APEX checkpoint log_std has the wrong shape")
        with torch.no_grad():
            self.log_std.copy_(checkpoint_log_std.to(self.device))
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
        self._restore_generator(checkpoint)
        self.normalization_state = self._checkpoint_normalization_state(checkpoint)
        self.global_environment_steps = int(
            checkpoint.get("global_environment_steps", 0)
        )
        self.prior_schedule_state = checkpoint.get("prior_schedule_state")
        return dict(checkpoint.get("metadata", {}))

    @classmethod
    def load(
        cls, path: Path | str, *, device: str | None = "cpu"
    ) -> tuple["ApexAgent", dict[str, Any]]:
        """Construct an APEX agent from a full training checkpoint."""
        from apex_getup.rl.config import PPOConfig

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config_values = dict(checkpoint["config"])
        config_values["actor_hidden_dimensions"] = tuple(
            config_values["actor_hidden_dimensions"]
        )
        config_values["critic_hidden_dimensions"] = tuple(
            config_values["critic_hidden_dimensions"]
        )
        if device is not None:
            config_values["device"] = device
        agent = cls(
            int(checkpoint["actor_observation_size"]),
            int(checkpoint["critic_observation_size"]),
            int(checkpoint["action_size"]),
            PPOConfig(**config_values),
            ApexConfig(**dict(checkpoint.get("apex_config", {}))),
        )
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        with torch.no_grad():
            agent.log_std.copy_(checkpoint["log_std"].to(agent.device))
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in agent.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(agent.device)
        agent._restore_generator(checkpoint)
        agent.normalization_state = agent._checkpoint_normalization_state(checkpoint)
        agent.global_environment_steps = int(
            checkpoint.get("global_environment_steps", 0)
        )
        agent.prior_schedule_state = checkpoint.get("prior_schedule_state")
        return agent, dict(checkpoint.get("metadata", {}))

    @staticmethod
    def _checkpoint_normalization_state(
        checkpoint: dict[str, Any],
    ) -> dict[str, object] | None:
        state = checkpoint.get("apex_normalization_state")
        if state is None:
            return None
        from apex_getup.rl.normalization import load_apex_observation_normalizers

        load_apex_observation_normalizers(state, training=False)
        return dict(state)

    def _restore_generator(self, checkpoint: dict[str, Any]) -> None:
        if "generator_state" not in checkpoint:
            return
        try:
            self.generator.set_state(checkpoint["generator_state"].cpu())
        except RuntimeError:
            warnings.warn(
                "checkpoint RNG state is incompatible with the selected "
                "device; using the configured seed instead",
                RuntimeWarning,
                stacklevel=2,
            )
