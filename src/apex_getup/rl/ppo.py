"""PyTorch PPO with a tanh-squashed diagonal Gaussian actor."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from apex_getup.rl.buffer import RolloutBuffer
from apex_getup.rl.config import PPOConfig


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _mlp(
    input_size: int,
    hidden: tuple[int, ...],
    output_size: int,
    *,
    output_gain: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    sizes = (input_size, *hidden, output_size)
    for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        linear = nn.Linear(fan_in, fan_out)
        nn.init.xavier_normal_(
            linear.weight, gain=output_gain if index == len(sizes) - 2 else 1.0
        )
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if index != len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class PPOAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: PPOConfig | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.config.validate()
        self.observation_size = observation_size
        self.action_size = action_size
        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.actor = _mlp(
            observation_size,
            self.config.actor_hidden_dimensions,
            action_size,
            output_gain=0.01,
        ).to(self.device)
        self.critic = _mlp(
            observation_size,
            self.config.critic_hidden_dimensions,
            1,
            output_gain=1.0,
        ).to(self.device)
        self.log_std = nn.Parameter(
            torch.full(
                (action_size,),
                float(np.log(self.config.initial_action_std)),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters(), self.log_std],
            lr=self.config.learning_rate,
        )
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(self.config.seed)
        self.normalization_state: dict[str, object] | None = None

    @property
    def parameters(self) -> list[nn.Parameter]:
        return [*self.actor.parameters(), *self.critic.parameters(), self.log_std]

    def _observation_tensor(self, observation: NDArray[np.float64]) -> Tensor:
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device)

    def value(self, observation: NDArray[np.float64]) -> float:
        return float(self.values(np.asarray(observation)[None, :])[0])

    def values(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return values for a batch of observations in one PyTorch call."""
        array = np.asarray(observations, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.observation_size:
            raise ValueError("observations must have shape [batch, observation_size]")
        with torch.no_grad():
            values = self.critic(self._observation_tensor(array)).squeeze(-1)
        return values.cpu().numpy().astype(np.float64)

    def deterministic_action(
        self, observation: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        with torch.no_grad():
            mean = self.actor(self._observation_tensor(observation))
            action = torch.tanh(mean)
        return action.cpu().numpy().astype(np.float64)

    def sample_action(
        self, observation: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        actions, latents, log_probabilities, values = self.sample_actions(
            np.asarray(observation)[None, :]
        )
        return actions[0], latents[0], float(log_probabilities[0]), float(values[0])

    def sample_actions(
        self, observations: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Sample a batch of actions and values with one actor/critic pass."""
        array = np.asarray(observations, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.observation_size:
            raise ValueError("observations must have shape [batch, observation_size]")
        with torch.no_grad():
            observation_tensor = self._observation_tensor(array)
            mean = self.actor(observation_tensor)
            standard_deviation = self.log_std.exp()
            noise = torch.randn(
                mean.shape, generator=self.generator, device=self.device
            )
            latent = mean + standard_deviation * noise
            action = torch.tanh(latent)
            log_probability = self._log_probability_tensor(latent, action, mean)
            value = self.critic(observation_tensor).squeeze(-1)
        return (
            action.cpu().numpy().astype(np.float64),
            latent.cpu().numpy().astype(np.float64),
            log_probability.cpu().numpy().astype(np.float64),
            value.cpu().numpy().astype(np.float64),
        )

    def _log_probability_tensor(
        self, latent: Tensor, action: Tensor, mean: Tensor
    ) -> Tensor:
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        gaussian = distribution.log_prob(latent)
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (gaussian - correction).sum(dim=-1)

    def policy_log_probability(
        self,
        observations: NDArray[np.float64],
        latent_actions: NDArray[np.float64],
        policy_actions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Log probability of sampled policy actions, never executed actions."""
        with torch.no_grad():
            observations_tensor = self._observation_tensor(observations)
            latent_tensor = torch.as_tensor(
                latent_actions, dtype=torch.float32, device=self.device
            )
            action_tensor = torch.as_tensor(
                policy_actions, dtype=torch.float32, device=self.device
            )
            mean = self.actor(observations_tensor)
            result = self._log_probability_tensor(latent_tensor, action_tensor, mean)
        return result.cpu().numpy().astype(np.float64)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        size = buffer.transition_count
        if size <= 0:
            raise ValueError("cannot update PPO from an empty rollout")
        observations = torch.as_tensor(
            buffer.flatten(buffer.observations), dtype=torch.float32, device=self.device
        )
        latent_actions = torch.as_tensor(
            buffer.flatten(buffer.latent_actions), dtype=torch.float32, device=self.device
        )
        policy_actions = torch.as_tensor(
            buffer.flatten(buffer.policy_actions), dtype=torch.float32, device=self.device
        )
        old_log_probabilities = torch.as_tensor(
            buffer.flatten(buffer.log_probabilities),
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.as_tensor(
            buffer.flatten(buffer.advantages), dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            buffer.flatten(buffer.returns), dtype=torch.float32, device=self.device
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        metrics: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approximate_kl": [],
            "clip_fraction": [],
            "gradient_norm": [],
        }
        stop_early = False
        for _ in range(self.config.update_epochs):
            order = torch.randperm(size, generator=self.generator, device=self.device)
            for start in range(0, size, self.config.minibatch_size):
                indexes = order[start : start + self.config.minibatch_size]
                mean = self.actor(observations[indexes])
                values = self.critic(observations[indexes]).squeeze(-1)
                new_log_probability = self._log_probability_tensor(
                    latent_actions[indexes], policy_actions[indexes], mean
                )
                log_ratio = torch.clamp(
                    new_log_probability - old_log_probabilities[indexes], -20.0, 20.0
                )
                ratio = log_ratio.exp()
                batch_advantages = advantages[indexes]
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                ) * batch_advantages
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = torch.mean((values - returns[indexes]).square())
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
                metrics["actor_loss"].append(float(actor_loss.item()))
                metrics["critic_loss"].append(float(critic_loss.item()))
                metrics["entropy"].append(float(entropy.item()))
                metrics["approximate_kl"].append(float(approximate_kl.item()))
                metrics["clip_fraction"].append(float(clip_fraction.item()))
                metrics["gradient_norm"].append(float(gradient_norm.item()))
                if (
                    self.config.target_kl is not None
                    and float(approximate_kl.item()) > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break
        result = {name: float(np.mean(values)) for name, values in metrics.items()}
        result["early_stop_kl"] = float(stop_early)
        result["action_std"] = float(self.log_std.exp().mean().item())
        result["log_std"] = float(self.log_std.mean().item())
        return result

    def save(
        self,
        path: Path | str,
        *,
        metadata: dict[str, object] | None = None,
        normalization_state: dict[str, object] | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "config": asdict(self.config),
                "metadata": metadata or {},
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "log_std": self.log_std.detach().cpu(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "generator_state": self.generator.get_state(),
                "normalization_state": (
                    self.normalization_state
                    if normalization_state is None
                    else normalization_state
                ),
            },
            destination,
        )
        return destination

    def warm_start(
        self, path: Path | str, *, load_optimizer: bool = True
    ) -> dict[str, Any]:
        """Restore training state while retaining this new run configuration."""
        source = Path(path)
        if source.suffix.lower() == ".npz":
            raise ValueError("warm starts require a PyTorch .pt checkpoint")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        checkpoint_shape = (
            int(checkpoint["observation_size"]),
            int(checkpoint["action_size"]),
        )
        current_shape = (self.observation_size, self.action_size)
        if checkpoint_shape != current_shape:
            raise ValueError(
                "warm-start checkpoint dimensions do not match the current run: "
                f"checkpoint={checkpoint_shape}, current={current_shape}"
            )
        try:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
        except RuntimeError as error:
            raise ValueError(
                "warm-start checkpoint network architecture does not match "
                "the current PPO configuration"
            ) from error
        checkpoint_log_std = torch.as_tensor(checkpoint["log_std"])
        if tuple(checkpoint_log_std.shape) != tuple(self.log_std.shape):
            raise ValueError("warm-start checkpoint log_std has the wrong shape")
        with torch.no_grad():
            self.log_std.copy_(checkpoint_log_std.to(self.device))
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = self.config.learning_rate
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, Tensor):
                        state[key] = value.to(self.device)
        if "generator_state" in checkpoint:
            try:
                self.generator.set_state(checkpoint["generator_state"].cpu())
            except RuntimeError:
                warnings.warn(
                    "checkpoint RNG state is incompatible with the selected "
                    "device; using the configured seed instead",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.normalization_state = checkpoint.get("normalization_state")
        return dict(checkpoint.get("metadata", {}))

    @classmethod
    def load(
        cls, path: Path | str, *, device: str | None = "cpu"
    ) -> tuple["PPOAgent", dict[str, Any]]:
        source = Path(path)
        if source.suffix.lower() == ".npz":
            return cls._load_legacy_numpy_checkpoint(source, device=device)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
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
            int(checkpoint["observation_size"]),
            int(checkpoint["action_size"]),
            PPOConfig(**config_values),
        )
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        with torch.no_grad():
            agent.log_std.copy_(checkpoint["log_std"].to(agent.device))
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in agent.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, Tensor):
                        state[key] = value.to(agent.device)
        if "generator_state" in checkpoint:
            try:
                agent.generator.set_state(checkpoint["generator_state"].cpu())
            except RuntimeError:
                # CUDA and CPU generators use different state encodings. An
                # older CUDA checkpoint can still be evaluated on CPU; its
                # generator remains deterministically initialized from seed.
                warnings.warn(
                    "checkpoint RNG state is incompatible with the selected "
                    "device; using the configured seed instead",
                    RuntimeWarning,
                    stacklevel=2,
                )
        agent.normalization_state = checkpoint.get("normalization_state")
        return agent, dict(checkpoint.get("metadata", {}))

    @classmethod
    def _load_legacy_numpy_checkpoint(
        cls, path: Path, *, device: str | None
    ) -> tuple["PPOAgent", dict[str, Any]]:
        """Load pre-PyTorch checkpoints for evaluation with a fresh optimizer."""
        with np.load(path, allow_pickle=False) as archive:
            config_values = json.loads(str(archive["config_json"].item()))
            config_values["actor_hidden_dimensions"] = tuple(
                config_values["actor_hidden_dimensions"]
            )
            config_values["critic_hidden_dimensions"] = tuple(
                config_values["critic_hidden_dimensions"]
            )
            config_values["device"] = device or "cpu"
            agent = cls(
                int(archive["observation_size"]),
                int(archive["action_size"]),
                PPOConfig(**config_values),
            )
            with torch.no_grad():
                for prefix, network in (("actor", agent.actor), ("critic", agent.critic)):
                    linear_layers = [
                        module for module in network if isinstance(module, nn.Linear)
                    ]
                    for index, layer in enumerate(linear_layers):
                        layer.weight.copy_(torch.as_tensor(
                            archive[f"{prefix}_weight_{index}"].T,
                            dtype=torch.float32,
                            device=agent.device,
                        ))
                        layer.bias.copy_(torch.as_tensor(
                            archive[f"{prefix}_bias_{index}"],
                            dtype=torch.float32,
                            device=agent.device,
                        ))
                agent.log_std.copy_(torch.as_tensor(
                    archive["log_std"], dtype=torch.float32, device=agent.device
                ))
            metadata = json.loads(str(archive["metadata_json"].item()))
        return agent, metadata
