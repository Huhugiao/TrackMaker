"""Feed-forward, multi-style PPO model for the Attacker.

The decentralized actor receives only body-frame task features and local
radar.  The CTDE value function receives the full 72-D training observation.
Reward styles share NMN-style encoders but retain independent policy/value
heads and exploration parameters.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from attacker.observations import build_decentralized_actor_observation
from attacker.multistyle_rewards import validate_reward_styles
from configs.attacker_config import MultiStylePPOParameters, NetParameters


NETWORK_TYPE = "attacker_nmn_mlp_multistyle_ppo"


def _init_linear(module: nn.Module, output_gain: Optional[float] = None) -> None:
    if not isinstance(module, nn.Linear):
        return
    gain = math.sqrt(2.0) if output_gain is None else float(output_gain)
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class AttackerNMNMLPPPOActorCritic(nn.Module):
    """NMN-MLP actor/critic with one pair of heads per reward style."""

    def __init__(self, num_styles: int):
        super().__init__()
        self.num_styles = int(num_styles)
        branch_dim = int(NetParameters.BRANCH_HIDDEN_DIM)
        hidden_dim = int(NetParameters.HIDDEN_DIM)

        self.actor_task_branch = nn.Sequential(
            nn.Linear(int(NetParameters.DECENTRALIZED_ACTOR_TASK_DIM), branch_dim),
            nn.ReLU(),
            nn.Linear(branch_dim, branch_dim),
            nn.ReLU(),
        )
        self.actor_obstacle_branch = nn.Sequential(
            nn.Linear(int(NetParameters.RADAR_DIM), branch_dim),
            nn.ReLU(),
            nn.Linear(branch_dim, branch_dim),
            nn.ReLU(),
        )
        self.actor_merge = nn.Sequential(
            nn.Linear(branch_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_heads = nn.ModuleList(
            nn.Linear(hidden_dim, int(NetParameters.ACTION_DIM))
            for _ in range(self.num_styles)
        )
        self.log_std = nn.Parameter(
            torch.full(
                (self.num_styles, int(NetParameters.ACTION_DIM)),
                float(MultiStylePPOParameters.INITIAL_LOG_STD),
            )
        )

        self.critic_task_branch = nn.Sequential(
            nn.Linear(int(NetParameters.SCALAR_LEN), branch_dim),
            nn.ReLU(),
            nn.Linear(branch_dim, branch_dim),
            nn.ReLU(),
        )
        self.critic_obstacle_branch = nn.Sequential(
            nn.Linear(int(NetParameters.RADAR_DIM), branch_dim),
            nn.ReLU(),
            nn.Linear(branch_dim, branch_dim),
            nn.ReLU(),
        )
        self.critic_merge = nn.Sequential(
            nn.Linear(branch_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_heads = nn.ModuleList(
            nn.Linear(hidden_dim, 1) for _ in range(self.num_styles)
        )

        self.apply(_init_linear)
        for head in self.policy_heads:
            _init_linear(head, output_gain=0.01)
        for head in self.value_heads:
            _init_linear(head, output_gain=1.0)

    @staticmethod
    def _select_heads(features, style_ids, heads):
        style_ids = style_ids.to(device=features.device, dtype=torch.long).reshape(-1)
        if features.dim() != 2 or style_ids.shape[0] != features.shape[0]:
            raise ValueError("observations and style ids require matching batch dimensions")
        all_outputs = torch.stack([head(features) for head in heads], dim=1)
        return all_outputs[
            torch.arange(features.shape[0], device=features.device), style_ids
        ]

    def forward(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        style_ids: torch.Tensor,
    ):
        if actor_obs.shape[-1] != int(NetParameters.DECENTRALIZED_ACTOR_OBS_DIM):
            raise ValueError("PPO actor observation has an invalid shape")
        if critic_obs.shape[-1] != int(NetParameters.PRIVILEGED_CRITIC_OBS_DIM):
            raise ValueError("PPO critic observation has an invalid shape")
        style_ids = style_ids.to(device=actor_obs.device, dtype=torch.long).reshape(-1)
        if bool(torch.any(style_ids < 0)) or bool(torch.any(style_ids >= self.num_styles)):
            raise ValueError("reward style id is out of range")

        task_dim = int(NetParameters.DECENTRALIZED_ACTOR_TASK_DIM)
        actor_features = self.actor_merge(
            torch.cat(
                [
                    self.actor_task_branch(actor_obs[..., :task_dim]),
                    self.actor_obstacle_branch(actor_obs[..., task_dim:]),
                ],
                dim=-1,
            )
        )
        critic_features = self.critic_merge(
            torch.cat(
                [
                    self.critic_task_branch(
                        critic_obs[..., : int(NetParameters.SCALAR_LEN)]
                    ),
                    self.critic_obstacle_branch(
                        critic_obs[..., int(NetParameters.SCALAR_LEN) :]
                    ),
                ],
                dim=-1,
            )
        )
        mean = self._select_heads(actor_features, style_ids, self.policy_heads)
        value = self._select_heads(critic_features, style_ids, self.value_heads)
        log_std = self.log_std[style_ids].clamp(
            float(MultiStylePPOParameters.LOG_STD_MIN),
            float(MultiStylePPOParameters.LOG_STD_MAX),
        )
        return mean, log_std, value.squeeze(-1)

    @staticmethod
    def log_prob(pre_tanh, mean, log_std):
        distribution = torch.distributions.Normal(mean, torch.exp(log_std))
        base_log_prob = distribution.log_prob(pre_tanh)
        correction = torch.log(1.0 - torch.tanh(pre_tanh).pow(2) + 1e-6)
        return (base_log_prob - correction).sum(dim=-1)

    def step(self, critic_obs, style_ids, deterministic=False):
        actor_obs = build_decentralized_actor_observation(critic_obs)
        mean, log_std, value = self(actor_obs, critic_obs, style_ids)
        pre_tanh = mean if deterministic else mean + torch.exp(log_std) * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_prob = self.log_prob(pre_tanh, mean, log_std)
        return action, pre_tanh, log_prob, value


class AttackerMultiStylePPO:
    """Training, inference and checkpoint wrapper for multi-style PPO."""

    network_type = NETWORK_TYPE
    is_recurrent = False

    def __init__(
        self,
        reward_styles: Sequence[Mapping],
        device="cpu",
        training: bool = True,
    ):
        self.reward_styles = validate_reward_styles(reward_styles)
        self.style_names = tuple(style["name"] for style in self.reward_styles)
        self.num_styles = len(self.reward_styles)
        self.device = torch.device(device)
        self.network = AttackerNMNMLPPPOActorCritic(self.num_styles).to(self.device)
        self.training_enabled = bool(training)
        self.optimizer = None
        if self.training_enabled:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=float(MultiStylePPOParameters.LEARNING_RATE),
                eps=1e-5,
            )

    def resolve_style_id(self, style) -> int:
        if isinstance(style, (int, np.integer)):
            style_id = int(style)
        else:
            name = str(style).strip().lower()
            if name not in self.style_names:
                raise ValueError(f"unknown reward style {style!r}; valid={self.style_names}")
            style_id = self.style_names.index(name)
        if style_id < 0 or style_id >= self.num_styles:
            raise ValueError(f"reward style id out of range: {style_id}")
        return style_id

    def weights(self) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.network.state_dict().items()
        }

    def set_weights(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        state = {
            name: torch.as_tensor(value, device=self.device)
            for name, value in state_dict.items()
        }
        self.network.load_state_dict(state, strict=True)

    def update_learning_rate(
        self,
        progress: float,
        initial: Optional[float] = None,
        final: Optional[float] = None,
    ) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        initial = float(
            MultiStylePPOParameters.LEARNING_RATE if initial is None else initial
        )
        final = float(MultiStylePPOParameters.LR_FINAL if final is None else final)
        learning_rate = initial + (final - initial) * progress
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        return learning_rate

    @torch.no_grad()
    def act(self, observation, style, deterministic=False) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim == 1:
            observation = observation[None, :]
        critic_obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if np.isscalar(style) or isinstance(style, str):
            style_ids = torch.full(
                (critic_obs.shape[0],),
                self.resolve_style_id(style),
                dtype=torch.long,
                device=self.device,
            )
        else:
            style_ids = torch.as_tensor(style, dtype=torch.long, device=self.device)
        action, _pre_tanh, _log_prob, _value = self.network.step(
            critic_obs,
            style_ids,
            deterministic=bool(deterministic),
        )
        return action.cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _style_balanced_mean(values, style_ids, num_styles):
        means = [
            values[style_ids == style_id].mean()
            for style_id in range(int(num_styles))
            if bool(torch.any(style_ids == style_id))
        ]
        if not means:
            raise ValueError("PPO minibatch contains no reward styles")
        return torch.stack(means).mean()

    def update(self, batch: Mapping[str, np.ndarray]) -> Dict[str, float]:
        if not self.training_enabled or self.optimizer is None:
            raise RuntimeError("PPO model was created for inference only")
        critic_obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actor_obs = build_decentralized_actor_observation(critic_obs)
        pre_tanh = torch.as_tensor(
            batch["pre_tanh_actions"], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.as_tensor(
            batch["log_probs"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        old_values = torch.as_tensor(
            batch["values"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        returns = torch.as_tensor(
            batch["returns"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        advantages = torch.as_tensor(
            batch["advantages"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        style_ids = torch.as_tensor(
            batch["style_ids"], dtype=torch.long, device=self.device
        ).reshape(-1)
        size = int(critic_obs.shape[0])
        if size <= 0 or any(
            tensor.shape[0] != size
            for tensor in (
                actor_obs,
                pre_tanh,
                old_log_probs,
                old_values,
                returns,
                advantages,
                style_ids,
            )
        ):
            raise ValueError("PPO batch arrays require matching non-empty leading dimensions")
        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                critic_obs,
                pre_tanh,
                old_log_probs,
                old_values,
                returns,
                advantages,
            )
        ):
            raise FloatingPointError("non-finite PPO batch")

        normalized_advantages = torch.zeros_like(advantages)
        value_scales = torch.ones_like(returns)
        for style_id in range(self.num_styles):
            mask = style_ids == style_id
            if not bool(torch.any(mask)):
                raise ValueError(f"PPO rollout omitted reward style {self.style_names[style_id]!r}")
            style_advantages = advantages[mask]
            if style_advantages.numel() > 1:
                normalized_advantages[mask] = (
                    style_advantages - style_advantages.mean()
                ) / (style_advantages.std(unbiased=False) + 1e-8)
            style_scale = returns[mask].std(unbiased=False).clamp_min(1.0)
            value_scales[mask] = style_scale

        minibatch_size = min(int(MultiStylePPOParameters.MINIBATCH_SIZE), size)
        style_indices = [
            torch.nonzero(style_ids == style_id, as_tuple=False).reshape(-1)
            for style_id in range(self.num_styles)
        ]
        if any(indices.numel() == 0 for indices in style_indices):
            raise ValueError("PPO update requires samples from every reward style")
        minibatches_per_epoch = max(1, int(math.ceil(size / minibatch_size)))
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        updates = 0
        stopped_early = False
        self.network.train()
        for _epoch in range(int(MultiStylePPOParameters.N_EPOCHS)):
            shuffled_by_style = [
                indices[torch.randperm(indices.numel(), device=self.device)]
                for indices in style_indices
            ]
            epoch_kls = []
            chunks_by_style = [
                torch.tensor_split(indices, minibatches_per_epoch)
                for indices in shuffled_by_style
            ]
            for minibatch_index in range(minibatches_per_epoch):
                mb = torch.cat(
                    [
                        chunks[minibatch_index]
                        for chunks in chunks_by_style
                    ],
                    dim=0,
                )
                mb = mb[torch.randperm(mb.numel(), device=self.device)]
                mb_styles = style_ids[mb]
                mean, log_std, new_values = self.network(
                    actor_obs[mb], critic_obs[mb], mb_styles
                )
                new_log_probs = self.network.log_prob(pre_tanh[mb], mean, log_std)
                log_ratio = new_log_probs - old_log_probs[mb]
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                unclipped = ratio * normalized_advantages[mb]
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(MultiStylePPOParameters.CLIP_RANGE),
                    1.0 + float(MultiStylePPOParameters.CLIP_RANGE),
                ) * normalized_advantages[mb]
                policy_loss = -self._style_balanced_mean(
                    torch.minimum(unclipped, clipped), mb_styles, self.num_styles
                )

                value_delta = new_values - old_values[mb]
                clipped_values = old_values[mb] + torch.clamp(
                    value_delta,
                    -float(MultiStylePPOParameters.VALUE_CLIP_RANGE),
                    float(MultiStylePPOParameters.VALUE_CLIP_RANGE),
                )
                value_loss_raw = torch.maximum(
                    (new_values - returns[mb]).pow(2),
                    (clipped_values - returns[mb]).pow(2),
                ) / value_scales[mb].pow(2)
                value_loss = self._style_balanced_mean(
                    value_loss_raw, mb_styles, self.num_styles
                )
                entropy_per_sample = torch.distributions.Normal(
                    mean, torch.exp(log_std)
                ).entropy().sum(dim=-1)
                entropy = self._style_balanced_mean(
                    entropy_per_sample, mb_styles, self.num_styles
                )
                loss = (
                    policy_loss
                    + float(MultiStylePPOParameters.VALUE_COEF) * value_loss
                    - float(MultiStylePPOParameters.ENTROPY_COEF) * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    float(MultiStylePPOParameters.MAX_GRAD_NORM),
                )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs()
                        > float(MultiStylePPOParameters.CLIP_RANGE)
                    ).float().mean()
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["approx_kl"] += float(approx_kl.item())
                totals["clip_fraction"] += float(clip_fraction.item())
                totals["grad_norm"] += float(grad_norm.item())
                updates += 1
                epoch_kls.append(float(approx_kl.item()))
            if epoch_kls and float(np.mean(epoch_kls)) > float(MultiStylePPOParameters.TARGET_KL):
                stopped_early = True
                break

        metrics = {key: value / max(updates, 1) for key, value in totals.items()}
        metrics["updates"] = float(updates)
        metrics["kl_early_stop"] = float(stopped_early)
        for style_id, style_name in enumerate(self.style_names):
            mask = style_ids == style_id
            metrics[f"return/{style_name}"] = float(returns[mask].mean().item())
            metrics[f"value/{style_name}"] = float(old_values[mask].mean().item())
            metrics[f"std/{style_name}"] = float(
                torch.exp(self.network.log_std[style_id].clamp(
                    float(MultiStylePPOParameters.LOG_STD_MIN),
                    float(MultiStylePPOParameters.LOG_STD_MAX),
                )).mean().item()
            )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite PPO metrics: {metrics}")
        return metrics

    def save(self, path, step: int = 0, extra_metadata: Optional[Mapping] = None) -> None:
        checkpoint = {
            "network_type": self.network_type,
            "model": self.weights(),
            "reward_styles": [dict(style) for style in self.reward_styles],
            "style_names": list(self.style_names),
            "step": int(step),
            "recurrent": False,
        }
        if self.optimizer is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint.update(dict(extra_metadata or {}))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    @classmethod
    def from_checkpoint(cls, path, device="cpu", training=False):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("network_type") != NETWORK_TYPE:
            raise ValueError(f"checkpoint is not {NETWORK_TYPE}")
        model = cls(checkpoint["reward_styles"], device=device, training=training)
        model.set_weights(checkpoint["model"])
        if training and model.optimizer is not None and "optimizer" in checkpoint:
            model.optimizer.load_state_dict(checkpoint["optimizer"])
        return model, checkpoint


__all__ = [
    "AttackerMultiStylePPO",
    "AttackerNMNMLPPPOActorCritic",
    "NETWORK_TYPE",
]
