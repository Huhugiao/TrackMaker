"""On-policy rollout workers for competence-gated multi-style Attacker PPO."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import ray
import torch

from attacker.multistyle_rewards import validate_reward_styles
from attacker.ppo_model import AttackerNMNMLPPPOActorCritic
from configs.attacker_config import MultiStylePPOParameters
from envs.attacker_env import AttackerEnv


def _terminal_outcome(info, truncated=False):
    reason = str(dict(info or {}).get("reason", "")).strip().lower()
    target_success = reason == "attacker_caught_target"
    defender_capture = reason == "defender_caught_attacker"
    timeout = bool(
        ("timeout" in reason)
        or ("time_limit" in reason)
        or ("max_steps" in reason)
        or (truncated and not target_success and not defender_capture)
    )
    return {
        "reason": reason or "unknown",
        "target_success": float(target_success),
        "defender_capture": float(defender_capture),
        "timeout": float(timeout),
        "defender_collision": float(reason in {"defender_collision", "defender_out"}),
    }


class _MultiStylePPORunnerCore:
    """One environment process permanently assigned to one reward style."""

    def __init__(
        self,
        worker_id: int,
        env_config: Mapping,
        reward_styles: Sequence[Mapping],
        seed: int,
        eval_seed: int,
        style_id: int | None = None,
    ):
        self.worker_id = int(worker_id)
        self.reward_styles = validate_reward_styles(reward_styles)
        self.style_names = tuple(style["name"] for style in self.reward_styles)
        self.num_styles = len(self.reward_styles)
        self.style_id = (
            self.worker_id % self.num_styles
            if style_id is None
            else int(style_id)
        )
        if self.style_id < 0 or self.style_id >= self.num_styles:
            raise ValueError(f"worker reward style id is out of range: {self.style_id}")
        self.style_name = self.style_names[self.style_id]
        self.seed = int(seed)
        self.eval_seed = int(eval_seed)
        # Ray workers are separate processes, but relying on inherited/default
        # Torch RNG state can still correlate their exploration streams.
        torch.manual_seed(self.seed + self.worker_id * 1_000_003)
        self.env_config = dict(env_config)
        obsolete_safety_keys = {
            "attacker_hard_action_mask",
            "attacker_hard_action_mask_params",
        } & set(self.env_config)
        if obsolete_safety_keys:
            raise ValueError(
                "multi-style PPO rejects retired Attacker safety-layer options: "
                f"{sorted(obsolete_safety_keys)}"
            )
        self.env_config["reward_styles"] = self.reward_styles
        self.env = AttackerEnv(**self.env_config)
        if bool(getattr(self.env, "attacker_hard_action_mask", False)):
            raise RuntimeError("Attacker action safety layer unexpectedly enabled")
        self.network = AttackerNMNMLPPPOActorCritic(self.num_styles).cpu().eval()
        self.train_episode_index = 0
        self.obs = None
        self.episode_reward = 0.0
        self.episode_len = 0
        self.current_reset_info = {}
        self._active_distribution = None
        self._active_distribution_signature = None
        self._reset(for_eval=False)

    @staticmethod
    def _distribution_signature(bins) -> tuple:
        signature = []
        for item in bins:
            entry = dict(item)
            signature.append(tuple(sorted((str(key), repr(value)) for key, value in entry.items())))
        return tuple(signature)

    def set_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        state = {
            name: torch.as_tensor(value, device="cpu")
            for name, value in weights.items()
        }
        self.network.load_state_dict(state, strict=True)
        self.network.eval()

    def sampling_state(self):
        """Return a checkpoint-safe state that never repeats a partial episode."""
        resume_episode_index = int(self.train_episode_index)
        if self.episode_len > 0:
            resume_episode_index += 1
        return {
            "worker_id": int(self.worker_id),
            "train_episode_index": resume_episode_index,
            "torch_rng_state": torch.get_rng_state().clone(),
        }

    def set_sampling_state(self, state: Mapping) -> None:
        if int(state.get("worker_id", self.worker_id)) != self.worker_id:
            raise ValueError("PPO sampling state belongs to a different worker")
        self.train_episode_index = int(state["train_episode_index"])
        torch.set_rng_state(
            torch.as_tensor(state["torch_rng_state"], dtype=torch.uint8).cpu()
        )
        self._reset(for_eval=False)

    def _set_distribution(self, bins, force_reset=False) -> None:
        bins = tuple(dict(item) for item in bins)
        signature = self._distribution_signature(bins)
        changed = signature != self._active_distribution_signature
        if changed:
            self.env.set_curriculum_bins(bins, enabled=True)
            self._active_distribution = bins
            self._active_distribution_signature = signature
        if changed or force_reset:
            self._reset(for_eval=False)

    def _train_seed(self) -> int:
        return int(self.seed + self.worker_id * 100_000 + self.train_episode_index)

    def _reset(self, for_eval=False, episode_index=0):
        seed = (
            int(self.eval_seed + int(episode_index))
            if for_eval
            else self._train_seed()
        )
        obs, info = self.env.reset(seed=seed)
        self.obs = np.asarray(obs, dtype=np.float32)
        self.current_reset_info = dict(info or {})
        self.episode_reward = 0.0
        self.episode_len = 0
        return self.current_reset_info

    @torch.no_grad()
    def _policy_step(self, style_id: int, deterministic=False):
        critic_obs = torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)
        style_ids = torch.tensor([int(style_id)], dtype=torch.long)
        action, pre_tanh, log_prob, value = self.network.step(
            critic_obs,
            style_ids,
            deterministic=bool(deterministic),
        )
        return (
            action[0].cpu().numpy().astype(np.float32, copy=False),
            pre_tanh[0].cpu().numpy().astype(np.float32, copy=False),
            float(log_prob.item()),
            float(value.item()),
        )

    def collect(
        self,
        num_steps: int,
        weights,
        curriculum_bins,
        policy_version: int,
    ):
        self.set_weights(weights)
        self._set_distribution(curriculum_bins)
        num_steps = int(num_steps)
        arrays = {
            "obs": [],
            "pre_tanh_actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "style_ids": [],
        }
        episodes = []
        curriculum_steps = Counter()
        last_done = False

        for _ in range(num_steps):
            action, pre_tanh, log_prob, value = self._policy_step(
                self.style_id,
                deterministic=False,
            )
            previous_obs = self.obs.copy()
            next_obs, _legacy_reward, terminated, truncated, info = self.env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            done = bool(terminated or truncated)
            reward_vector = np.asarray(
                info.get("attacker_reward_vector"), dtype=np.float32
            ).reshape(-1)
            if reward_vector.shape != (self.num_styles,):
                raise RuntimeError(
                    f"environment returned reward vector {reward_vector.shape}; "
                    f"expected ({self.num_styles},)"
                )
            reward = float(reward_vector[self.style_id])
            if not bool(
                np.isfinite(previous_obs).all()
                and np.isfinite(pre_tanh).all()
                and np.isfinite(reward)
                and np.isfinite(next_obs).all()
            ):
                raise FloatingPointError("non-finite parallel Attacker PPO transition")

            arrays["obs"].append(previous_obs)
            arrays["pre_tanh_actions"].append(pre_tanh)
            arrays["log_probs"].append(log_prob)
            arrays["values"].append(value)
            arrays["rewards"].append(reward)
            arrays["dones"].append(float(done))
            arrays["style_ids"].append(self.style_id)
            self.obs = next_obs
            self.episode_reward += reward
            self.episode_len += 1
            curriculum_steps[str(info.get("curriculum_bin", "default"))] += 1
            last_done = done

            if done:
                episodes.append(
                    {
                        **_terminal_outcome(info, truncated=bool(truncated)),
                        "behavior_style": self.style_name,
                        "behavior_style_id": self.style_id,
                        "episode_len": int(self.episode_len),
                        "episode_reward": float(self.episode_reward),
                        "curriculum_bin": str(info.get("curriculum_bin", "default")),
                        "obstacle_density": str(info.get("obstacle_density", "unknown")),
                        "defender_strategy": str(info.get("defender_strategy", "unknown")),
                    }
                )
                self.train_episode_index += 1
                self._reset(for_eval=False)

        if last_done:
            last_value = 0.0
        else:
            _action, _pre_tanh, _log_prob, last_value = self._policy_step(
                self.style_id,
                deterministic=True,
            )
        rewards = np.asarray(arrays["rewards"], dtype=np.float32)
        dones = np.asarray(arrays["dones"], dtype=np.float32)
        values = np.asarray(arrays["values"], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for index in reversed(range(num_steps)):
            next_value = last_value if index == num_steps - 1 else float(values[index + 1])
            keep = 1.0 - float(dones[index])
            delta = (
                float(rewards[index])
                + float(MultiStylePPOParameters.GAMMA) * next_value * keep
                - float(values[index])
            )
            last_gae = (
                delta
                + float(MultiStylePPOParameters.GAMMA)
                * float(MultiStylePPOParameters.GAE_LAMBDA)
                * keep
                * last_gae
            )
            advantages[index] = last_gae

        return {
            "obs": np.asarray(arrays["obs"], dtype=np.float32),
            "pre_tanh_actions": np.asarray(
                arrays["pre_tanh_actions"], dtype=np.float32
            ),
            "log_probs": np.asarray(arrays["log_probs"], dtype=np.float32),
            "values": values,
            "returns": advantages + values,
            "advantages": advantages,
            "dones": dones,
            "style_ids": np.asarray(arrays["style_ids"], dtype=np.int16),
            "episodes": episodes,
            "curriculum_steps": dict(curriculum_steps),
            "policy_version": int(policy_version),
            "worker_id": self.worker_id,
            "style_name": self.style_name,
            "sampling_state": self.sampling_state(),
        }

    def evaluate_style(
        self,
        weights,
        style_id: int,
        num_episodes: int,
        curriculum_bins,
    ):
        self.set_weights(weights)
        style_id = int(style_id)
        if style_id < 0 or style_id >= self.num_styles:
            raise ValueError(f"reward style id out of range: {style_id}")
        style_name = self.style_names[style_id]
        previous_distribution = self._active_distribution
        outcomes = []
        try:
            self.env.set_curriculum_bins(curriculum_bins, enabled=True)
            for episode_index in range(int(num_episodes)):
                reset_info = self._reset(for_eval=True, episode_index=episode_index)
                episode_reward = 0.0
                done = False
                while not done:
                    action, _pre_tanh, _log_prob, _value = self._policy_step(
                        style_id,
                        deterministic=True,
                    )
                    next_obs, _legacy_reward, terminated, truncated, info = self.env.step(action)
                    self.obs = np.asarray(next_obs, dtype=np.float32)
                    reward_vector = np.asarray(
                        info["attacker_reward_vector"], dtype=np.float32
                    )
                    episode_reward += float(reward_vector[style_id])
                    self.episode_len += 1
                    done = bool(terminated or truncated)
                outcomes.append(
                    {
                        **_terminal_outcome(info, truncated=bool(truncated)),
                        "seed": int(self.eval_seed + episode_index),
                        "episode_reward": float(episode_reward),
                        "episode_len": int(self.episode_len),
                        "curriculum_bin": str(
                            info.get(
                                "curriculum_bin",
                                reset_info.get("curriculum_bin", "default"),
                            )
                        ),
                        "obstacle_density": str(
                            info.get(
                                "obstacle_density",
                                reset_info.get("obstacle_density", "unknown"),
                            )
                        ),
                    }
                )
        finally:
            if previous_distribution is not None:
                self.env.set_curriculum_bins(previous_distribution, enabled=True)
            self._reset(for_eval=False)
        return {
            style_name: {
                "episodes": len(outcomes),
                "target_success_rate": float(
                    np.mean([item["target_success"] for item in outcomes])
                ),
                "defender_capture_rate": float(
                    np.mean([item["defender_capture"] for item in outcomes])
                ),
                "timeout_rate": float(np.mean([item["timeout"] for item in outcomes])),
                "defender_collision_rate": float(
                    np.mean([item["defender_collision"] for item in outcomes])
                ),
                "mean_reward": float(
                    np.mean([item["episode_reward"] for item in outcomes])
                ),
                "mean_episode_len": float(
                    np.mean([item["episode_len"] for item in outcomes])
                ),
                "outcomes": outcomes,
            }
        }

    def close(self):
        self.env.close()


class LocalMultiStylePPORunner(_MultiStylePPORunnerCore):
    """Local worker used by unit tests and dry-run diagnostics."""


@ray.remote(num_cpus=1, num_gpus=0)
class MultiStylePPORunner(_MultiStylePPORunnerCore):
    """Ray worker for style-stratified on-policy collection."""


__all__ = ["LocalMultiStylePPORunner", "MultiStylePPORunner"]
