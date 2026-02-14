import random
from typing import Optional

import gymnasium as gym
import numpy as np

from env import TADEnv
from rule_policies.attacker_global import SUPPORTED_STRATEGIES, TRAINING_STRATEGIES, AttackerGlobalPolicy


class BaselineEnv(gym.Env):
    """Single-level RL wrapper that controls attacker internally."""

    def __init__(self, attacker_strategy: str = "random", reward_mode: str = "standard"):
        super().__init__()
        self.env = TADEnv(reward_mode=reward_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.attacker_strategy_mode = attacker_strategy

        self._attacker_policy: Optional[AttackerGlobalPolicy] = None
        self._static_action = np.zeros(2, dtype=np.float32)
        self._init_attacker_policy(attacker_strategy)

    def _init_attacker_policy(self, strategy: str):
        mode = strategy.lower()
        if mode == "attacker_global":
            mode = "default"

        if mode == "static":
            self._attacker_policy = None
            return

        if mode == "random":
            mode = random.choice(TRAINING_STRATEGIES)

        if mode not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported attacker strategy for BaselineEnv: {strategy}")

        self._attacker_policy = AttackerGlobalPolicy(strategy=mode)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        if self.attacker_strategy_mode.lower() == "random":
            self._init_attacker_policy("random")
        elif self._attacker_policy is not None and hasattr(self._attacker_policy, "reset"):
            self._attacker_policy.reset()

        return obs, info

    def _get_attacker_action(self):
        mode = self.attacker_strategy_mode.lower()
        if mode == "static":
            return self._static_action

        if self._attacker_policy is None:
            self._init_attacker_policy(self.attacker_strategy_mode)

        attacker_obs = None
        if isinstance(self.env.current_obs, tuple) and len(self.env.current_obs) == 2:
            attacker_obs = self.env.current_obs[1]

        if attacker_obs is None:
            return self._static_action

        return self._attacker_policy.get_action(attacker_obs)

    def step(self, action):
        attacker_action = self._get_attacker_action()
        return self.env.step(action=action, attacker_action=attacker_action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
