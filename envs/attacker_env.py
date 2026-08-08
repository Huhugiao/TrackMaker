"""Attacker-centric environment wrapper for on- and off-policy training."""

import math
from typing import Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from attacker.defender_pool import (
    SUPPORTED_DEFENDER_STRATEGIES,
    create_defender_controller,
    normalize_defender_policy_specs,
)
from attacker.multistyle_rewards import (
    reward_feature_vector,
    reward_weight_matrix,
    validate_reward_styles,
)
from configs import map_config
from configs.attacker_config import RewardParameters
from envs.tad_env import TADEnv
from utils.path_risk import astar_path_length


class AttackerEnv(gym.Env):
    """Wrap `TADEnv` so the attacker becomes the learning agent."""

    def __init__(
        self,
        defender_strategy: str = "chase",
        defender_strategy_pool: Optional[Sequence[str]] = None,
        defender_strategy_weights: Optional[Dict[str, float]] = None,
        reward_mode: str = "standard",
        defender_strategy_params: Optional[Dict] = None,
        env_kwargs: Optional[Dict] = None,
        curriculum_enabled: bool = False,
        curriculum_bins: Optional[Sequence[Dict]] = None,
        curriculum_keep_base_speeds: bool = True,
        target_progress_metric: str = "euclidean",
        path_grid_size: float = 8.0,
        path_obstacle_padding: float = 10.0,
        reward_parameters: Optional[Dict[str, float]] = None,
        defender_policy_specs: Optional[Dict[str, Dict]] = None,
        reward_styles: Optional[Sequence[Dict]] = None,
    ):
        super().__init__()
        self.env = TADEnv(reward_mode=reward_mode, **dict(env_kwargs or {}))
        if bool(getattr(self.env, "attacker_hard_action_mask", False)):
            raise ValueError("AttackerEnv no longer supports an action safety layer")
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(72,),
            dtype=np.float32,
        )
        self.action_space = self.env.action_space

        self.defender_policy_specs = normalize_defender_policy_specs(defender_policy_specs)
        self.defender_strategy_mode = str(defender_strategy).strip().lower()
        self.defender_strategy_pool = self._normalize_pool(defender_strategy_pool)
        self.defender_strategy_weights = self._normalize_weights(
            defender_strategy_weights,
            self.defender_strategy_pool,
        )
        self.defender_strategy_params = dict(defender_strategy_params or {})
        self.curriculum_enabled = bool(curriculum_enabled)
        self.curriculum_bins = self._normalize_curriculum_bins(curriculum_bins)
        self.curriculum_keep_base_speeds = bool(curriculum_keep_base_speeds)
        self.curriculum_base_obstacle_density = str(
            getattr(map_config, "current_obstacle_density", map_config.ObstacleDensity.DENSE)
        )
        self.target_progress_metric = str(target_progress_metric).strip().lower()
        if self.target_progress_metric not in {"euclidean", "astar"}:
            raise ValueError(
                f"target_progress_metric must be 'euclidean' or 'astar', got {target_progress_metric!r}"
            )
        self.path_grid_size = float(path_grid_size)
        self.path_obstacle_padding = float(path_obstacle_padding)
        self.reward_parameters = {
            str(key): float(value)
            for key, value in dict(reward_parameters or {}).items()
        }
        self.reward_styles = (
            validate_reward_styles(reward_styles)
            if reward_styles is not None
            else ()
        )
        self.reward_style_names = tuple(
            style["name"] for style in self.reward_styles
        )
        self._reward_weight_matrix = (
            reward_weight_matrix(self.reward_styles)
            if self.reward_styles
            else None
        )
        self.current_curriculum_bin = "default"

        self.current_defender_strategy = None
        self.defender_policy = None
        self._defender_controllers: Dict[str, object] = {}
        self.current_obs = None
        self._attacker_collision_active = False
        self._attacker_collision_events = 0

    @staticmethod
    def _normalize_curriculum_bins(bins: Optional[Sequence[Dict]]) -> Tuple[Dict, ...]:
        if not bins:
            return ()
        normalized = []
        for index, item in enumerate(bins):
            entry = dict(item)
            name = str(entry.get("name", f"bin_{index}"))
            is_default = bool(entry.get("default_distribution", False))
            if is_default:
                weight = float(entry.get("weight", 1.0))
                if weight < 0.0:
                    raise ValueError(f"curriculum bin {index} has negative weight={weight}")
                normalized.append(
                    {
                        "name": name,
                        "default_distribution": True,
                        "weight": weight,
                    }
                )
                continue
            margin = tuple(float(v) for v in entry.get("time_margin", ()))
            if len(margin) != 2:
                raise ValueError(f"curriculum bin {index} requires time_margin=(low, high)")
            astar_distance = tuple(
                float(v) for v in entry.get("attacker_target_astar_distance", ())
            )
            if astar_distance and len(astar_distance) != 2:
                raise ValueError(
                    f"curriculum bin {index} requires "
                    "attacker_target_astar_distance=(low, high)"
                )
            weight = float(entry.get("weight", 1.0))
            if weight < 0.0:
                raise ValueError(f"curriculum bin {index} has negative weight={weight}")
            normalized_entry = {
                "name": name,
                "time_margin": (min(margin), max(margin)),
                "weight": weight,
            }
            if astar_distance:
                normalized_entry["attacker_target_astar_distance"] = (
                    min(astar_distance),
                    max(astar_distance),
                )

            raw_densities = entry.get("obstacle_densities", ())
            if isinstance(raw_densities, str):
                raw_densities = (raw_densities,)
            densities = tuple(
                str(value).strip().lower()
                for value in raw_densities
                if str(value).strip()
            )
            invalid_densities = sorted(
                set(densities) - set(map_config.ObstacleDensity.ALL_LEVELS)
            )
            if invalid_densities:
                raise ValueError(
                    f"curriculum bin {index} has invalid obstacle densities: {invalid_densities}"
                )
            if densities:
                raw_density_weights = entry.get("obstacle_density_weights", None)
                if raw_density_weights is None:
                    density_weights = tuple(1.0 for _ in densities)
                elif isinstance(raw_density_weights, dict):
                    density_weights = tuple(
                        float(raw_density_weights.get(name, 0.0)) for name in densities
                    )
                else:
                    density_weights = tuple(float(value) for value in raw_density_weights)
                if len(density_weights) != len(densities):
                    raise ValueError(
                        f"curriculum bin {index} obstacle density weights must match densities"
                    )
                if any(value < 0.0 for value in density_weights):
                    raise ValueError(
                        f"curriculum bin {index} has negative obstacle density weight"
                    )
                if sum(density_weights) <= 0.0:
                    raise ValueError(
                        f"curriculum bin {index} requires a positive obstacle density weight"
                    )
                normalized_entry["obstacle_densities"] = densities
                normalized_entry["obstacle_density_weights"] = density_weights
            normalized.append(normalized_entry)
        if sum(float(item["weight"]) for item in normalized) <= 0.0:
            raise ValueError("curriculum bins require a positive total weight")
        return tuple(normalized)

    def set_curriculum_bins(
        self,
        bins: Optional[Sequence[Dict]],
        enabled: bool = True,
    ) -> None:
        """Replace the episode distribution used by the active curriculum.

        Training workers call this only between PPO rollouts.  Keeping the
        mutation on the environment wrapper makes stage changes explicit and
        avoids rebuilding the learned Defender controller at every gate.
        """
        normalized = self._normalize_curriculum_bins(bins)
        if bool(enabled) and not normalized:
            raise ValueError("an enabled Attacker curriculum requires at least one bin")
        self.curriculum_bins = normalized
        self.curriculum_enabled = bool(enabled)

    def _sample_curriculum_options(self, seed=None) -> Dict:
        if not self.curriculum_enabled or not self.curriculum_bins:
            self.current_curriculum_bin = "default"
            return {}
        rng = np.random.default_rng(seed) if seed is not None else self.env.np_random
        weights = np.asarray([float(item["weight"]) for item in self.curriculum_bins], dtype=np.float64)
        weights /= weights.sum()
        index = int(rng.choice(np.arange(len(self.curriculum_bins)), p=weights))
        selected = self.curriculum_bins[index]
        self.current_curriculum_bin = str(selected["name"])
        if bool(selected.get("default_distribution", False)):
            # Omit geometry constraints but explicitly restore the native
            # obstacle density.  TADEnv otherwise retains the previous
            # episode's process-global density, which would contaminate a
            # default evaluation following an easy/scattered episode.
            return {"obstacle_density": self.curriculum_base_obstacle_density}
        options = {
            "regime": self.current_curriculum_bin,
            "curriculum_bin": self.current_curriculum_bin,
            "time_margin_range": tuple(selected["time_margin"]),
        }
        if "attacker_target_astar_distance" in selected:
            options["attacker_target_astar_distance_range"] = tuple(
                selected["attacker_target_astar_distance"]
            )
            options["curriculum_astar_grid_size"] = float(self.path_grid_size)
            options["curriculum_astar_obstacle_padding"] = float(
                self.path_obstacle_padding
            )
        densities = tuple(selected.get("obstacle_densities", ()))
        if densities:
            density_weights = np.asarray(
                selected["obstacle_density_weights"], dtype=np.float64
            )
            density_weights /= density_weights.sum()
            options["obstacle_density"] = str(
                rng.choice(np.asarray(densities, dtype=object), p=density_weights)
            )
        else:
            options["obstacle_density"] = self.curriculum_base_obstacle_density
        if self.curriculum_keep_base_speeds:
            options["attacker_speed"] = float(getattr(self.env, "_base_attacker_speed", map_config.attacker_speed))
            options["defender_speed"] = float(getattr(self.env, "_base_defender_speed", map_config.defender_speed))
        return options

    def _normalize_pool(self, pool: Optional[Sequence[str]]) -> Tuple[str, ...]:
        raw = pool or SUPPORTED_DEFENDER_STRATEGIES
        valid_strategies = tuple(SUPPORTED_DEFENDER_STRATEGIES) + tuple(self.defender_policy_specs)
        normalized = []
        for item in raw:
            key = str(item).strip().lower()
            if not key:
                continue
            if key not in valid_strategies:
                raise ValueError(
                    f"Unsupported defender strategy in attacker pool: {item}. "
                    f"Valid={valid_strategies}"
                )
            if key not in normalized:
                normalized.append(key)
        if not normalized:
            raise ValueError("defender_strategy_pool must contain at least one valid strategy.")
        return tuple(normalized)

    @staticmethod
    def _normalize_weights(
        weights: Optional[Dict[str, float]],
        pool: Sequence[str],
    ) -> Optional[Tuple[float, ...]]:
        if weights is None:
            return None

        normalized_weights = {
            str(key).strip().lower(): float(value)
            for key, value in weights.items()
        }
        normalized = []
        for item in pool:
            value = float(normalized_weights.get(item, 0.0))
            if value < 0.0:
                raise ValueError(f"defender_strategy_weights[{item!r}] must be non-negative, got {value}.")
            normalized.append(value)

        total = float(sum(normalized))
        if total <= 0.0:
            raise ValueError("defender_strategy_weights must assign a positive total weight to the active pool.")
        return tuple(normalized)

    def _sample_defender_strategy(self) -> str:
        mode = self.defender_strategy_mode
        if mode == "random":
            if self.defender_strategy_weights is not None:
                probabilities = np.asarray(self.defender_strategy_weights, dtype=np.float64)
                probabilities /= probabilities.sum()
                return str(self.env.np_random.choice(self.defender_strategy_pool, p=probabilities))
            return str(self.env.np_random.choice(self.defender_strategy_pool))
        if mode not in SUPPORTED_DEFENDER_STRATEGIES and mode not in self.defender_policy_specs:
            valid_strategies = tuple(SUPPORTED_DEFENDER_STRATEGIES) + tuple(self.defender_policy_specs)
            raise ValueError(
                f"Unsupported defender_strategy={self.defender_strategy_mode!r}. "
                f"Valid={valid_strategies} or 'random'."
            )
        return mode

    def _build_defender_policy(self, strategy: str):
        if strategy not in self._defender_controllers:
            controller = create_defender_controller(
                strategy=strategy,
                env=self.env,
                controller_config=self.defender_strategy_params,
                policy_specs=self.defender_policy_specs,
            )
            self._defender_controllers[strategy] = controller
        return self._defender_controllers[strategy]

    def _get_defender_action(self) -> np.ndarray:
        if self.defender_policy is None:
            raise RuntimeError("Defender policy is not initialized. Call reset() first.")
        if not (isinstance(self.env.current_obs, tuple) and len(self.env.current_obs) == 2):
            raise RuntimeError("Underlying env must expose (defender_obs, attacker_obs).")
        defender_obs, attacker_obs = self.env.current_obs
        return self.defender_policy.get_action(defender_obs, attacker_obs, self.env)

    @staticmethod
    def _center_xy(agent: Dict, pixel_size: float) -> Tuple[float, float]:
        return float(agent["x"] + pixel_size * 0.5), float(agent["y"] + pixel_size * 0.5)

    def _boundary_distance(self, src: Dict, dst: Dict, boundary: float) -> float:
        src_xy = self._center_xy(src, self.env.pixel_size)
        dst_xy = self._center_xy(dst, self.env.pixel_size)
        return max(0.0, math.hypot(dst_xy[0] - src_xy[0], dst_xy[1] - src_xy[1]) - float(boundary))

    def _reward_parameter(self, name: str) -> float:
        return float(self.reward_parameters.get(name, getattr(RewardParameters, name)))

    def _update_attacker_collision_event(self, collision_now: bool) -> Tuple[bool, bool]:
        collision_now = bool(collision_now)
        collision_event = bool(collision_now and not self._attacker_collision_active)
        self._attacker_collision_active = collision_now
        if collision_event:
            self._attacker_collision_events += 1
        max_events = max(0, int(self._reward_parameter("ATTACKER_COLLISION_MAX_EVENTS")))
        penalty_applied = bool(collision_event and self._attacker_collision_events <= max_events)
        return collision_event, penalty_applied

    def _target_path_distance(self, state: Dict, reach_radius: float) -> float:
        if self.target_progress_metric != "astar":
            return self._boundary_distance(state["attacker"], state["target"], reach_radius)
        attacker_xy = self._center_xy(state["attacker"], self.env.pixel_size)
        target_xy = self._center_xy(state["target"], self.env.pixel_size)
        path_length = astar_path_length(
            attacker_xy,
            target_xy,
            width=float(self.env.width),
            height=float(self.env.height),
            grid_size=self.path_grid_size,
            obstacle_padding=self.path_obstacle_padding,
            obstacles=getattr(map_config, "obstacles", []),
        )
        return max(0.0, float(path_length) - float(reach_radius))

    def _compute_attacker_reward(self, prev_state: Dict, next_state: Dict, info: Dict) -> Tuple[float, Dict]:
        agent_radius = float(getattr(map_config, "agent_radius", 8.0))
        target_radius = float(getattr(map_config, "target_radius", 16.0))
        capture_radius = float(getattr(map_config, "capture_radius", agent_radius * 2.0))
        reach_radius = target_radius + agent_radius
        map_diag = max(1.0, math.hypot(self.env.width, self.env.height))

        prev_target_dist = self._target_path_distance(prev_state, reach_radius)
        curr_target_dist = self._target_path_distance(next_state, reach_radius)
        target_progress = (prev_target_dist - curr_target_dist) / map_diag

        prev_def_dist = self._boundary_distance(prev_state["attacker"], prev_state["defender"], capture_radius)
        curr_def_dist = self._boundary_distance(next_state["attacker"], next_state["defender"], capture_radius)

        evade_reward = 0.0
        evade_threshold = self._reward_parameter("EVADE_DISTANCE_THRESHOLD")
        if min(prev_def_dist, curr_def_dist) < evade_threshold:
            prev_margin = min(prev_def_dist, evade_threshold)
            curr_margin = min(curr_def_dist, evade_threshold)
            evade_reward = (curr_margin - prev_margin) / max(evade_threshold, 1.0)

        step_penalty = self._reward_parameter("STEP_PENALTY")
        target_progress_scale = self._reward_parameter("TARGET_PROGRESS_SCALE")
        evade_progress_scale = self._reward_parameter("EVADE_PROGRESS_SCALE")
        attacker_collision_penalty = self._reward_parameter("ATTACKER_COLLISION_PENALTY")
        reward = step_penalty
        reward += target_progress_scale * float(target_progress)
        reward += evade_progress_scale * float(evade_reward)

        if info.get("attacker_collision_penalty_applied", False):
            reward += attacker_collision_penalty

        reason = str(info.get("reason", "")).strip().lower()
        target_success = False
        defender_collision_forfeit = False
        if reason == "attacker_caught_target":
            reward += self._reward_parameter("SUCCESS_REWARD")
            target_success = True
        elif reason == "defender_caught_attacker":
            reward += self._reward_parameter("CAPTURE_PENALTY")
        elif reason == "timeout_defender_wins":
            reward += self._reward_parameter("TIMEOUT_PENALTY")
        elif reason == "defender_collision":
            reward += self._reward_parameter("DEFENDER_COLLISION_REWARD")
            defender_collision_forfeit = True

        # These features are independent of the selected reward style.  The
        # off-policy trainer stores them once and relabels the transition for
        # every style in the shared replay buffer.
        reward_features = {
            "step": 1.0,
            "target_progress": float(target_progress),
            "evade_progress": float(evade_reward),
            "attacker_collision": float(
                bool(info.get("attacker_collision_penalty_applied", False))
            ),
            "target_success": float(reason == "attacker_caught_target"),
            "defender_capture": float(reason == "defender_caught_attacker"),
            "timeout": float(
                ("timeout" in reason)
                or ("time_limit" in reason)
                or ("max_steps" in reason)
            ),
            "defender_collision": float(reason == "defender_collision"),
        }
        info["attacker_reward_features"] = reward_features
        if self.reward_styles:
            style_rewards = self._reward_weight_matrix @ reward_feature_vector(
                reward_features
            )
            style_rewards = style_rewards.astype(np.float32, copy=False)
            info["attacker_reward_vector"] = style_rewards
            info["attacker_reward_by_style"] = {
                name: float(style_rewards[index])
                for index, name in enumerate(self.reward_style_names)
            }

        terms = {
            "step_penalty": step_penalty,
            "target_progress": float(target_progress_scale * target_progress),
            "evade_progress": float(evade_progress_scale * evade_reward),
            "attacker_collision": float(
                attacker_collision_penalty
                if info.get("attacker_collision_penalty_applied", False)
                else 0.0
            ),
            "terminal_bonus": float(
                reward
                - step_penalty
                - float(target_progress_scale * target_progress)
                - float(evade_progress_scale * evade_reward)
                - float(
                    attacker_collision_penalty
                    if info.get("attacker_collision_penalty_applied", False)
                    else 0.0
                )
            ),
        }
        info["attacker_win"] = bool(target_success)
        info["attacker_target_success"] = bool(target_success)
        info["attacker_defender_collision_forfeit"] = bool(defender_collision_forfeit)
        info["attacker_target_progress_metric"] = self.target_progress_metric
        info["attacker_target_path_distance"] = float(curr_target_dist)
        return float(reward), terms

    def reset(self, seed=None, options=None):
        self._attacker_collision_active = False
        self._attacker_collision_events = 0
        reset_options = dict(options or {})
        curriculum_options = self._sample_curriculum_options(seed=seed)
        for key, value in curriculum_options.items():
            reset_options.setdefault(key, value)
        obs, info = self.env.reset(seed=seed, options=reset_options or None)
        self.current_defender_strategy = self._sample_defender_strategy()
        self.defender_policy = self._build_defender_policy(self.current_defender_strategy)
        if hasattr(self.defender_policy, "reset"):
            self.defender_policy.reset()
        self.current_obs = np.asarray(obs[1], dtype=np.float32).copy()
        info = dict(info or {})
        info["defender_strategy"] = self.current_defender_strategy
        info["curriculum_bin"] = self.current_curriculum_bin
        info["obstacle_density"] = str(
            getattr(map_config, "current_obstacle_density", "unknown")
        )
        state = self.env.get_privileged_state()
        reach_radius = float(
            getattr(map_config, "target_radius", 16.0)
            + getattr(map_config, "agent_radius", 8.0)
        )
        info["attacker_target_path_distance"] = float(
            self._target_path_distance(state, reach_radius)
        )
        return self.current_obs.copy(), info

    def step(self, action):
        prev_state = self.env.get_privileged_state()
        defender_action = self._get_defender_action()
        obs, _defender_reward, terminated, truncated, info = self.env.step(
            action=defender_action,
            attacker_action=action,
        )
        next_state = self.env.get_privileged_state()
        attacker_obs = np.asarray(obs[1], dtype=np.float32).copy()
        self.current_obs = attacker_obs

        info = dict(info or {})
        collision_event, collision_penalty_applied = self._update_attacker_collision_event(
            info.get("attacker_collision", False)
        )
        info["attacker_collision_event"] = collision_event
        info["attacker_collision_event_count"] = int(self._attacker_collision_events)
        info["attacker_collision_penalty_applied"] = collision_penalty_applied
        reward, terms = self._compute_attacker_reward(prev_state, next_state, info)
        info["attacker_reward"] = float(reward)
        info["attacker_reward_terms"] = terms
        info["defender_strategy"] = self.current_defender_strategy
        info["curriculum_bin"] = self.current_curriculum_bin
        return attacker_obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
