import os
import sys
import math
import copy
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from configs import map_config
from envs import env_lib
from envs import rewards as reward_fn
from configs.map_config import EnvParameters
from utils.path_risk import astar_path_length

_ORIGINAL_DEFENDER_SPEED = float(getattr(map_config, 'defender_speed', 2.6))
_ORIGINAL_ATTACKER_SPEED = float(getattr(map_config, 'attacker_speed', 2.0))

# Initialize pygame in headless mode for Ray workers
# This prevents hanging when creating surfaces in distributed environments
if not pygame.get_init():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("DISPLAY", "")
    pygame.init()
    pygame.display.set_mode((1, 1))  # Minimal display surface


def _read_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, None)
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        print(f"[WARN] Invalid env {name}={raw!r}; fallback to {default}")
        return float(default)


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, None)
    if raw is None:
        return int(default)
    text = str(raw).strip()
    if not text:
        return int(default)
    try:
        return int(text)
    except (TypeError, ValueError):
        print(f"[WARN] Invalid env {name}={raw!r}; fallback to {default}")
        return int(default)


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if not text:
        return bool(default)
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    print(f"[WARN] Invalid env {name}={raw!r}; fallback to {default}")
    return bool(default)


def _read_env_optional_float(name: str):
    raw = os.environ.get(name, None)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        print(f"[WARN] Invalid env {name}={raw!r}; ignore fixed value")
        return None


def _read_env_float_tuple(name: str, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return tuple(float(v) for v in default)
    text = str(raw).strip()
    if not text:
        return tuple(float(v) for v in default)
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except (TypeError, ValueError):
            print(f"[WARN] Invalid env {name}={raw!r}; fallback to {default}")
            return tuple(float(v) for v in default)
    return tuple(values) if values else tuple(float(v) for v in default)


def _read_env_choice_tuple(name: str, default):
    raw = os.environ.get(name, None)
    if raw is None:
        return tuple(default)
    text = str(raw).strip()
    if not text:
        return tuple(default)
    values = tuple(token.strip().lower() for token in text.split(',') if token.strip())
    return values if values else tuple(default)


class TADEnv(gym.Env):
    Metadata = {'render_modes': ['rgb_array'], 'render_fps': 40}
    _ued_bin_stats = {}
    _ued_continuous_stats = {}

    def __init__(
        self,
        spawn_outside_fov=False,
        reward_mode='standard',
        hard_action_mask=False,
        emit_skill_rewards=False,
    ):
        super().__init__()
        self.spawn_outside_fov = bool(spawn_outside_fov)
        self.reward_mode = str(reward_mode).strip().lower()
        self.emit_skill_rewards = bool(emit_skill_rewards)
        self.mask_flag = getattr(map_config, 'mask_flag', False)
        self.hard_action_mask = bool(hard_action_mask)
        self.defender_hard_action_mask = bool(hard_action_mask)
        self.attacker_hard_action_mask = bool(hard_action_mask)
        self.hard_mask_obstacle_margin = float(max(0.0, _read_env_float('TAD_HARD_MASK_MARGIN', 0.0)))
        self.hard_mask_radius_scale = float(max(0.0, _read_env_float('TAD_HARD_MASK_RADIUS_SCALE', 1.0)))
        self.hard_mask_lookahead_steps = int(max(1, _read_env_int('TAD_HARD_MASK_LOOKAHEAD', 1)))
        self.hard_mask_speed_cost_weight = float(max(0.0, _read_env_float('TAD_HARD_MASK_SPEED_COST_WEIGHT', 0.35)))
        self.hard_mask_max_deviation_cost = _read_env_float('TAD_HARD_MASK_MAX_DEVIATION_COST', -1.0)
        if float(self.hard_mask_max_deviation_cost) < 0.0:
            self.hard_mask_max_deviation_cost = None
        else:
            self.hard_mask_max_deviation_cost = float(self.hard_mask_max_deviation_cost)
        self.hard_mask_allow_zero_fallback = bool(_read_env_bool('TAD_HARD_MASK_ALLOW_ZERO_FALLBACK', True))
        self.hard_mask_speed_scales = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
        # Absolute turn fractions, not offsets from the requested turn.  This
        # keeps the opposite steering direction available even when the actor
        # saturates at either turn limit.
        self.hard_mask_angle_fracs = (0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, -1.0, 1.0)
        self.hard_mask_clearance_extras = (0.0, 4.0, 8.0)
        self.hard_mask_recovery_window = 20
        self.hard_mask_recovery_trigger_rate = 0.8
        self.hard_mask_recovery_min_progress = 2.0
        self.hard_mask_recovery_steps = 8
        self.hard_mask_recovery_speed_fraction = 0.3
        self._last_hard_mask_diagnostics = {}
        self._hard_mask_outcome_history = {
            'defender': deque(maxlen=self.hard_mask_recovery_window),
            'attacker': deque(maxlen=self.hard_mask_recovery_window),
        }
        self._hard_mask_recovery_remaining = {'defender': 0, 'attacker': 0}
        self._hard_mask_recovery_turn_sign = {'defender': 0.0, 'attacker': 0.0}
        self.width = map_config.width
        self.height = map_config.height
        self.pixel_size = map_config.pixel_size
        self.attacker_speed = map_config.attacker_speed
        self.defender_speed = map_config.defender_speed
        self._base_attacker_speed = _read_env_float('TAD_REGIME_BASE_ATTACKER_SPEED', _ORIGINAL_ATTACKER_SPEED)
        self._base_defender_speed = _read_env_float('TAD_REGIME_BASE_DEFENDER_SPEED', _ORIGINAL_DEFENDER_SPEED)
        self.current_regime = 'default'
        self.current_regime_info = {}
        self.current_speed_regime = 'default'
        self.current_margin_regime = 'default'
        self.current_ued_key = None
        self._target_time_margin_range = None
        self._attacker_target_astar_distance_range = None
        self._curriculum_astar_grid_size = 8.0
        self._curriculum_astar_obstacle_padding = 10.0
        self._curriculum_sampling_info = {}

        self.defender = None
        self.attacker = None
        self.target = None
        self._render_surface = None
        self.defender_trajectory = []
        self.attacker_trajectory = []
        self.defender_start_pos = None
        self.attacker_start_pos = None
        self.step_count = 0
        self.prev_defender_pos = None
        self.last_defender_pos = None
        self.prev_attacker_pos = None
        self.last_attacker_pos = None

        self.fov_angle = EnvParameters.FOV_ANGLE
        self.fov_range = EnvParameters.FOV_RANGE
        self.radar_rays = EnvParameters.RADAR_RAYS

        self.capture_radius = float(getattr(map_config, 'capture_radius', 20.0))
        self.capture_sector_angle_deg = float(getattr(map_config, 'capture_sector_angle_deg', 30.0))
        self.capture_required_steps = int(getattr(map_config, 'capture_required_steps', 1))
        self._capture_counter_defender = 0
        self._capture_counter_attacker = 0

        self.last_observed_attacker_pos = None
        self.steps_since_observed = 0
        self._best_distance_attacker = None
        self._best_distance_target = None

        obs_dim = 5 + 64 + 2  # 71维: attacker_info(5) + radar(64) + target_info(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.current_obs = None
        self._fov_cache = None
        self._fov_cache_valid = False

    @staticmethod
    def _sample_range(rng, bounds, fallback):
        values = tuple(float(v) for v in (bounds if bounds is not None else fallback))
        if len(values) < 2:
            return float(values[0]) if values else float(fallback[0])
        low, high = min(values[0], values[1]), max(values[0], values[1])
        return float(rng.uniform(low, high))

    def _select_regime_name(self, options=None) -> str:
        options = dict(options or {})
        explicit = options.get('regime', None)
        if explicit is None:
            explicit = os.environ.get('TAD_REGIME', None)
        mode = str(explicit).strip().lower() if explicit is not None else os.environ.get('TAD_REGIME_MODE', 'mixed').strip().lower()
        aliases = {
            'adv': 'advantage',
            'easy': 'advantage',
            'mid': 'neutral',
            'hard': 'disadvantage',
            'disadv': 'disadvantage',
            'ood': 'disadvantage',
        }
        mode = aliases.get(mode, mode)
        valid = ('advantage', 'neutral', 'disadvantage')
        if mode in valid:
            return mode

        regimes = _read_env_choice_tuple('TAD_REGIME_CHOICES', valid)
        regimes = tuple(r for r in regimes if r in valid) or valid
        probs = _read_env_float_tuple('TAD_REGIME_PROBS', (0.34, 0.33, 0.33))
        if len(probs) != len(regimes):
            probs = tuple(1.0 for _ in regimes)
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        probs = probs / max(float(probs.sum()), 1e-12)
        return str(self.np_random.choice(np.asarray(regimes, dtype=object), p=probs))

    @staticmethod
    def _regime_alias(name: str) -> str:
        text = str(name).strip().lower()
        aliases = {
            'adv': 'advantage',
            'easy': 'advantage',
            'def': 'advantage',
            'defender': 'advantage',
            'defender_adv': 'advantage',
            'defender_advantage': 'advantage',
            'mid': 'neutral',
            'equal': 'neutral',
            'hard': 'disadvantage',
            'disadv': 'disadvantage',
            'ood': 'disadvantage',
            'atk': 'disadvantage',
            'attacker': 'disadvantage',
            'attacker_adv': 'disadvantage',
            'attacker_advantage': 'disadvantage',
        }
        return aliases.get(text, text)

    def _select_decoupled_regime_pair(self, options=None):
        options = dict(options or {})
        valid = ('advantage', 'neutral', 'disadvantage')
        explicit_speed = options.get('speed_regime', None) or os.environ.get('TAD_SPEED_REGIME', None)
        explicit_margin = options.get('margin_regime', None) or os.environ.get('TAD_MARGIN_REGIME', None)
        if explicit_speed is not None and explicit_margin is not None:
            speed_regime = self._regime_alias(explicit_speed)
            margin_regime = self._regime_alias(explicit_margin)
            if speed_regime in valid and margin_regime in valid:
                return speed_regime, margin_regime

        choices = []
        for speed_regime in _read_env_choice_tuple('TAD_SPEED_REGIME_CHOICES', valid):
            speed_regime = self._regime_alias(speed_regime)
            if speed_regime not in valid:
                continue
            for margin_regime in _read_env_choice_tuple('TAD_MARGIN_REGIME_CHOICES', valid):
                margin_regime = self._regime_alias(margin_regime)
                if margin_regime in valid:
                    choices.append((speed_regime, margin_regime))
        choices = choices or [(s, m) for s in valid for m in valid]

        if _read_env_bool('TAD_UED_SELF_PACED_ENABLE', False):
            easy_rate = float(np.clip(_read_env_float('TAD_UED_EASY_CHASE_RATE', 0.18), 0.0, 1.0))
            if self.np_random.random() < easy_rate:
                return 'advantage', 'advantage'
            weights = np.asarray([self._ued_bin_weight(pair) for pair in choices], dtype=np.float64)
        else:
            speed_probs = _read_env_float_tuple('TAD_SPEED_REGIME_PROBS', (0.34, 0.33, 0.33))
            margin_probs = _read_env_float_tuple('TAD_MARGIN_REGIME_PROBS', (0.34, 0.33, 0.33))
            speed_map = self._prob_map(_read_env_choice_tuple('TAD_SPEED_REGIME_CHOICES', valid), speed_probs, valid)
            margin_map = self._prob_map(_read_env_choice_tuple('TAD_MARGIN_REGIME_CHOICES', valid), margin_probs, valid)
            weights = np.asarray([
                float(speed_map.get(s, 1.0)) * float(margin_map.get(m, 1.0))
                for s, m in choices
            ], dtype=np.float64)

        weights = np.maximum(weights, 1e-6)
        weights = weights / max(float(weights.sum()), 1e-12)
        idx = int(self.np_random.choice(np.arange(len(choices)), p=weights))
        return choices[idx]

    @staticmethod
    def _prob_map(names, probs, default_names):
        names = tuple(TADEnv._regime_alias(n) for n in names)
        names = tuple(n for n in names if n in default_names) or tuple(default_names)
        if len(probs) != len(names):
            probs = tuple(1.0 for _ in names)
        arr = np.asarray(probs, dtype=np.float64)
        arr = np.maximum(arr, 0.0)
        arr = arr / max(float(arr.sum()), 1e-12)
        return {name: float(arr[i]) for i, name in enumerate(names)}

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return float(a) + (float(b) - float(a)) * t

    @staticmethod
    def _difficulty_label(value: float) -> str:
        value = float(np.clip(value, 0.0, 1.0))
        if value < 1.0 / 3.0:
            return 'advantage'
        if value < 2.0 / 3.0:
            return 'neutral'
        return 'disadvantage'

    @classmethod
    def _ued_stats_weight(cls, stats):
        count = int(stats.get('count', 0))
        wins = float(stats.get('wins', 0.0))
        collisions = float(stats.get('collisions', 0.0))
        captures = float(stats.get('captures', 0.0))
        if count <= 0:
            return 2.0
        win_rate = wins / max(1.0, float(count))
        collision_rate = collisions / max(1.0, float(count))
        capture_rate = captures / max(1.0, float(count))
        target = float(np.clip(_read_env_float('TAD_UED_TARGET_WIN_RATE', 0.78), 0.05, 0.98))
        width = max(0.05, _read_env_float('TAD_UED_TARGET_WIDTH', 0.18))
        just_right = math.exp(-((win_rate - target) / width) ** 2)
        novelty = 1.0 / math.sqrt(float(count) + 1.0)
        safe_capture = 0.35 * capture_rate
        collision_penalty = 0.45 * collision_rate
        floor = _read_env_float('TAD_UED_WEIGHT_FLOOR', 0.10)
        return max(float(floor), 0.65 * just_right + 0.35 * novelty + safe_capture - collision_penalty)

    def _sample_continuous_difficulties(self):
        explicit_speed = _read_env_optional_float('TAD_SPEED_DIFFICULTY')
        explicit_margin = _read_env_optional_float('TAD_MARGIN_DIFFICULTY')
        if explicit_speed is not None and explicit_margin is not None:
            return float(np.clip(explicit_speed, 0.0, 1.0)), float(np.clip(explicit_margin, 0.0, 1.0)), None

        if _read_env_bool('TAD_UED_SELF_PACED_ENABLE', False):
            easy_rate = float(np.clip(_read_env_float('TAD_UED_EASY_CHASE_RATE', 0.16), 0.0, 1.0))
            if self.np_random.random() < easy_rate:
                center = float(np.clip(_read_env_float('TAD_UED_CONTINUOUS_EASY_CENTER', 0.16), 0.0, 1.0))
                jitter = max(0.0, _read_env_float('TAD_UED_CONTINUOUS_EASY_JITTER', 0.06))
                speed_d = float(np.clip(self.np_random.normal(center, jitter), 0.0, 1.0))
                margin_d = float(np.clip(self.np_random.normal(center, jitter), 0.0, 1.0))
                return speed_d, margin_d, ('continuous', int(speed_d * 100), int(margin_d * 100))

            grid = int(max(2, _read_env_int('TAD_UED_CONTINUOUS_GRID', 6)))
            choices = [(i, j) for i in range(grid) for j in range(grid)]
            weights = np.asarray(
                [self._ued_stats_weight(self._ued_continuous_stats.get((i, j), {})) for i, j in choices],
                dtype=np.float64,
            )
            weights = np.maximum(weights, 1e-6)
            weights = weights / max(float(weights.sum()), 1e-12)
            idx = int(self.np_random.choice(np.arange(len(choices)), p=weights))
            i, j = choices[idx]
            jitter = max(0.0, _read_env_float('TAD_UED_CONTINUOUS_JITTER', 0.10))
            speed_d = float(np.clip((i + 0.5) / grid + self.np_random.normal(0.0, jitter), 0.0, 1.0))
            margin_d = float(np.clip((j + 0.5) / grid + self.np_random.normal(0.0, jitter), 0.0, 1.0))
            cell = (
                int(np.clip(math.floor(speed_d * grid), 0, grid - 1)),
                int(np.clip(math.floor(margin_d * grid), 0, grid - 1)),
            )
            return speed_d, margin_d, cell

        speed_alpha, speed_beta = _read_env_float_tuple('TAD_CONTINUOUS_SPEED_BETA', (1.25, 2.35))
        margin_alpha, margin_beta = _read_env_float_tuple('TAD_CONTINUOUS_MARGIN_BETA', (1.15, 2.65))
        speed_d = float(self.np_random.beta(max(0.05, speed_alpha), max(0.05, speed_beta)))
        margin_d = float(self.np_random.beta(max(0.05, margin_alpha), max(0.05, margin_beta)))
        return speed_d, margin_d, None

    def _apply_continuous_episode_regime(self):
        speed_d, margin_d, ued_key = self._sample_continuous_difficulties()
        speed_regime = self._difficulty_label(speed_d)
        margin_regime = self._difficulty_label(margin_d)

        easy_def = _read_env_float_tuple('TAD_CONT_SPEED_EASY_DEFENDER_MULT', (1.06, 1.22))
        hard_def = _read_env_float_tuple('TAD_CONT_SPEED_HARD_DEFENDER_MULT', (0.96, 1.08))
        easy_att = _read_env_float_tuple('TAD_CONT_SPEED_EASY_ATTACKER_MULT', (0.82, 0.96))
        hard_att = _read_env_float_tuple('TAD_CONT_SPEED_HARD_ATTACKER_MULT', (0.96, 1.12))
        easy_margin = _read_env_float_tuple('TAD_CONT_MARGIN_EASY', (45.0, 160.0))
        hard_margin = _read_env_float_tuple('TAD_CONT_MARGIN_HARD', (-18.0, 48.0))

        def _range_pair(values, fallback):
            if len(values) < 2:
                values = fallback
            return min(float(values[0]), float(values[1])), max(float(values[0]), float(values[1]))

        easy_def = _range_pair(easy_def, (1.06, 1.22))
        hard_def = _range_pair(hard_def, (0.96, 1.08))
        easy_att = _range_pair(easy_att, (0.82, 0.96))
        hard_att = _range_pair(hard_att, (0.96, 1.12))
        easy_margin = _range_pair(easy_margin, (45.0, 160.0))
        hard_margin = _range_pair(hard_margin, (-18.0, 48.0))

        defender_low = self._lerp(easy_def[0], hard_def[0], speed_d)
        defender_high = self._lerp(easy_def[1], hard_def[1], speed_d)
        attacker_low = self._lerp(easy_att[0], hard_att[0], speed_d)
        attacker_high = self._lerp(easy_att[1], hard_att[1], speed_d)
        margin_low = self._lerp(easy_margin[0], hard_margin[0], margin_d)
        margin_high = self._lerp(easy_margin[1], hard_margin[1], margin_d)

        defender_mult = self._sample_range(self.np_random, (defender_low, defender_high), (defender_low, defender_high))
        attacker_mult = self._sample_range(self.np_random, (attacker_low, attacker_high), (attacker_low, attacker_high))
        margin_low, margin_high = min(margin_low, margin_high), max(margin_low, margin_high)

        map_config.defender_speed = float(self._base_defender_speed * defender_mult)
        map_config.attacker_speed = float(self._base_attacker_speed * attacker_mult)
        self.defender_speed = float(map_config.defender_speed)
        self.attacker_speed = float(map_config.attacker_speed)
        self._target_time_margin_range = (float(margin_low), float(margin_high))
        self.current_speed_regime = speed_regime
        self.current_margin_regime = margin_regime
        self.current_regime = f"continuous_s{speed_d:.2f}_m{margin_d:.2f}"
        self.current_ued_key = ued_key
        self.current_regime_info = {
            'regime': self.current_regime,
            'speed_regime': speed_regime,
            'margin_regime': margin_regime,
            'regime_continuous': True,
            'speed_difficulty': float(speed_d),
            'margin_difficulty': float(margin_d),
            'defender_speed': float(self.defender_speed),
            'attacker_speed': float(self.attacker_speed),
            'defender_speed_mult': float(defender_mult),
            'attacker_speed_mult': float(attacker_mult),
            'target_time_margin_low': float(margin_low),
            'target_time_margin_high': float(margin_high),
            'ued_key': list(ued_key) if ued_key is not None else None,
        }

    @classmethod
    def _ued_bin_weight(cls, pair):
        return cls._ued_stats_weight(cls._ued_bin_stats.get(tuple(pair), {}))

    @classmethod
    def _update_ued_bin_stats(cls, pair, info):
        if not _read_env_bool('TAD_UED_SELF_PACED_ENABLE', False):
            return
        if pair is None or not isinstance(info, dict):
            return
        key = tuple(pair)
        stats = cls._ued_bin_stats.setdefault(key, {'count': 0, 'wins': 0.0, 'collisions': 0.0, 'captures': 0.0})
        reason = str(info.get('reason', '')).strip().lower()
        win = bool(info.get('win', False))
        collision = bool(info.get('defender_collision', False)) or 'defender_collision' in reason or 'defender_out' in reason
        capture = 'defender_caught_attacker' in reason
        stats['count'] = int(stats.get('count', 0)) + 1
        stats['wins'] = float(stats.get('wins', 0.0)) + (1.0 if win else 0.0)
        stats['collisions'] = float(stats.get('collisions', 0.0)) + (1.0 if collision else 0.0)
        stats['captures'] = float(stats.get('captures', 0.0)) + (1.0 if capture else 0.0)

    @classmethod
    def _update_ued_continuous_stats(cls, key, info):
        if not _read_env_bool('TAD_UED_SELF_PACED_ENABLE', False):
            return
        if key is None or not isinstance(info, dict):
            return
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            return
        key = (int(key[0]), int(key[1]))
        stats = cls._ued_continuous_stats.setdefault(
            key,
            {'count': 0, 'wins': 0.0, 'collisions': 0.0, 'captures': 0.0},
        )
        reason = str(info.get('reason', '')).strip().lower()
        win = bool(info.get('win', False))
        collision = bool(info.get('defender_collision', False)) or 'defender_collision' in reason or 'defender_out' in reason
        capture = 'defender_caught_attacker' in reason
        stats['count'] = int(stats.get('count', 0)) + 1
        stats['wins'] = float(stats.get('wins', 0.0)) + (1.0 if win else 0.0)
        stats['collisions'] = float(stats.get('collisions', 0.0)) + (1.0 if collision else 0.0)
        stats['captures'] = float(stats.get('captures', 0.0)) + (1.0 if capture else 0.0)

    def _apply_episode_regime(self, options=None):
        """Optionally randomize speed and target time-advantage regime.

        This is disabled by default and only activates when
        TAD_REGIME_RANDOMIZATION=1 or a launcher sets it explicitly.
        """
        self.current_regime = 'default'
        self.current_regime_info = {}
        self.current_speed_regime = 'default'
        self.current_margin_regime = 'default'
        self.current_ued_key = None
        self._target_time_margin_range = None
        self._attacker_target_astar_distance_range = None

        options = dict(options or {})
        explicit_margin = options.get('time_margin_range', None)
        explicit_astar_distance = options.get('attacker_target_astar_distance_range', None)
        explicit_attacker_speed = options.get('attacker_speed', None)
        explicit_defender_speed = options.get('defender_speed', None)
        self._curriculum_astar_grid_size = float(
            options.get('curriculum_astar_grid_size', self._curriculum_astar_grid_size)
        )
        self._curriculum_astar_obstacle_padding = float(
            options.get('curriculum_astar_obstacle_padding', self._curriculum_astar_obstacle_padding)
        )
        if self._curriculum_astar_grid_size <= 0.0:
            raise ValueError('curriculum_astar_grid_size must be positive')
        if self._curriculum_astar_obstacle_padding < 0.0:
            raise ValueError('curriculum_astar_obstacle_padding must be non-negative')
        if (
            explicit_margin is not None
            or explicit_astar_distance is not None
            or explicit_attacker_speed is not None
            or explicit_defender_speed is not None
        ):
            margin_values = tuple(float(v) for v in (explicit_margin or ()))
            if margin_values and len(margin_values) < 2:
                margin_values = (margin_values[0], margin_values[0])
            if margin_values:
                self._target_time_margin_range = (
                    min(margin_values[0], margin_values[1]),
                    max(margin_values[0], margin_values[1]),
                )
            astar_values = tuple(float(v) for v in (explicit_astar_distance or ()))
            if astar_values and len(astar_values) < 2:
                astar_values = (astar_values[0], astar_values[0])
            if astar_values:
                self._attacker_target_astar_distance_range = (
                    min(astar_values[0], astar_values[1]),
                    max(astar_values[0], astar_values[1]),
                )
            map_config.defender_speed = float(
                self._base_defender_speed if explicit_defender_speed is None else explicit_defender_speed
            )
            map_config.attacker_speed = float(
                self._base_attacker_speed if explicit_attacker_speed is None else explicit_attacker_speed
            )
            self.defender_speed = float(map_config.defender_speed)
            self.attacker_speed = float(map_config.attacker_speed)
            label = str(options.get('regime', options.get('curriculum_bin', 'custom')))
            self.current_speed_regime = 'fixed'
            self.current_margin_regime = label
            self.current_regime = label
            self.current_regime_info = {
                'regime': label,
                'speed_regime': 'fixed',
                'margin_regime': label,
                'regime_decoupled': True,
                'defender_speed': self.defender_speed,
                'attacker_speed': self.attacker_speed,
                'target_time_margin_low': (
                    self._target_time_margin_range[0] if self._target_time_margin_range is not None else None
                ),
                'target_time_margin_high': (
                    self._target_time_margin_range[1] if self._target_time_margin_range is not None else None
                ),
                'attacker_target_astar_distance_low': (
                    self._attacker_target_astar_distance_range[0]
                    if self._attacker_target_astar_distance_range is not None else None
                ),
                'attacker_target_astar_distance_high': (
                    self._attacker_target_astar_distance_range[1]
                    if self._attacker_target_astar_distance_range is not None else None
                ),
            }
            return

        if not _read_env_bool('TAD_REGIME_RANDOMIZATION', False):
            map_config.defender_speed = float(self._base_defender_speed)
            map_config.attacker_speed = float(self._base_attacker_speed)
            self.defender_speed = float(map_config.defender_speed)
            self.attacker_speed = float(map_config.attacker_speed)
            return

        if _read_env_bool('TAD_REGIME_CONTINUOUS', False):
            self._apply_continuous_episode_regime()
            return

        use_decoupled = _read_env_bool('TAD_REGIME_DECOUPLED', False)
        if use_decoupled:
            speed_regime, margin_regime = self._select_decoupled_regime_pair(options=options)
        else:
            speed_regime = margin_regime = self._select_regime_name(options=options)

        speed_defaults = {
            'advantage': {
                'defender_mult': (1.02, 1.22),
                'attacker_mult': (0.82, 1.00),
            },
            'neutral': {
                'defender_mult': (0.95, 1.08),
                'attacker_mult': (0.94, 1.10),
            },
            'disadvantage': {
                'defender_mult': (0.82, 1.00),
                'attacker_mult': (1.05, 1.28),
            },
        }
        margin_defaults = {
            'advantage': {
                'margin': (30.0, 180.0),
            },
            'neutral': {
                'margin': (-20.0, 100.0),
            },
            'disadvantage': {
                'margin': (-80.0, 45.0),
            },
        }
        if not use_decoupled:
            speed_defaults = {
                'advantage': {'defender_mult': (1.00, 1.12), 'attacker_mult': (0.90, 1.00)},
                'neutral': {'defender_mult': (0.96, 1.06), 'attacker_mult': (0.96, 1.10)},
                'disadvantage': {'defender_mult': (0.92, 1.02), 'attacker_mult': (1.02, 1.16)},
            }
            margin_defaults = {
                'advantage': {'margin': (20.0, 140.0)},
                'neutral': {'margin': (-10.0, 80.0)},
                'disadvantage': {'margin': (-30.0, 50.0)},
            }

        speed_cfg = speed_defaults[speed_regime]
        margin_cfg = margin_defaults[margin_regime]
        speed_prefix = f"TAD_SPEED_REGIME_{speed_regime.upper()}" if use_decoupled else f"TAD_REGIME_{speed_regime.upper()}"
        margin_prefix = f"TAD_MARGIN_REGIME_{margin_regime.upper()}" if use_decoupled else f"TAD_REGIME_{margin_regime.upper()}"
        defender_mult = self._sample_range(
            self.np_random,
            _read_env_float_tuple(f'{speed_prefix}_DEFENDER_SPEED_MULT', speed_cfg['defender_mult']),
            speed_cfg['defender_mult'],
        )
        attacker_mult = self._sample_range(
            self.np_random,
            _read_env_float_tuple(f'{speed_prefix}_ATTACKER_SPEED_MULT', speed_cfg['attacker_mult']),
            speed_cfg['attacker_mult'],
        )
        margin_range = _read_env_float_tuple(f'{margin_prefix}_TIME_MARGIN', margin_cfg['margin'])
        if len(margin_range) < 2:
            margin_range = margin_cfg['margin']
        margin_low = min(float(margin_range[0]), float(margin_range[1]))
        margin_high = max(float(margin_range[0]), float(margin_range[1]))

        map_config.defender_speed = float(self._base_defender_speed * defender_mult)
        map_config.attacker_speed = float(self._base_attacker_speed * attacker_mult)
        self.defender_speed = float(map_config.defender_speed)
        self.attacker_speed = float(map_config.attacker_speed)
        self._target_time_margin_range = (float(margin_low), float(margin_high))
        self.current_speed_regime = speed_regime
        self.current_margin_regime = margin_regime
        self.current_regime = margin_regime if not use_decoupled else f"{speed_regime}_speed__{margin_regime}_margin"
        self.current_regime_info = {
            'regime': self.current_regime,
            'speed_regime': speed_regime,
            'margin_regime': margin_regime,
            'regime_decoupled': bool(use_decoupled),
            'defender_speed': float(self.defender_speed),
            'attacker_speed': float(self.attacker_speed),
            'defender_speed_mult': float(defender_mult),
            'attacker_speed_mult': float(attacker_mult),
            'target_time_margin_low': float(margin_low),
            'target_time_margin_high': float(margin_high),
        }

    def _augment_info_with_regime(self, info):
        if info is None:
            info = {}
        info['regime'] = str(getattr(self, 'current_regime', 'default'))
        info['speed_regime'] = str(getattr(self, 'current_speed_regime', 'default'))
        info['margin_regime'] = str(getattr(self, 'current_margin_regime', 'default'))
        sampling_info = dict(getattr(self, '_curriculum_sampling_info', {}) or {})
        regime_info = dict(getattr(self, 'current_regime_info', {}) or {})
        regime_info.update(sampling_info)
        info['regime_info'] = regime_info
        info.update(sampling_info)
        info['obstacle_density'] = str(
            getattr(map_config, 'current_obstacle_density', 'unknown')
        )
        return info

    def _regime_terminal_bonus(self, info):
        """Optional terminal-only reward for regime-adaptive end-to-end RL."""
        if not _read_env_bool('TAD_REGIME_TERMINAL_REWARD_ENABLE', False):
            return 0.0
        if not isinstance(info, dict):
            return 0.0

        reason = str(info.get('reason', '')).strip().lower()
        if not reason:
            return 0.0

        regime = str(getattr(self, 'current_margin_regime', getattr(self, 'current_regime', 'default'))).strip().lower()
        defaults = {
            'advantage': {
                'capture': 8.0,
                'timeout': 0.0,
                'attacker': -8.0,
                'collision': -8.0,
            },
            'neutral': {
                'capture': 4.0,
                'timeout': 3.0,
                'attacker': -10.0,
                'collision': -8.0,
            },
            'disadvantage': {
                'capture': 2.0,
                'timeout': 10.0,
                'attacker': -14.0,
                'collision': -10.0,
            },
        }
        cfg = defaults.get(regime, defaults['neutral'])
        prefix = f"TAD_REGIME_TERMINAL_{regime.upper()}"

        if 'defender_caught_attacker' in reason:
            key = 'capture'
        elif 'timeout_defender_wins' in reason:
            key = 'timeout'
        elif 'attacker_caught_target' in reason or 'attacker_win' in reason:
            key = 'attacker'
        elif 'defender_collision' in reason or 'defender_out' in reason:
            key = 'collision'
        else:
            return 0.0

        bonus = _read_env_float(f'{prefix}_{key.upper()}_BONUS', cfg[key])
        info['regime_terminal_bonus'] = float(bonus)
        info['regime_terminal_bonus_key'] = key
        return float(bonus)

    @staticmethod
    def _copy_obs(obs):
        if obs is None:
            return None
        if isinstance(obs, tuple):
            return tuple(TADEnv._copy_obs(x) for x in obs)
        if isinstance(obs, list):
            return [TADEnv._copy_obs(x) for x in obs]
        if isinstance(obs, np.ndarray):
            return obs.copy()
        return copy.deepcopy(obs)

    def snapshot_state(self):
        """Capture mutable simulator state for short counterfactual rollouts."""
        rng_state = None
        if hasattr(self, "np_random") and self.np_random is not None:
            rng_state = copy.deepcopy(self.np_random.bit_generator.state)
        return {
            'defender': copy.deepcopy(self.defender),
            'attacker': copy.deepcopy(self.attacker),
            'target': copy.deepcopy(self.target),
            'defender_trajectory': copy.deepcopy(self.defender_trajectory),
            'attacker_trajectory': copy.deepcopy(self.attacker_trajectory),
            'defender_start_pos': copy.deepcopy(self.defender_start_pos),
            'attacker_start_pos': copy.deepcopy(self.attacker_start_pos),
            'step_count': int(self.step_count),
            'prev_defender_pos': copy.deepcopy(self.prev_defender_pos),
            'last_defender_pos': copy.deepcopy(self.last_defender_pos),
            'prev_attacker_pos': copy.deepcopy(self.prev_attacker_pos),
            'last_attacker_pos': copy.deepcopy(self.last_attacker_pos),
            'capture_counter_defender': int(self._capture_counter_defender),
            'capture_counter_attacker': int(self._capture_counter_attacker),
            'last_observed_attacker_pos': copy.deepcopy(self.last_observed_attacker_pos),
            'steps_since_observed': int(self.steps_since_observed),
            'best_distance_attacker': copy.deepcopy(self._best_distance_attacker),
            'best_distance_target': copy.deepcopy(self._best_distance_target),
            'initial_dist_def_tgt': copy.deepcopy(getattr(self, 'initial_dist_def_tgt', None)),
            'initial_dist_def_att': copy.deepcopy(getattr(self, 'initial_dist_def_att', None)),
            'current_obs': self._copy_obs(self.current_obs),
            'fov_cache': self._copy_obs(self._fov_cache),
            'fov_cache_valid': bool(self._fov_cache_valid),
            'rng_state': rng_state,
        }

    def restore_state(self, state):
        """Restore a state captured by snapshot_state()."""
        self.defender = copy.deepcopy(state['defender'])
        self.attacker = copy.deepcopy(state['attacker'])
        self.target = copy.deepcopy(state['target'])
        self.defender_trajectory = copy.deepcopy(state['defender_trajectory'])
        self.attacker_trajectory = copy.deepcopy(state['attacker_trajectory'])
        self.defender_start_pos = copy.deepcopy(state['defender_start_pos'])
        self.attacker_start_pos = copy.deepcopy(state['attacker_start_pos'])
        self.step_count = int(state['step_count'])
        self.prev_defender_pos = copy.deepcopy(state['prev_defender_pos'])
        self.last_defender_pos = copy.deepcopy(state['last_defender_pos'])
        self.prev_attacker_pos = copy.deepcopy(state['prev_attacker_pos'])
        self.last_attacker_pos = copy.deepcopy(state['last_attacker_pos'])
        self._capture_counter_defender = int(state['capture_counter_defender'])
        self._capture_counter_attacker = int(state['capture_counter_attacker'])
        self.last_observed_attacker_pos = copy.deepcopy(state['last_observed_attacker_pos'])
        self.steps_since_observed = int(state['steps_since_observed'])
        self._best_distance_attacker = copy.deepcopy(state['best_distance_attacker'])
        self._best_distance_target = copy.deepcopy(state['best_distance_target'])
        if state.get('initial_dist_def_tgt') is not None:
            self.initial_dist_def_tgt = float(state['initial_dist_def_tgt'])
        if state.get('initial_dist_def_att') is not None:
            self.initial_dist_def_att = float(state['initial_dist_def_att'])
        self.current_obs = self._copy_obs(state['current_obs'])
        self._fov_cache = self._copy_obs(state['fov_cache'])
        self._fov_cache_valid = bool(state['fov_cache_valid'])
        rng_state = state.get('rng_state')
        if rng_state is not None and hasattr(self, "np_random") and self.np_random is not None:
            self.np_random.bit_generator.state = copy.deepcopy(rng_state)

    def _compute_skill_reward_components(
        self,
        defender_blocked: bool,
        attacker_blocked: bool,
        defender_radar,
        prev_defender_radar,
    ):
        common = dict(
            defender=self.defender,
            attacker=self.attacker,
            target=self.target,
            prev_defender=self.prev_defender_pos,
            prev_attacker=self.prev_attacker_pos,
            defender_collision=bool(defender_blocked),
            attacker_collision=bool(attacker_blocked),
            defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
            attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
            capture_progress_defender=int(self._capture_counter_defender),
            capture_progress_attacker=int(self._capture_counter_attacker),
            capture_required_steps=int(self.capture_required_steps),
        )

        rewards = {}
        rewards['chase'] = float(reward_fn.reward_calculate_chase(
            **common,
            radar=defender_radar,
            prev_radar=prev_defender_radar,
            initial_dist_def_att=self.initial_dist_def_att,
        )[0])
        rewards['protect'] = float(reward_fn.reward_calculate_protect(
            **common,
            radar=defender_radar,
            initial_dist_def_tgt=self.initial_dist_def_tgt,
            initial_dist_def_att=self.initial_dist_def_att,
        )[0])
        return rewards

    def _get_obs_features(self):
        defender_obs = self._get_defender_observation()
        attacker_obs = self._get_attacker_observation()
        return defender_obs, attacker_obs

    def get_normalized_attacker_info(self):
        """
        Get normalized attacker relative position and visibility flag for GRU training.

        Returns:
            rel_x_norm: Normalized relative x in [0, 1]
            rel_y_norm: Normalized relative y in [0, 1]
            is_visible: Boolean indicating if attacker is visible (in FOV and not occluded)
        """
        true_rel_vec, true_dist = self._get_relative_position(self.defender, self.attacker)
        absolute_angle = math.atan2(true_rel_vec[1], true_rel_vec[0])
        true_rel_angle_deg = self._normalize_angle(math.degrees(absolute_angle) - self.defender['theta'])
        fov_half = self.fov_angle * 0.5

        in_fov, occluded = self._update_visibility(true_rel_angle_deg, true_dist, fov_half)
        is_visible = (in_fov > 0.5 and occluded < 0.5)

        # Calculate relative position in [0, 1] range
        map_diagonal = math.hypot(self.width, self.height)
        normalized_distance = np.clip((true_dist / map_diagonal) * 2.0 - 1.0, -1.0, 1.0)

        abs_ang = math.atan2(true_rel_vec[1], true_rel_vec[0])
        rel_ang = self._normalize_angle(math.degrees(abs_ang) - self.defender['theta'])

        # Convert to normalized [0, 1] coordinates
        # Map [-1, 1] to [0, 1]
        rel_x_norm = (normalized_distance + 1.0) / 2.0
        rel_y_norm = (rel_ang / 180.0 + 1.0) / 2.0

        return float(rel_x_norm), float(rel_y_norm), bool(is_visible)

    def _get_defender_observation(self, use_privileged=False, gru_prediction=None):
        """
        Get defender observation.
        
        观测结构 (71维):
          [0:5]  = attacker_info: [distance, bearing, fov_edge, is_visible, unobserved_time]
          [5:7]  = target_info: [distance, bearing]
          [7:71] = radar (64维)

        Args:
            use_privileged: If True, use true attacker position even when occluded
            gru_prediction: Optional tuple (pred_x_norm, pred_y_norm) from GRU in [0, 1] range
                          Used when attacker is occluded and use_privileged is False
        """
        obs = np.zeros(71, dtype=np.float32)

        true_rel_vec, true_dist = self._get_relative_position(self.defender, self.attacker)
        absolute_angle = math.atan2(true_rel_vec[1], true_rel_vec[0])
        true_rel_angle_deg = self._normalize_angle(math.degrees(absolute_angle) - self.defender['theta'])
        fov_half = self.fov_angle * 0.5

        in_fov, occluded = self._update_visibility(true_rel_angle_deg, true_dist, fov_half)

        obs_attacker_state = None
        is_visible = (in_fov > 0.5 and occluded < 0.5)

        if is_visible or use_privileged:
            obs_attacker_state = self.attacker
        elif gru_prediction is not None:
            # Use GRU prediction: convert normalized [0, 1] back to position
            pred_x_norm, pred_y_norm = gru_prediction
            # Convert [0, 1] back to [-1, 1]
            pred_dist_norm = pred_x_norm * 2.0 - 1.0
            pred_bearing_norm = pred_y_norm * 2.0 - 1.0
            # Convert [-1, 1] to actual values
            map_diagonal = math.hypot(self.width, self.height)
            pred_dist = ((pred_dist_norm + 1.0) / 2.0) * map_diagonal
            pred_bearing = pred_bearing_norm * 180.0
            # Calculate position from distance and bearing
            pred_abs_angle = math.radians(self._normalize_angle(pred_bearing + self.defender['theta']))
            pred_rel_x = pred_dist * math.cos(pred_abs_angle)
            pred_rel_y = pred_dist * math.sin(pred_abs_angle)
            # Create attacker state from prediction
            defender_center = np.array([self.defender['x'] + self.pixel_size * 0.5,
                                       self.defender['y'] + self.pixel_size * 0.5])
            pred_center = defender_center + np.array([pred_rel_x, pred_rel_y])
            obs_attacker_state = {
                'x': pred_center[0] - self.pixel_size * 0.5,
                'y': pred_center[1] - self.pixel_size * 0.5,
                'theta': 0.0
            }
        elif self.last_observed_attacker_pos is not None:
            obs_attacker_state = {
                'x': self.last_observed_attacker_pos[0] - self.pixel_size * 0.5,
                'y': self.last_observed_attacker_pos[1] - self.pixel_size * 0.5,
                'theta': 0.0
            }
        else:
            obs_attacker_state = None

        if obs_attacker_state is not None:
            rel_vec, distance = self._get_relative_position(self.defender, obs_attacker_state)
            map_diagonal = math.hypot(self.width, self.height)
            normalized_distance = np.clip((distance / map_diagonal) * 2.0 - 1.0, -1.0, 1.0)

            abs_ang = math.atan2(rel_vec[1], rel_vec[0])
            rel_ang = self._normalize_angle(math.degrees(abs_ang) - self.defender['theta'])
            normalized_bearing = np.clip(rel_ang / 180.0, -1.0, 1.0)

            fov_edge_angle = min(abs(rel_ang + fov_half), abs(rel_ang - fov_half))
            normalized_fov_edge = np.clip((fov_edge_angle / fov_half) * 2.0 - 1.0, -1.0, 1.0) if fov_half > 0 else 0.0
        else:
            normalized_distance = 1.0
            normalized_bearing = 0.0
            normalized_fov_edge = 1.0

        # attacker_info [0:5]
        obs[0] = normalized_distance
        obs[1] = normalized_bearing
        obs[2] = normalized_fov_edge
        obs[3] = 1.0 if is_visible else 0.0
        max_unobserved = float(EnvParameters.MAX_UNOBSERVED_STEPS)
        normalized_unobserved = np.clip((self.steps_since_observed / max_unobserved) * 2.0 - 1.0, -1.0, 1.0)
        obs[4] = normalized_unobserved

        # target_info [5:7]
        target_rel_vec, target_dist = self._get_relative_position(self.defender, self.target)
        target_map_diagonal = math.hypot(self.width, self.height)
        target_normalized_dist = np.clip((target_dist / target_map_diagonal) * 2.0 - 1.0, -1.0, 1.0)
        target_abs_ang = math.atan2(target_rel_vec[1], target_rel_vec[0])
        target_rel_ang = self._normalize_angle(math.degrees(target_abs_ang) - self.defender['theta'])
        target_normalized_bearing = np.clip(target_rel_ang / 180.0, -1.0, 1.0)
        obs[5] = target_normalized_dist
        obs[6] = target_normalized_bearing

        # radar [7:71]
        obs[7:71] = self._sense_agent_radar(self.defender, num_rays=self.radar_rays, full_circle=True)

        return obs

    def _get_velocity(self, agent, prev_pos):
        if prev_pos is not None:
            dx = agent['x'] - prev_pos['x']
            dy = agent['y'] - prev_pos['y']
            return np.array([dx, dy], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_angular_velocity(self, agent, prev_pos, max_ang_speed):
        if prev_pos is not None:
            prev_heading = prev_pos.get('theta', 0.0)
            angle_change = self._normalize_angle(agent['theta'] - prev_heading)
            return np.clip(angle_change / (max_ang_speed + 1e-6), -1.0, 1.0)
        return 0.0

    def _get_relative_position(self, from_agent, to_agent):
        from_center = np.array([from_agent['x'] + self.pixel_size * 0.5, from_agent['y'] + self.pixel_size * 0.5], dtype=np.float32)
        to_center = np.array([to_agent['x'] + self.pixel_size * 0.5, to_agent['y'] + self.pixel_size * 0.5], dtype=np.float32)
        relative_vec = to_center - from_center
        distance = float(np.linalg.norm(relative_vec))
        return relative_vec, distance

    def _update_visibility(self, relative_angle_deg, distance, fov_half_deg):
        in_angle = abs(relative_angle_deg) <= fov_half_deg
        in_range = True
        in_fov = 1.0 if (in_angle and in_range) else 0.0
        occluded = 0.0

        if in_fov > 0.5:
            if self._is_line_blocked(self.defender, self.attacker):
                occluded = 1.0
                self.steps_since_observed += 1
            else:
                attacker_center = np.array([self.attacker['x'] + self.pixel_size * 0.5, self.attacker['y'] + self.pixel_size * 0.5], dtype=np.float32)
                self.last_observed_attacker_pos = attacker_center.copy()
                self.steps_since_observed = 0
        else:
            self.steps_since_observed += 1

        return in_fov, occluded

    def _get_attacker_observation(self):
        """
        Get attacker observation (privileged, for Critic CTDE).
        
        观测结构 (72维):
          [0:8]  = 标量: [attacker_x, attacker_y, attacker_heading, defender_x, defender_y, defender_heading, target_x, target_y]
          [8:72] = radar (64维)
        """
        obs = np.zeros(72, dtype=np.float32)

        # 标量部分 [0:8]
        obs[0] = (self.attacker['x'] / self.width) * 2.0 - 1.0
        obs[1] = (self.attacker['y'] / self.height) * 2.0 - 1.0
        obs[2] = (self.attacker['theta'] / 180.0) - 1.0
        obs[3] = (self.defender['x'] / self.width) * 2.0 - 1.0
        obs[4] = (self.defender['y'] / self.height) * 2.0 - 1.0
        obs[5] = (self.defender['theta'] / 180.0) - 1.0  # 新增: defender朝向
        obs[6] = (self.target['x'] / self.width) * 2.0 - 1.0
        obs[7] = (self.target['y'] / self.height) * 2.0 - 1.0

        # 雷达部分 [8:72]
        obs[8:72] = self._sense_agent_radar(self.attacker, num_rays=self.radar_rays, full_circle=True)

        return obs

    def _normalize_angle(self, angle_deg):
        angle_deg = angle_deg % 360.0
        if angle_deg > 180.0:
            angle_deg -= 360.0
        return float(angle_deg)

    def _sense_agent_radar(self, agent, num_rays=10, full_circle=False):
        center = np.array([agent['x'] + self.pixel_size * 0.5, agent['y'] + self.pixel_size * 0.5], dtype=float)
        heading = math.radians(agent.get('theta', 0.0))
        if full_circle:
            angles = [heading + 2 * math.pi * i / num_rays for i in range(num_rays)]
        else:
            angle_range = math.pi
            angles = [heading + (i / (num_rays - 1) - 0.5) * angle_range for i in range(num_rays)]
        max_radar_range = float(min(EnvParameters.FOV_RANGE, math.hypot(self.width, self.height)))
        # 不使用padding，雷达探测点应该在障碍物表面
        pad = 0.0
        dists = env_lib.ray_distances_multi(center, angles, max_radar_range, padding=pad)
        readings = (np.asarray(dists, dtype=np.float32) / max_radar_range) * 2.0 - 1.0
        return readings

    def _is_line_blocked(self, agent1, agent2, padding=0.0):
        x1 = agent1['x'] + self.pixel_size * 0.5
        y1 = agent1['y'] + self.pixel_size * 0.5
        x2 = agent2['x'] + self.pixel_size * 0.5
        y2 = agent2['y'] + self.pixel_size * 0.5
        dx, dy = (x2 - x1), (y2 - y1)
        line_len = math.hypot(dx, dy)
        if line_len <= 1e-6:
            return False

        angle = math.atan2(dy, dx)
        check_len = line_len
        dist = env_lib.ray_distance_grid((x1, y1), angle, check_len, padding=padding)
        return bool(dist < check_len - 1e-3)

    def _parse_actions(self, action, target_action=None):
        # 如果 action 是元组且包含两个元素，则解包为 (defender_action, attacker_action)
        if isinstance(action, (tuple, list)) and len(action) == 2 and target_action is None:
            # 检查第一个元素是否是动作（2个元素的数组）
            first_elem = np.asarray(action[0], dtype=np.float32).reshape(-1)
            if first_elem.size == 2:
                return action[0], action[1]
        return action, target_action

    def set_hard_action_mask(self, enabled: bool):
        self.hard_action_mask = bool(enabled)
        self.defender_hard_action_mask = bool(enabled)
        self.attacker_hard_action_mask = bool(enabled)

    def configure_hard_action_mask(
        self,
        enabled: bool,
        role: str = 'both',
        obstacle_margin: float = None,
        radius_scale: float = None,
        lookahead_steps: int = None,
        speed_cost_weight: float = None,
        max_deviation_cost: float = None,
        allow_zero_fallback: bool = None,
        speed_scales=None,
        angle_fracs=None,
        clearance_extras=None,
        recovery_window: int = None,
        recovery_trigger_rate: float = None,
        recovery_min_progress: float = None,
        recovery_steps: int = None,
        recovery_speed_fraction: float = None,
    ):
        role_name = str(role).strip().lower()
        flag = bool(enabled)
        if role_name in ('both', 'all'):
            self.defender_hard_action_mask = flag
            self.attacker_hard_action_mask = flag
        elif role_name == 'defender':
            self.defender_hard_action_mask = flag
        elif role_name == 'attacker':
            self.attacker_hard_action_mask = flag
        else:
            raise ValueError(f'Unknown hard mask role: {role}')

        self.hard_action_mask = bool(self.defender_hard_action_mask or self.attacker_hard_action_mask)

        if obstacle_margin is not None:
            self.hard_mask_obstacle_margin = float(max(0.0, obstacle_margin))
        if radius_scale is not None:
            self.hard_mask_radius_scale = float(max(0.0, radius_scale))
        if lookahead_steps is not None:
            self.hard_mask_lookahead_steps = int(max(1, lookahead_steps))
        if speed_cost_weight is not None:
            self.hard_mask_speed_cost_weight = float(max(0.0, speed_cost_weight))
        if max_deviation_cost is not None:
            max_deviation_cost = float(max_deviation_cost)
            self.hard_mask_max_deviation_cost = None if max_deviation_cost < 0.0 else max_deviation_cost
        if allow_zero_fallback is not None:
            self.hard_mask_allow_zero_fallback = bool(allow_zero_fallback)
        if speed_scales is not None:
            values = tuple(float(v) for v in speed_scales)
            if values:
                self.hard_mask_speed_scales = values
        if angle_fracs is not None:
            values = tuple(float(v) for v in angle_fracs)
            if values:
                self.hard_mask_angle_fracs = values
        if clearance_extras is not None:
            values = tuple(float(v) for v in clearance_extras)
            if values:
                self.hard_mask_clearance_extras = values
        if recovery_window is not None:
            self.hard_mask_recovery_window = int(max(2, recovery_window))
            for role_name in ('defender', 'attacker'):
                previous = list(self._hard_mask_outcome_history.get(role_name, ()))
                self._hard_mask_outcome_history[role_name] = deque(
                    previous[-self.hard_mask_recovery_window:],
                    maxlen=self.hard_mask_recovery_window,
                )
        if recovery_trigger_rate is not None:
            self.hard_mask_recovery_trigger_rate = float(
                np.clip(recovery_trigger_rate, 0.0, 1.0)
            )
        if recovery_min_progress is not None:
            self.hard_mask_recovery_min_progress = float(max(0.0, recovery_min_progress))
        if recovery_steps is not None:
            self.hard_mask_recovery_steps = int(max(1, recovery_steps))
        if recovery_speed_fraction is not None:
            self.hard_mask_recovery_speed_fraction = float(
                np.clip(recovery_speed_fraction, 0.0, 1.0)
            )

    def reset_hard_mask_recovery(self):
        for role_name in ('defender', 'attacker'):
            self._hard_mask_outcome_history[role_name].clear()
            self._hard_mask_recovery_remaining[role_name] = 0
            self._hard_mask_recovery_turn_sign[role_name] = 0.0

    def record_hard_mask_outcome(
        self,
        role: str,
        path_distance: float,
        intervened: bool,
    ):
        role_name = str(role).strip().lower()
        if role_name not in self._hard_mask_outcome_history:
            return
        value = float(path_distance)
        if not np.isfinite(value):
            return
        self._hard_mask_outcome_history[role_name].append(
            (value, bool(intervened))
        )

    def _hard_mask_recovery_requested(self, role: str) -> bool:
        history = self._hard_mask_outcome_history.get(role, ())
        if len(history) < int(self.hard_mask_recovery_window):
            return False
        intervention_rate = float(
            np.mean([float(item[1]) for item in history])
        )
        net_progress = float(history[0][0] - history[-1][0])
        return bool(
            intervention_rate >= self.hard_mask_recovery_trigger_rate
            and net_progress < self.hard_mask_recovery_min_progress
        )

    def _role_hard_action_mask_enabled(self, role: str) -> bool:
        role_name = str(role).strip().lower()
        if role_name == 'defender':
            return bool(self.defender_hard_action_mask)
        if role_name == 'attacker':
            return bool(self.attacker_hard_action_mask)
        return bool(self.hard_action_mask)

    def _estimate_candidate_clearance_level(self, center_x, center_y, base_padding: float) -> int:
        level = 0
        for extra in self.hard_mask_clearance_extras:
            padding = float(max(0.0, base_padding + float(extra)))
            if env_lib.is_point_blocked(center_x, center_y, padding=padding):
                break
            level += 1
        return level

    def _simulate_candidate_safety(self, agent, angle_delta, speed, role, base_padding: float):
        tmp_agent = dict(agent)
        min_clearance_level = None
        lookahead_steps = int(max(1, self.hard_mask_lookahead_steps))

        for _ in range(lookahead_steps):
            nx, ny = self._simulate_motion(tmp_agent, angle_delta, speed, role)
            cx = nx + self.pixel_size * 0.5
            cy = ny + self.pixel_size * 0.5
            clearance_level = self._estimate_candidate_clearance_level(cx, cy, base_padding)
            if clearance_level <= 0:
                return False, 0
            if min_clearance_level is None or clearance_level < min_clearance_level:
                min_clearance_level = clearance_level

            tmp_agent['x'] = float(nx)
            tmp_agent['y'] = float(ny)
            tmp_agent['theta'] = float((tmp_agent.get('theta', 0.0) + angle_delta) % 360.0)
            if role == 'attacker':
                tmp_agent['v'] = float(speed)

        return True, int(min_clearance_level or 0)

    def _get_action_limits(self, role):
        if role == 'attacker':
            max_turn = float(getattr(map_config, 'attacker_max_angular_speed', 12.0))
            max_speed = float(getattr(map_config, 'attacker_speed', 2.0))
            max_acc = float(getattr(map_config, 'attacker_max_acc', 0.6))
        else:
            max_turn = float(getattr(map_config, 'defender_max_angular_speed', 6.0))
            max_speed = float(getattr(map_config, 'defender_speed', 2.6))
            max_acc = None
        return max_turn, max_speed, max_acc

    def _simulate_motion(self, agent, angle_delta, speed, role):
        max_turn, max_speed, max_acc = self._get_action_limits(role)
        angle_delta = float(np.clip(angle_delta, -max_turn, max_turn))
        speed = float(np.clip(speed, 0.0, max_speed))

        if role == 'attacker' and max_acc is not None:
            prev_speed = float(agent.get('v', 0.0))
            speed = prev_speed + float(np.clip(speed - prev_speed, -max_acc, max_acc))

        new_theta = float((agent.get('theta', 0.0) + angle_delta) % 360.0)
        rad_theta = math.radians(new_theta)
        new_x = float(np.clip(
            agent['x'] + speed * math.cos(rad_theta),
            0, self.width - self.pixel_size
        ))
        new_y = float(np.clip(
            agent['y'] + speed * math.sin(rad_theta),
            0, self.height - self.pixel_size
        ))
        return new_x, new_y

    def _action_would_hit_obstacle(self, agent, angle_delta, speed, role) -> bool:
        new_x, new_y = self._simulate_motion(agent, angle_delta, speed, role)
        agent_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        center_x = float(new_x + self.pixel_size * 0.5)
        center_y = float(new_y + self.pixel_size * 0.5)
        return bool(env_lib.is_point_blocked(center_x, center_y, padding=agent_radius))

    def _encode_action_like_input(self, angle_delta, speed, role, normalized_input):
        if normalized_input:
            max_turn, max_speed, _ = self._get_action_limits(role)
            turn_norm = 0.0 if max_turn <= 1e-6 else float(np.clip(angle_delta / max_turn, -1.0, 1.0))
            speed_norm = 0.0 if max_speed <= 1e-6 else float(np.clip((speed / max_speed) * 2.0 - 1.0, -1.0, 1.0))
            return np.array([turn_norm, speed_norm], dtype=np.float32)
        return np.array([float(angle_delta), float(speed)], dtype=np.float32)

    def _apply_hard_action_mask(self, action, role):
        role_name = str(role).strip().lower()
        diagnostics = {
            'enabled': bool(self._role_hard_action_mask_enabled(role_name)),
            'original_unsafe': False,
            'intervened': False,
            'zero_fallback': False,
            'unsafe_passthrough': False,
            'action_delta': 0.0,
            'lookahead_steps': int(max(1, self.hard_mask_lookahead_steps)),
            'recovery_active': False,
            'recovery_triggered': False,
        }
        self._last_hard_mask_diagnostics[role_name] = diagnostics

        if action is None or not diagnostics['enabled']:
            return action

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            return action

        normalized_input = bool(np.all(np.abs(arr) <= 1.0 + 1e-6))
        physical = self._control_to_physical(arr, role)
        if physical is None:
            return action

        orig_angle, orig_speed = float(physical[0]), float(physical[1])
        max_turn, max_speed, _ = self._get_action_limits(role)
        ref_agent = self.attacker if role == 'attacker' else self.defender
        agent_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        obstacle_padding = float(agent_radius * self.hard_mask_radius_scale + self.hard_mask_obstacle_margin)

        recovery_remaining = int(self._hard_mask_recovery_remaining.get(role_name, 0))
        if recovery_remaining <= 0 and self._hard_mask_recovery_requested(role_name):
            recovery_remaining = int(self.hard_mask_recovery_steps)
            self._hard_mask_recovery_remaining[role_name] = recovery_remaining
            self._hard_mask_recovery_turn_sign[role_name] = 0.0
            diagnostics['recovery_triggered'] = True
        recovery_active = recovery_remaining > 0
        diagnostics['recovery_active'] = recovery_active

        orig_safe, _orig_clearance = self._simulate_candidate_safety(
            ref_agent,
            orig_angle,
            orig_speed,
            role,
            obstacle_padding,
        )
        if orig_safe and not recovery_active:
            return action
        diagnostics['original_unsafe'] = not orig_safe

        best = None
        candidate_speeds = [
            float(np.clip(orig_speed * s, 0.0, max_speed))
            for s in self.hard_mask_speed_scales
        ]
        if recovery_active:
            candidate_speeds.append(
                float(max_speed * self.hard_mask_recovery_speed_fraction)
            )
        candidate_speeds = list(dict.fromkeys(round(value, 8) for value in candidate_speeds))

        candidate_angles = [orig_angle]
        candidate_angles.extend(
            float(np.clip(frac * max_turn, -max_turn, max_turn))
            for frac in self.hard_mask_angle_fracs
        )
        candidate_angles = list(dict.fromkeys(round(value, 8) for value in candidate_angles))

        recovery_turn_sign = float(
            self._hard_mask_recovery_turn_sign.get(role_name, 0.0)
        )
        recovery_speed = float(max_speed * self.hard_mask_recovery_speed_fraction)

        for cand_speed in candidate_speeds:
            for cand_angle in candidate_angles:
                safe, clearance_level = self._simulate_candidate_safety(
                    ref_agent,
                    cand_angle,
                    cand_speed,
                    role,
                    obstacle_padding,
                )
                if not safe:
                    continue

                angle_cost = abs(cand_angle - orig_angle) / (max_turn + 1e-6)
                speed_cost = abs(cand_speed - orig_speed) / (max_speed + 1e-6)
                deviation_cost = angle_cost + self.hard_mask_speed_cost_weight * speed_cost
                if (
                    self.hard_mask_max_deviation_cost is not None
                    and deviation_cost > float(self.hard_mask_max_deviation_cost)
                ):
                    continue
                if recovery_active:
                    candidate_sign = float(np.sign(cand_angle))
                    sign_cost = 0.0
                    if recovery_turn_sign != 0.0:
                        sign_cost = 0.0 if candidate_sign == recovery_turn_sign else 1.0
                    speed_target_cost = abs(cand_speed - recovery_speed) / (max_speed + 1e-6)
                    score = (
                        sign_cost,
                        speed_target_cost,
                        -clearance_level,
                        -abs(cand_angle),
                        deviation_cost,
                    )
                else:
                    # Every candidate is already collision-free.  Preserve the
                    # actor command first; use extra clearance only to break
                    # ties instead of steering aggressively by default.
                    score = (deviation_cost, -clearance_level, -cand_speed)
                if best is None or score < best[0]:
                    best = (score, cand_angle, cand_speed)

        if best is None:
            if not self.hard_mask_allow_zero_fallback:
                diagnostics['unsafe_passthrough'] = True
                return action
            applied = self._encode_action_like_input(0.0, 0.0, role, normalized_input)
            diagnostics['zero_fallback'] = True
        else:
            applied = self._encode_action_like_input(best[1], best[2], role, normalized_input)
            if recovery_active and self._hard_mask_recovery_turn_sign[role_name] == 0.0:
                self._hard_mask_recovery_turn_sign[role_name] = float(np.sign(best[1]))

        if recovery_active:
            recovery_remaining = max(0, recovery_remaining - 1)
            self._hard_mask_recovery_remaining[role_name] = recovery_remaining
            if recovery_remaining == 0:
                self._hard_mask_outcome_history[role_name].clear()
                self._hard_mask_recovery_turn_sign[role_name] = 0.0

        applied_arr = np.asarray(applied, dtype=np.float32).reshape(-1)
        diagnostics['intervened'] = not bool(np.allclose(applied_arr, arr, atol=1e-6))
        diagnostics['action_delta'] = float(np.linalg.norm(applied_arr - arr, ord=2))
        return applied

    def _augment_info_with_hard_mask(self, info):
        info = dict(info or {})
        for role in ('defender', 'attacker'):
            diagnostics = dict(self._last_hard_mask_diagnostics.get(role, {}) or {})
            enabled = bool(diagnostics.get('enabled', self._role_hard_action_mask_enabled(role)))
            prefix = f'{role}_hard_mask_'
            info[prefix + 'enabled'] = enabled
            info[prefix + 'original_unsafe'] = bool(diagnostics.get('original_unsafe', False))
            info[prefix + 'intervened'] = bool(diagnostics.get('intervened', False))
            info[prefix + 'zero_fallback'] = bool(diagnostics.get('zero_fallback', False))
            info[prefix + 'unsafe_passthrough'] = bool(diagnostics.get('unsafe_passthrough', False))
            info[prefix + 'action_delta'] = float(diagnostics.get('action_delta', 0.0))
            info[prefix + 'recovery_active'] = bool(diagnostics.get('recovery_active', False))
            info[prefix + 'recovery_triggered'] = bool(diagnostics.get('recovery_triggered', False))
        return info

    def _control_to_physical(self, action, role):
        if action is None:
            return None
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != 2:
            raise ValueError("action must contain exactly two elements")
        if np.all(np.abs(arr) <= 1.0 + 1e-6):
            if role == 'defender':
                max_turn = float(getattr(map_config, 'defender_max_angular_speed', 6.0))
                max_speed = float(getattr(map_config, 'defender_speed', 2.6))
            elif role == 'attacker':
                max_turn = float(getattr(map_config, 'attacker_max_angular_speed', 12.0))
                max_speed = float(getattr(map_config, 'attacker_speed', 2.0))
            else:
                max_turn = 6.0
                max_speed = 2.0

            angle_delta = float(np.clip(arr[0], -1.0, 1.0) * max_turn)
            speed_factor = float(np.clip(arr[1], -1.0, 1.0))
            speed = (speed_factor + 1.0) * 0.5 * max_speed

            return angle_delta, speed
        return float(arr[0]), float(arr[1])

    def step(self, action=None, attacker_action=None):
        self.step_count += 1
        self._last_hard_mask_diagnostics = {}
        defender_action, attacker_action = self._parse_actions(action, attacker_action)

        # 保存移动前的位置（用于奖励计算）
        self.prev_defender_pos = self.defender.copy()
        self.prev_attacker_pos = self.attacker.copy()
        prev_defender_radar = None
        if self.current_obs is not None:
            prev_defender_obs = self.current_obs[0] if isinstance(self.current_obs, tuple) else self.current_obs
            prev_defender_obs = np.asarray(prev_defender_obs, dtype=np.float32).reshape(-1)
            radar_start = 7
            radar_end = radar_start + int(self.radar_rays)
            if prev_defender_obs.size >= radar_end:
                prev_defender_radar = prev_defender_obs[radar_start:radar_end]

        if defender_action is not None:
            defender_action = self._apply_hard_action_mask(defender_action, role='defender')
            defender_phys = self._control_to_physical(defender_action, 'defender')
            if defender_phys is not None:
                angle_delta, speed = defender_phys
                self.defender = env_lib.agent_move_velocity(self.defender, angle_delta, speed, self.defender_speed, role='defender')

        attacker_collision_attempt = False
        if attacker_action is not None:
            attacker_action = self._apply_hard_action_mask(attacker_action, role='attacker')
            attacker_phys = self._control_to_physical(attacker_action, 'attacker')
            if attacker_phys is not None:
                angle_delta, speed = attacker_phys
                attacker_mask = dict(
                    self._last_hard_mask_diagnostics.get('attacker', {}) or {}
                )
                if attacker_mask.get('zero_fallback', False):
                    # A target speed of zero still obeys the normal acceleration
                    # limit.  In the emergency fallback there is no collision-free
                    # dynamically feasible candidate, so the safety shield must
                    # override residual velocity as a last resort.
                    self.attacker['v'] = 0.0
                attacker_collision_attempt = self._action_would_hit_obstacle(
                    self.attacker,
                    angle_delta,
                    speed,
                    role='attacker',
                )
                self.attacker = env_lib.agent_move_velocity(self.attacker, angle_delta, speed, self.attacker_speed, role='attacker')

        self._fov_cache_valid = False
        self.defender_trajectory.append((self.defender['x'] + self.pixel_size / 2.0, self.defender['y'] + self.pixel_size / 2.0))
        self.attacker_trajectory.append((self.attacker['x'] + self.pixel_size / 2.0, self.attacker['y'] + self.pixel_size / 2.0))
        # 轨迹不截断，保留完整历史 (GIF/PNG绘制需要完整轨迹)

        # 更新 last_xxx_pos（用于轨迹记录等）
        self.last_defender_pos = self.defender.copy()
        self.last_attacker_pos = self.attacker.copy()

        agent_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        defender_center_x = self.defender['x'] + self.pixel_size * 0.5
        defender_center_y = self.defender['y'] + self.pixel_size * 0.5
        defender_blocked = env_lib.is_point_blocked(defender_center_x, defender_center_y, padding=agent_radius)
        if defender_blocked:
            # Defender 碰撞时回滚位置，避免穿墙
            self.defender = env_lib._resolve_obstacle_collision(self.prev_defender_pos, self.defender)
        attacker_center_x = self.attacker['x'] + self.pixel_size * 0.5
        attacker_center_y = self.attacker['y'] + self.pixel_size * 0.5
        attacker_blocked = bool(
            attacker_collision_attempt
            or env_lib.is_point_blocked(attacker_center_x, attacker_center_y, padding=agent_radius)
        )

        defender_captures_attacker = self._is_defender_capturing_attacker()
        if defender_captures_attacker:
            self._capture_counter_defender = min(self._capture_counter_defender + 1, self.capture_required_steps)
        else:
            self._capture_counter_defender = 0

        attacker_captures_target = self._is_attacker_capturing_target()
        if attacker_captures_target:
            self._capture_counter_attacker = min(self._capture_counter_attacker + 1, self.capture_required_steps)
        else:
            self._capture_counter_attacker = 0

        defender_radar = self._sense_agent_radar(self.defender, num_rays=self.radar_rays, full_circle=True)

        # Calculate reward based on reward_mode
        if self.reward_mode == 'chase':
            reward, terminated, truncated, info = reward_fn.reward_calculate_chase(
                self.defender, self.attacker, self.target,
                prev_defender=self.prev_defender_pos,
                prev_attacker=self.prev_attacker_pos,
                defender_collision=bool(defender_blocked),
                attacker_collision=bool(attacker_blocked),
                defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
                attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
                capture_progress_defender=int(self._capture_counter_defender),
                capture_progress_attacker=int(self._capture_counter_attacker),
                capture_required_steps=int(self.capture_required_steps),
                radar=defender_radar,
                prev_radar=prev_defender_radar,
                initial_dist_def_att=self.initial_dist_def_att
            )
        elif self.reward_mode == 'hrl':
            # HRL: use native TAD reward (no guidance terms).
            reward, terminated, truncated, info = reward_fn.reward_calculate_tad(
                self.defender, self.attacker, self.target,
                prev_defender=self.prev_defender_pos,
                prev_attacker=self.prev_attacker_pos,
                defender_collision=bool(defender_blocked),
                attacker_collision=bool(attacker_blocked),
                defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
                attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
                capture_progress_defender=int(self._capture_counter_defender),
                capture_progress_attacker=int(self._capture_counter_attacker),
                capture_required_steps=int(self.capture_required_steps),
                radar=defender_radar,
                initial_dist_def_tgt=self.initial_dist_def_tgt,
            )
        elif self.reward_mode == 'protect':
            reward, terminated, truncated, info = reward_fn.reward_calculate_protect(
                self.defender, self.attacker, self.target,
                prev_defender=self.prev_defender_pos,
                prev_attacker=self.prev_attacker_pos,
                defender_collision=bool(defender_blocked),
                attacker_collision=bool(attacker_blocked),
                defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
                attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
                capture_progress_defender=int(self._capture_counter_defender),
                capture_progress_attacker=int(self._capture_counter_attacker),
                capture_required_steps=int(self.capture_required_steps),
                radar=defender_radar,
                initial_dist_def_tgt=self.initial_dist_def_tgt,
                initial_dist_def_att=self.initial_dist_def_att,
            )
        else:  # 'standard'
            reward, terminated, truncated, info = reward_fn.reward_calculate_tad(
                self.defender, self.attacker, self.target,
                prev_defender=self.prev_defender_pos,
                prev_attacker=self.prev_attacker_pos,
                defender_collision=bool(defender_blocked),
                attacker_collision=bool(attacker_blocked),
                defender_captured=bool(self._capture_counter_defender >= self.capture_required_steps),
                attacker_captured=bool(self._capture_counter_attacker >= self.capture_required_steps),
                capture_progress_defender=int(self._capture_counter_defender),
                capture_progress_attacker=int(self._capture_counter_attacker),
                capture_required_steps=int(self.capture_required_steps),
                radar=defender_radar,
                initial_dist_def_tgt=self.initial_dist_def_tgt,
            )

        if self.reward_mode == 'hrl' or self.emit_skill_rewards:
            info['skill_rewards'] = self._compute_skill_reward_components(
                defender_blocked=bool(defender_blocked),
                attacker_blocked=bool(attacker_blocked),
                defender_radar=defender_radar,
                prev_defender_radar=prev_defender_radar,
            )
        info['defender_collision'] = bool(info.get('defender_collision', defender_blocked))
        info['attacker_collision'] = bool(info.get('attacker_collision', attacker_blocked))

        cur_dist_defender = float(math.hypot(self.defender['x'] - self.attacker['x'], self.defender['y'] - self.attacker['y']))
        if self._best_distance_attacker is None or cur_dist_defender < (self._best_distance_attacker - 1e-6):
            self._best_distance_attacker = cur_dist_defender

        cur_dist_target = float(math.hypot(self.attacker['x'] - self.target['x'], self.attacker['y'] - self.target['y']))
        if self._best_distance_target is None or cur_dist_target < (self._best_distance_target - 1e-6):
            self._best_distance_target = cur_dist_target

        info['closest_attacker_record_value'] = float(self._best_distance_attacker if self._best_distance_attacker is not None else cur_dist_defender)
        info['closest_target_record_value'] = float(self._best_distance_target if self._best_distance_target is not None else cur_dist_target)

        self.current_obs = self._get_obs_features()
        if self.step_count >= EnvParameters.EPISODE_LEN and not terminated:
            truncated = True
            # standard 超时按 defender 胜利并给予 +success_reward
            if self.reward_mode == 'standard':
                reward += float(getattr(map_config, 'success_reward', 20.0))
            if self.reward_mode == 'chase':
                timeout_info = dict(info) if info is not None else {}
                timeout_info['reason'] = 'timeout_task_failed'
                timeout_info['win'] = False
                info = timeout_info
            else:
                info = reward_fn.apply_timeout_defender_win(info)

        reward += self._regime_terminal_bonus(info)
        info = self._augment_info_with_regime(info)
        info = self._augment_info_with_hard_mask(info)
        if terminated or truncated:
            if _read_env_bool('TAD_REGIME_CONTINUOUS', False):
                self._update_ued_continuous_stats(getattr(self, 'current_ued_key', None), info)
            else:
                self._update_ued_bin_stats(
                    (getattr(self, 'current_speed_regime', 'default'), getattr(self, 'current_margin_regime', 'default')),
                    info,
                )
        return self.current_obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_hard_mask_recovery()
        options = dict(options or {})
        obstacle_density = options.get('obstacle_density', None)
        if obstacle_density is not None:
            map_config.set_obstacle_density(str(obstacle_density).strip().lower())
        self._apply_episode_regime(options=options)
        obstacle_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._curriculum_sampling_info = {
            'curriculum_sampling_attempts': 0,
            'curriculum_sampling_rejections': 0,
            'curriculum_sampling_astar_checks': 0,
            'curriculum_sampling_agent_pairs': 0,
            'curriculum_sampling_rejection_counts': {},
            'curriculum_sampling_fallback': False,
            'curriculum_sampling_fallback_count': 0,
            'curriculum_sampling_fallback_reason': '',
        }

        # 先构建一次障碍物（用于初始位置采样）
        density_level = getattr(map_config, 'current_obstacle_density', None)
        if hasattr(map_config, "regenerate_obstacles"):
            map_config.regenerate_obstacles(density_level=density_level, seed=obstacle_seed, target_pos=None)
        env_lib.build_occupancy(
            width=self.width,
            height=self.height,
            cell=getattr(map_config, 'occ_cell', getattr(map_config, 'pixel_size', self.pixel_size)),
            obstacles=getattr(map_config, 'obstacles', [])
        )

        # 采样 Defender、Attacker、Target 位置
        # Target 采样时会自动检查时间优势约束
        fixed_target_enabled = (
            _read_env_optional_float('TAD_FIXED_TARGET_CENTER_X') is not None
            or _read_env_optional_float('TAD_FIXED_TARGET_CENTER_Y') is not None
        )
        self.target = None
        for _ in range(100):
            self._curriculum_sampling_info['curriculum_sampling_agent_pairs'] += 1
            self.defender = self._sample_defender_state()
            self.attacker = self._sample_attacker_state()
            target_state = self._sample_target_state()
            if target_state is None:
                continue
            self.target = target_state
            # 不再需要额外的位置检查，_sample_target_state 已经确保了时间优势
            break
        if self.target is None:
            if fixed_target_enabled:
                raise RuntimeError('Failed to sample defender/attacker compatible with fixed target.')
            self._curriculum_sampling_info['curriculum_sampling_fallback'] = True
            self._curriculum_sampling_info['curriculum_sampling_fallback_count'] = 1
            self._curriculum_sampling_info['curriculum_sampling_fallback_reason'] = (
                'no_valid_target_after_agent_resampling'
            )
            print("[WARNING] Failed to sample curriculum-compatible geometry, using fallback target")
            self.target = {
                'x': float(self.width / 2),
                'y': float(self.height / 2),
                'theta': 0.0,
            }

        # 根据采样的target位置重新生成障碍物（过滤与target重叠的障碍物）
        target_pos = {
            'x': self.target['x'] + self.pixel_size * 0.5,
            'y': self.target['y'] + self.pixel_size * 0.5,
            'r': getattr(map_config, 'target_radius', 16),
        }
        if hasattr(map_config, "regenerate_obstacles"):
            map_config.regenerate_obstacles(density_level=density_level, seed=obstacle_seed, target_pos=target_pos)
        env_lib.build_occupancy(
            width=self.width,
            height=self.height,
            cell=getattr(map_config, 'occ_cell', getattr(map_config, 'pixel_size', self.pixel_size)),
            obstacles=getattr(map_config, 'obstacles', [])
        )
        self._finalize_curriculum_sampling_info()

        self.step_count = 0
        # Calculate initial edge distance between defender and target
        dx_init = (self.defender['x'] + self.pixel_size * 0.5) - (self.target['x'] + self.pixel_size * 0.5)
        dy_init = (self.defender['y'] + self.pixel_size * 0.5) - (self.target['y'] + self.pixel_size * 0.5)
        init_center_dist_def_tgt = math.hypot(dx_init, dy_init)
        target_radius = float(getattr(map_config, 'target_radius', 16.0))
        agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
        reach_radius = target_radius + agent_radius
        self.initial_dist_def_tgt = max(0.0, init_center_dist_def_tgt - reach_radius)
        
        # Calculate initial distance between defender and attacker (for chase reward)
        dx_def_att = (self.defender['x'] + self.pixel_size * 0.5) - (self.attacker['x'] + self.pixel_size * 0.5)
        dy_def_att = (self.defender['y'] + self.pixel_size * 0.5) - (self.attacker['y'] + self.pixel_size * 0.5)
        self.initial_dist_def_att = math.hypot(dx_def_att, dy_def_att)
        self.defender_trajectory = []
        self.attacker_trajectory = []
        self.defender_start_pos = (self.defender['x'] + self.pixel_size / 2.0, self.defender['y'] + self.pixel_size / 2.0)
        self.attacker_start_pos = (self.attacker['x'] + self.pixel_size / 2.0, self.attacker['y'] + self.pixel_size / 2.0)

        self.prev_defender_pos = self.defender.copy()
        self.last_defender_pos = self.defender.copy()
        self.prev_attacker_pos = self.attacker.copy()
        self.last_attacker_pos = self.attacker.copy()

        self.last_observed_attacker_pos = None
        self.steps_since_observed = 0
        self._capture_counter_defender = 0
        self._capture_counter_attacker = 0
        self._best_distance_target = float(math.hypot(self.attacker['x'] - self.target['x'], self.attacker['y'] - self.target['y']))

        self._fov_cache = None
        self._fov_cache_valid = False
        self.current_obs = self._get_obs_features()
        return self.current_obs, self._augment_info_with_regime({})

    def _check_initial_positions(self):
        """简化后的位置检查 - 只检查基本的最小距离"""
        dx = (self.defender['x'] + self.pixel_size * 0.5) - (self.attacker['x'] + self.pixel_size * 0.5)
        dy = (self.defender['y'] + self.pixel_size * 0.5) - (self.attacker['y'] + self.pixel_size * 0.5)
        dist = math.hypot(dx, dy)
        min_gap = float(getattr(map_config, 'agent_spawn_min_gap', 100.0))
        return dist >= min_gap

    def _sample_defender_state(self):
        """在地图任意位置随机生成Defender（只需避开障碍物）"""
        margin = 30
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))

        for attempt in range(512):
            x = float(self.np_random.uniform(margin, self.width - margin - self.pixel_size))
            y = float(self.np_random.uniform(margin, self.height - margin - self.pixel_size))

            center_x = x + self.pixel_size * 0.5
            center_y = y + self.pixel_size * 0.5

            if env_lib.is_point_blocked(center_x, center_y, padding=pad):
                continue

            # 随机朝向
            theta = float(self.np_random.uniform(0.0, 360.0))
            return {'x': x, 'y': y, 'theta': theta}

        print("[WARNING] Failed to spawn defender, using fallback position")
        return {'x': float(margin), 'y': float(margin), 'theta': 0.0}

    def _sample_attacker_state(self):
        """在地图任意位置随机生成Attacker（只需避开障碍物、与Defender保持最小距离）"""
        margin = 30.0
        min_dist_to_defender = float(getattr(map_config, 'agent_spawn_min_gap', 100.0))
        pad = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
        
        defender_cx = self.defender['x'] + self.pixel_size * 0.5
        defender_cy = self.defender['y'] + self.pixel_size * 0.5

        for attempt in range(512):
            x = float(self.np_random.uniform(margin, self.width - margin - self.pixel_size))
            y = float(self.np_random.uniform(margin, self.height - margin - self.pixel_size))

            cx = x + self.pixel_size * 0.5
            cy = y + self.pixel_size * 0.5

            # 避开障碍物
            if env_lib.is_point_blocked(cx, cy, padding=pad):
                continue

            # 与Defender保持最小距离
            dist_to_defender = math.hypot(cx - defender_cx, cy - defender_cy)
            if dist_to_defender < min_dist_to_defender:
                continue

            # 随机朝向
            theta = float(self.np_random.uniform(0.0, 360.0))
            return {'x': x, 'y': y, 'theta': theta}

        print("[WARNING] Failed to spawn attacker, using fallback position")
        return {'x': float(self.width - margin - self.pixel_size), 
                'y': float(self.height - margin - self.pixel_size), 
                'theta': 0.0}

    def _sample_target_state(self):
        """
        在地图任意位置随机生成Target
        
        核心约束: 保证 Defender 有时间拦截 Attacker
        条件: dist(T,A) * defender_speed > dist(T,D) * attacker_speed
        即: Attacker 到达 Target 的时间 > Defender 拦截的时间
        """
        margin = 50
        target_radius = getattr(map_config, 'target_radius', 16)
        min_dist = 80.0  # 与任何Agent的最小距离
        
        defender_cx = self.defender['x'] + self.pixel_size * 0.5
        defender_cy = self.defender['y'] + self.pixel_size * 0.5
        attacker_cx = self.attacker['x'] + self.pixel_size * 0.5
        attacker_cy = self.attacker['y'] + self.pixel_size * 0.5
        
        defender_speed = float(getattr(map_config, 'defender_speed', 2.6))
        attacker_speed = float(getattr(map_config, 'attacker_speed', 2.0))
        time_adv_lhs_scale = _read_env_float('TAD_TIME_ADV_LHS_SCALE', 1.0)
        time_adv_rhs_scale = _read_env_float('TAD_TIME_ADV_RHS_SCALE', 1.0)
        regime_margin_range = getattr(self, '_target_time_margin_range', None)
        astar_distance_range = getattr(self, '_attacker_target_astar_distance_range', None)

        def reject(reason: str) -> bool:
            stats = self._curriculum_sampling_info
            stats['curriculum_sampling_rejections'] = int(
                stats.get('curriculum_sampling_rejections', 0)
            ) + 1
            counts = stats.setdefault('curriculum_sampling_rejection_counts', {})
            counts[reason] = int(counts.get(reason, 0)) + 1
            return False

        def is_valid_target_center(cx: float, cy: float) -> bool:
            stats = self._curriculum_sampling_info
            stats['curriculum_sampling_attempts'] = int(
                stats.get('curriculum_sampling_attempts', 0)
            ) + 1
            if (
                cx < (margin + self.pixel_size * 0.5)
                or cx > (self.width - margin - self.pixel_size * 0.5)
                or cy < (margin + self.pixel_size * 0.5)
                or cy > (self.height - margin - self.pixel_size * 0.5)
            ):
                return reject('bounds')

            if env_lib.is_point_blocked(cx, cy, padding=target_radius):
                return reject('obstacle')

            dist_to_defender = math.hypot(cx - defender_cx, cy - defender_cy)
            dist_to_attacker = math.hypot(cx - attacker_cx, cy - attacker_cy)
            if dist_to_defender < min_dist or dist_to_attacker < min_dist:
                return reject('agent_clearance')

            if regime_margin_range is not None:
                agent_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5))
                target_rad = float(getattr(map_config, 'target_radius', 16.0))
                reach_radius = agent_radius + target_rad
                t_attacker = max(0.0, dist_to_attacker - reach_radius) / max(attacker_speed, 1e-6)
                t_defender = max(0.0, dist_to_defender - reach_radius) / max(defender_speed, 1e-6)
                time_margin = float(t_attacker - t_defender)
                if not float(regime_margin_range[0]) <= time_margin <= float(regime_margin_range[1]):
                    return reject('time_margin')

            if astar_distance_range is not None:
                reach_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5)) + float(
                    getattr(map_config, 'target_radius', 16.0)
                )
                euclidean_boundary_distance = max(0.0, dist_to_attacker - reach_radius)
                if euclidean_boundary_distance > float(astar_distance_range[1]):
                    return reject('astar_distance')
                stats['curriculum_sampling_astar_checks'] = int(
                    stats.get('curriculum_sampling_astar_checks', 0)
                ) + 1
                astar_distance = self._attacker_target_astar_distance(cx, cy)
                if not (
                    float(astar_distance_range[0])
                    <= astar_distance
                    <= float(astar_distance_range[1])
                ):
                    return reject('astar_distance')

            valid_default_margin = (
                dist_to_attacker * defender_speed * time_adv_lhs_scale
                > dist_to_defender * attacker_speed * time_adv_rhs_scale
            )
            if regime_margin_range is None and not valid_default_margin:
                return reject('default_time_advantage')
            return True

        fixed_center_x = _read_env_optional_float('TAD_FIXED_TARGET_CENTER_X')
        fixed_center_y = _read_env_optional_float('TAD_FIXED_TARGET_CENTER_Y')
        if fixed_center_x is not None or fixed_center_y is not None:
            if fixed_center_x is None or fixed_center_y is None:
                print("[WARN] Fixed target requires both TAD_FIXED_TARGET_CENTER_X/Y; fallback to random target")
            else:
                fixed_cx = float(fixed_center_x)
                fixed_cy = float(fixed_center_y)
                if is_valid_target_center(fixed_cx, fixed_cy):
                    return {
                        'x': float(fixed_cx - self.pixel_size * 0.5),
                        'y': float(fixed_cy - self.pixel_size * 0.5),
                        'theta': 0.0,
                    }
                return None

        for attempt in range(500):
            x = float(self.np_random.uniform(margin, self.width - margin - self.pixel_size))
            y = float(self.np_random.uniform(margin, self.height - margin - self.pixel_size))

            cx = x + self.pixel_size * 0.5
            cy = y + self.pixel_size * 0.5

            # 时间优势约束: dist(T,A) * defender_speed > dist(T,D) * attacker_speed
            # 可通过环境变量调节难度：
            # - 困难: TAD_TIME_ADV_LHS_SCALE=0.85
            # - 简单: TAD_TIME_ADV_RHS_SCALE=0.85
            # Defender 有足够时间拦截 Attacker
            if is_valid_target_center(cx, cy):
                return {'x': x, 'y': y, 'theta': 0.0}

        if regime_margin_range is not None or astar_distance_range is not None:
            return None

        # Preserve the legacy unconstrained behavior when the default geometry
        # condition cannot be met for the sampled agent pair.
        print("[WARNING] Failed to find target position with time advantage, using fallback")
        self._curriculum_sampling_info['curriculum_sampling_fallback'] = True
        self._curriculum_sampling_info['curriculum_sampling_fallback_count'] = 1
        self._curriculum_sampling_info['curriculum_sampling_fallback_reason'] = 'no_valid_target'
        return {
            'x': float(self.width / 2),
            'y': float(self.height / 2),
            'theta': 0.0,
        }

    def _attacker_target_astar_distance(self, target_cx: float, target_cy: float) -> float:
        attacker_center = (
            float(self.attacker['x'] + self.pixel_size * 0.5),
            float(self.attacker['y'] + self.pixel_size * 0.5),
        )
        path_length = astar_path_length(
            attacker_center,
            (float(target_cx), float(target_cy)),
            width=float(self.width),
            height=float(self.height),
            grid_size=float(self._curriculum_astar_grid_size),
            obstacle_padding=float(self._curriculum_astar_obstacle_padding),
            obstacles=getattr(map_config, 'obstacles', []),
        )
        reach_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5)) + float(
            getattr(map_config, 'target_radius', 16.0)
        )
        return max(0.0, float(path_length) - reach_radius)

    def _finalize_curriculum_sampling_info(self):
        target_cx = float(self.target['x'] + self.pixel_size * 0.5)
        target_cy = float(self.target['y'] + self.pixel_size * 0.5)
        defender_cx = float(self.defender['x'] + self.pixel_size * 0.5)
        defender_cy = float(self.defender['y'] + self.pixel_size * 0.5)
        attacker_cx = float(self.attacker['x'] + self.pixel_size * 0.5)
        attacker_cy = float(self.attacker['y'] + self.pixel_size * 0.5)
        reach_radius = float(getattr(map_config, 'agent_radius', self.pixel_size * 0.5)) + float(
            getattr(map_config, 'target_radius', 16.0)
        )
        attacker_distance = math.hypot(target_cx - attacker_cx, target_cy - attacker_cy)
        defender_distance = math.hypot(target_cx - defender_cx, target_cy - defender_cy)
        time_margin = (
            max(0.0, attacker_distance - reach_radius) / max(float(self.attacker_speed), 1e-6)
            - max(0.0, defender_distance - reach_radius) / max(float(self.defender_speed), 1e-6)
        )
        astar_distance = self._attacker_target_astar_distance(target_cx, target_cy)
        margin_range = getattr(self, '_target_time_margin_range', None)
        astar_range = getattr(self, '_attacker_target_astar_distance_range', None)
        margin_satisfied = margin_range is None or (
            float(margin_range[0]) <= time_margin <= float(margin_range[1])
        )
        astar_satisfied = astar_range is None or (
            float(astar_range[0]) <= astar_distance <= float(astar_range[1])
        )
        fallback = bool(self._curriculum_sampling_info.get('curriculum_sampling_fallback', False))
        constraints_satisfied = bool(not fallback and margin_satisfied and astar_satisfied)
        self._curriculum_sampling_info.update({
            'attacker_target_astar_distance': float(astar_distance),
            'curriculum_time_margin': float(time_margin),
            'curriculum_constraints_satisfied': constraints_satisfied,
        })
        requested_constraints = margin_range is not None or astar_range is not None
        if requested_constraints and not fallback and not constraints_satisfied:
            raise RuntimeError(
                'Final curriculum geometry changed after obstacle reconstruction and no longer '
                f"satisfies the requested constraints: {self._curriculum_sampling_info}"
            )
        if (
            not constraints_satisfied
            and _read_env_bool('ATTACKER_ABORT_ON_CURRICULUM_FALLBACK', False)
        ):
            raise RuntimeError(
                'Curriculum sampling did not satisfy the requested constraints: '
                f"{self._curriculum_sampling_info}"
            )

    def _get_fov_points(self, force_recompute=False):
        if self._fov_cache_valid and self._fov_cache is not None and not force_recompute:
            return self._fov_cache

        ss = getattr(map_config, 'ssaa', 1)
        cx_world = self.defender['x'] + map_config.pixel_size * 0.5
        cy_world = self.defender['y'] + map_config.pixel_size * 0.5
        cx = cx_world * ss
        cy = cy_world * ss

        heading_rad = math.radians(self.defender.get('theta', 0.0))
        fov_half = math.radians(self.fov_angle / 2.0)
        max_range = min(self.fov_range if self.fov_range != float('inf') else 500.0, 500.0)

        num_rays = 64
        angles = np.linspace(heading_rad - fov_half, heading_rad + fov_half, num_rays)
        dists = env_lib.ray_distances_multi((cx_world, cy_world), angles, max_range, padding=0.0)

        pts = [(cx, cy)]
        for i in range(num_rays):
            dist = dists[i]
            angle = angles[i]
            px = cx + dist * ss * math.cos(angle)
            py = cy + dist * ss * math.sin(angle)
            pts.append((px, py))

        self._fov_cache = pts
        self._fov_cache_valid = True
        return pts

    def render(self, mode='rgb_array', collision_info=None, style=None):
        """Render the current frame.
        
        Args:
            mode: render mode ('rgb_array')
            collision_info: optional collision info dict
            style: 'pygame' (fast, default) or 'matplotlib' (academic PNG style)
        """
        fov_points = self._get_fov_points()
        
        if style == 'matplotlib':
            # Academic matplotlib style — matches trajectory PNG exactly
            return env_lib.get_canvas_tad_matplotlib(
                self.target, self.defender, self.attacker,
                self.defender_trajectory, self.attacker_trajectory,
                fov_points=fov_points,
                fov_angle=self.fov_angle,
                defender_start_pos=getattr(self, 'defender_start_pos', None),
                attacker_start_pos=getattr(self, 'attacker_start_pos', None),
                collision_info=collision_info
            )
        
        # Default: fast Pygame renderer
        if pygame is not None and self._render_surface is None:
            ss = getattr(map_config, 'ssaa', 1)
            self._render_surface = pygame.Surface((self.width * ss, self.height * ss), flags=pygame.SRCALPHA)
        canvas = env_lib.get_canvas_tad(
            self.target, self.defender, self.attacker,
            self.defender_trajectory, self.attacker_trajectory,
            surface=self._render_surface,
            fov_points=fov_points,
            fov_angle=self.fov_angle,
            collision_info=collision_info
        )
        return canvas

    def close(self):
        self._render_surface = None
        env_lib.reset_mpl_renderer()

    def _is_defender_capturing_attacker(self):
        tx = self.defender['x'] + self.pixel_size * 0.5
        ty = self.defender['y'] + self.pixel_size * 0.5
        gx = self.attacker['x'] + self.pixel_size * 0.5
        gy = self.attacker['y'] + self.pixel_size * 0.5
        dx, dy = (gx - tx), (gy - ty)
        dist = math.hypot(dx, dy)
        if dist > self.capture_radius:
            return False

        defender_heading = float(self.defender.get('theta', 0.0))
        angle_to_attacker = math.degrees(math.atan2(dy, dx))
        rel = self._normalize_angle(angle_to_attacker - defender_heading)
        half_sector = self.capture_sector_angle_deg * 0.5
        if abs(rel) > half_sector:
            return False

        fov_half = self.fov_angle * 0.5
        in_fov = (abs(rel) <= fov_half)
        if not in_fov:
            return False

        if self._is_line_blocked(self.defender, self.attacker):
            return False
        return True

    def _is_attacker_capturing_target(self):
        """
        检查Attacker是否捕获Target

        捕获条件：Attacker和Target边缘相碰
        即：两者中心距离 <= attacker_radius + target_radius
        """
        ax = self.attacker['x'] + self.pixel_size * 0.5
        ay = self.attacker['y'] + self.pixel_size * 0.5
        tx = self.target['x'] + self.pixel_size * 0.5
        ty = self.target['y'] + self.pixel_size * 0.5
        dx, dy = (tx - ax), (ty - ay)
        dist = math.hypot(dx, dy)

        # 获取半径
        attacker_radius = float(getattr(map_config, 'agent_radius', 8))
        target_radius = float(getattr(map_config, 'target_radius', 16))

        # 捕获条件：边缘相碰
        # 中心距离 <= attacker_radius + target_radius
        return dist <= (attacker_radius + target_radius)

    def get_privileged_state(self):
        return {
            'defender': {
                'x': float(self.defender['x']),
                'y': float(self.defender['y']),
                'theta': float(self.defender['theta']),
                'center_x': float(self.defender['x'] + self.pixel_size * 0.5),
                'center_y': float(self.defender['y'] + self.pixel_size * 0.5)
            },
            'attacker': {
                'x': float(self.attacker['x']),
                'y': float(self.attacker['y']),
                'theta': float(self.attacker['theta']),
                'center_x': float(self.attacker['x'] + self.pixel_size * 0.5),
                'center_y': float(self.attacker['y'] + self.pixel_size * 0.5)
            },
            'target': {
                'x': float(self.target['x']),
                'y': float(self.target['y']),
                'theta': float(self.target['theta']),
                'center_x': float(self.target['x'] + self.pixel_size * 0.5),
                'center_y': float(self.target['y'] + self.pixel_size * 0.5)
            },
            'map': {
                'width': float(self.width),
                'height': float(self.height)
            }
        }


TrackingEnv = TADEnv
