import math
import os
import random
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TADEnv
from hrl.predictor import AttackerGRUPredictor
from ppo.alg_parameters import NetParameters
from ppo.nets import create_network
from ppo.util import build_critic_observation
from rule_policies.attacker_global import SUPPORTED_STRATEGIES, TRAINING_STRATEGIES, AttackerGlobalPolicy
from rule_policies.defender_global import DefenderGlobalPolicy


class HRLEnv(gym.Env):
    """HRL env with macro top-level actions and a GRU attacker predictor.

    Top-level action (3D, tanh range):
      - action[0:2]: protect/chase preference logits
      - action[2]: hold length control
    """

    def __init__(
        self,
        protect_model_path,
        chase_model_path=None,
        attacker_strategy='random',
        device='cpu',
        predictor_hidden_dim: int = 64,
        predictor_lr: float = 1e-3,
        predictor_train: bool = True,
        hold_min: int = 1,
        hold_max: int = 1,
        macro_gamma: float = 0.95,
        disable_hold_control: bool = True,
        disable_predictor: bool = False,
    ):
        super().__init__()
        self.env = TADEnv(reward_mode='hrl')

        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.device = torch.device(device)
        self.attacker_strategy_mode = attacker_strategy

        self.hold_min = int(max(1, hold_min))
        self.hold_max = int(max(self.hold_min, hold_max))
        self.macro_gamma = float(macro_gamma)
        self.disable_hold_control = bool(disable_hold_control)
        self.disable_predictor = bool(disable_predictor)

        self.protect_net = self._load_skill_model(protect_model_path, skill_name='protect')

        self.chase_net = None
        self.chase_policy = None
        if chase_model_path is not None and os.path.exists(chase_model_path):
            self.chase_net = self._load_skill_model(chase_model_path, skill_name='chase')
        else:
            self.chase_policy = DefenderGlobalPolicy(skill_mode='chase')
            print('[HRLEnv] chase model path missing, fallback to rule-based chase policy.')

        if attacker_strategy == 'random':
            init_strat = random.choice(TRAINING_STRATEGIES)
        else:
            init_strat = attacker_strategy
        self.attacker_policy = AttackerGlobalPolicy(strategy=init_strat)

        self.predictor = None
        self.predictor_optimizer = None
        if not self.disable_predictor:
            self.predictor = AttackerGRUPredictor(hidden_dim=int(predictor_hidden_dim)).to(self.device)
            self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=float(predictor_lr))
        self.predictor_train = bool(predictor_train)
        self.predictor_hidden = None
        self.last_prediction = np.array([0.5, 0.5], dtype=np.float32)
        self.last_predictor_loss = 0.0

        self.step_count = 0
        self.cached_obs = None

    def set_predictor_train(self, enabled: bool):
        self.predictor_train = bool(enabled)

    @staticmethod
    def _compat_numpy_checkpoint_load(path, device):
        import numpy as _np
        import sys as _sys

        if not hasattr(_np, '_core'):
            _sys.modules['numpy._core'] = _np.core
            _sys.modules['numpy._core.multiarray'] = _np.core.multiarray
        return torch.load(path, map_location=device, weights_only=False)

    @staticmethod
    def _detect_network_type(state_dict):
        keys = set(state_dict.keys())
        if any('tracking_branch' in k for k in keys):
            return 'nmn'
        if any('actor_backbone' in k for k in keys):
            critic_in_dim = None
            if 'critic_backbone.0.weight' in state_dict and hasattr(state_dict['critic_backbone.0.weight'], 'shape'):
                critic_in_dim = int(state_dict['critic_backbone.0.weight'].shape[1])
            if critic_in_dim == NetParameters.ACTOR_VECTOR_LEN:
                return 'mlp_noctde'
            return 'mlp'
        return 'nmn'

    def _load_skill_model(self, model_path, skill_name='skill'):
        if model_path is None:
            raise ValueError(f'{skill_name} model path is required.')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{skill_name} model not found: {model_path}')

        checkpoint = self._compat_numpy_checkpoint_load(model_path, self.device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        network_type = self._detect_network_type(state_dict)

        net = create_network(network_type).to(self.device)
        net.load_state_dict(state_dict)
        net.eval()
        print(f'[HRLEnv] Loaded {skill_name} model ({network_type}) from {model_path}')
        return net

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exps = np.exp(np.clip(shifted, -20.0, 20.0))
        return exps / (np.sum(exps) + 1e-8)

    def _decode_top_action(self, top_action: np.ndarray):
        action = np.asarray(top_action, dtype=np.float32).reshape(-1)
        if action.size not in (2, 3):
            raise ValueError(f'Top-level action must have 2 or 3 dims, got {action.shape}')

        skill_probs = self._softmax(action[:2])
        skill_idx = int(np.argmax(skill_probs))

        if self.disable_hold_control:
            return skill_idx, 1, skill_probs

        if action.size == 2:
            # Backward compatibility with old HRL checkpoints.
            hold_steps = self.hold_min
        else:
            hold_ratio = 0.5 * (float(np.clip(action[2], -1.0, 1.0)) + 1.0)
            hold_steps = int(round(self.hold_min + hold_ratio * (self.hold_max - self.hold_min)))
            hold_steps = int(np.clip(hold_steps, self.hold_min, self.hold_max))

        return skill_idx, hold_steps, skill_probs

    def _skill_action_from_net(self, net, defender_obs, attacker_obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(defender_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            critic_obs = build_critic_observation(defender_obs, attacker_obs)
            critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mean, _, _ = net(obs_tensor, critic_tensor)
            return torch.tanh(mean).cpu().numpy()[0]

    def _get_chase_action(self, defender_obs, attacker_obs):
        if self.chase_net is not None:
            return self._skill_action_from_net(self.chase_net, defender_obs, attacker_obs)

        priv_state = self.env.get_privileged_state()
        return self.chase_policy.get_action(defender_obs, priv_state)

    def _build_true_attacker_measurement(self):
        p_state = self.env.get_privileged_state()
        defender = p_state['defender']
        attacker = p_state['attacker']

        dx = attacker['center_x'] - defender['center_x']
        dy = attacker['center_y'] - defender['center_y']
        dist = math.hypot(dx, dy)

        abs_ang = math.degrees(math.atan2(dy, dx))
        rel_ang = ((abs_ang - defender['theta'] + 180.0) % 360.0) - 180.0

        fov_half = float(self.env.fov_angle) * 0.5
        in_fov = abs(rel_ang) <= fov_half
        occluded = self.env._is_line_blocked(self.env.defender, self.env.attacker) if in_fov else False
        is_visible = bool(in_fov and not occluded)

        map_diagonal = math.hypot(self.env.width, self.env.height)
        normalized_distance = np.clip((dist / map_diagonal) * 2.0 - 1.0, -1.0, 1.0)
        normalized_bearing = np.clip(rel_ang / 180.0, -1.0, 1.0)

        rel_x_norm = (normalized_distance + 1.0) * 0.5
        rel_y_norm = (normalized_bearing + 1.0) * 0.5
        return float(rel_x_norm), float(rel_y_norm), is_visible

    def _predict_attacker(self, train_predictor: bool):
        if self.disable_predictor or self.predictor is None:
            true_x, true_y, is_visible = self._build_true_attacker_measurement()
            self.last_prediction = np.array([true_x, true_y], dtype=np.float32)
            self.last_predictor_loss = 0.0
            return self.last_prediction.copy(), bool(is_visible)

        true_x, true_y, is_visible = self._build_true_attacker_measurement()

        if is_visible:
            meas_x, meas_y = true_x, true_y
        else:
            meas_x, meas_y = float(self.last_prediction[0]), float(self.last_prediction[1])

        inp = torch.tensor([[[meas_x, meas_y, 1.0 if is_visible else 0.0]]],
                           dtype=torch.float32, device=self.device)
        tgt = torch.tensor([[true_x, true_y]], dtype=torch.float32, device=self.device)

        pred, next_hidden = self.predictor(inp, self.predictor_hidden)
        pred = torch.clamp(pred, 0.0, 1.0)
        loss = F.mse_loss(pred, tgt)

        if train_predictor and self.predictor_train:
            self.predictor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.predictor_optimizer.step()

        self.predictor_hidden = next_hidden.detach()
        self.last_prediction = pred.detach().cpu().numpy()[0].astype(np.float32)
        self.last_predictor_loss = float(loss.item())

        return self.last_prediction.copy(), bool(is_visible)

    def _inject_prediction_to_obs(self, defender_obs, prediction, is_visible):
        obs = np.asarray(defender_obs, dtype=np.float32).copy()
        if is_visible:
            return obs

        pred_dist_norm = np.clip(prediction[0] * 2.0 - 1.0, -1.0, 1.0)
        pred_bearing_norm = np.clip(prediction[1] * 2.0 - 1.0, -1.0, 1.0)
        rel_ang = pred_bearing_norm * 180.0

        fov_half = float(self.env.fov_angle) * 0.5
        if fov_half > 1e-6:
            fov_edge_angle = min(abs(rel_ang + fov_half), abs(rel_ang - fov_half))
            normalized_fov_edge = np.clip((fov_edge_angle / fov_half) * 2.0 - 1.0, -1.0, 1.0)
        else:
            normalized_fov_edge = 0.0

        obs[0] = float(pred_dist_norm)
        obs[1] = float(pred_bearing_norm)
        obs[2] = float(normalized_fov_edge)
        obs[3] = 0.0
        return obs

    def _process_observation(self, raw_obs, train_predictor: bool):
        defender_obs, attacker_obs = raw_obs
        if self.disable_predictor:
            processed = (
                np.asarray(defender_obs, dtype=np.float32),
                np.asarray(attacker_obs, dtype=np.float32),
            )
            self.cached_obs = processed
            return processed

        prediction, is_visible = self._predict_attacker(train_predictor=train_predictor)
        actor_obs = self._inject_prediction_to_obs(defender_obs, prediction, is_visible)

        processed = (
            np.asarray(actor_obs, dtype=np.float32),
            np.asarray(attacker_obs, dtype=np.float32),
        )
        self.cached_obs = processed
        return processed

    def reset(self, seed=None, options=None):
        raw_obs, info = self.env.reset(seed=seed, options=options)

        if self.chase_policy is not None:
            self.chase_policy.reset()

        if self.attacker_strategy_mode == 'random':
            new_strat = random.choice(TRAINING_STRATEGIES)
            self.attacker_policy = AttackerGlobalPolicy(strategy=new_strat)
        else:
            self.attacker_policy.reset()

        self.predictor_hidden = None
        self.last_prediction = np.array([0.5, 0.5], dtype=np.float32)
        self.last_predictor_loss = 0.0

        self.step_count = 0
        obs = self._process_observation(raw_obs, train_predictor=False)
        return obs, info

    def macro_step(self, top_action):
        """Execute one top-level decision for multiple primitive env steps."""
        if self.cached_obs is None:
            self.cached_obs = self._process_observation(self.env.current_obs, train_predictor=False)

        skill_idx, hold_steps, skill_probs = self._decode_top_action(top_action)

        macro_reward = 0.0
        raw_reward_sum = 0.0
        discount_acc = 1.0
        primitive_steps = 0

        terminated = False
        truncated = False
        info = {}

        for _ in range(hold_steps):
            defender_obs, attacker_obs = self.cached_obs

            a_protect = self._skill_action_from_net(self.protect_net, defender_obs, attacker_obs)
            a_chase = self._get_chase_action(defender_obs, attacker_obs)
            final_action = a_protect if skill_idx == 0 else a_chase

            if hasattr(self.attacker_policy, 'get_action_with_info'):
                a_attacker, _ = self.attacker_policy.get_action_with_info(attacker_obs)
            else:
                a_attacker = self.attacker_policy.get_action(attacker_obs)

            next_raw_obs, reward, terminated, truncated, info = self.env.step(
                action=final_action,
                attacker_action=a_attacker,
            )

            primitive_steps += 1
            self.step_count += 1
            raw_reward_sum += float(reward)
            macro_reward += discount_acc * float(reward)
            discount_acc *= self.macro_gamma

            self._process_observation(next_raw_obs, train_predictor=True)

            info['a_protect'] = a_protect
            info['a_chase'] = a_chase
            info['selected_skill'] = 'protect' if skill_idx == 0 else 'chase'

            if terminated or truncated:
                break

        info['top_skill_idx'] = int(skill_idx)
        info['top_skill_probs'] = skill_probs.astype(np.float32)
        info['top_hold_steps'] = int(hold_steps)
        info['macro_steps'] = int(primitive_steps)
        info['macro_reward'] = float(macro_reward)
        info['raw_reward_sum'] = float(raw_reward_sum)
        info['macro_discount'] = float(self.macro_gamma ** primitive_steps)
        info['predictor_loss'] = self.last_predictor_loss
        info['attacker_pred_norm'] = self.last_prediction.copy()

        return self.cached_obs, float(macro_reward), bool(terminated), bool(truncated), info

    def step(self, action):
        return self.macro_step(action)
