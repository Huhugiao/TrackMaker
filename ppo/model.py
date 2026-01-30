"""
TAD PPO Model

包含:
- Model: PPO Actor-Critic模型封装类
  - 推理: step(), evaluate()
  - 训练: train() - 纯RL训练
  - 训练: train_mixed() - IL+RL混合训练（加权组合，不使用梯度投影）
  - 训练: imitation_train() - 纯IL训练
  - 权重管理: get_weights(), set_weights()

IL+RL混合架构:
- 使用IL权重进行加权组合: total_loss = il_weight * il_loss + (1 - il_weight) * rl_loss
- IL权重使用余弦退火，从初始值逐渐衰减到最终值
"""

import importlib
import numpy as np
import torch
from ppo.alg_parameters import NetParameters, TrainingParameters
from ppo.nets import DefenderNetMLP


class Model(object):
    """
    PPO Actor-Critic 模型
    """
    
    @staticmethod
    def _angle_limit() -> float:
        map_cfg = importlib.import_module('map_config')
        limit = float(getattr(map_cfg, 'defender_max_angular_speed',
                      getattr(map_cfg, 'max_turn_deg', 45.0)))
        return max(1.0, limit)
    
    @staticmethod
    def to_normalized_action(pair):
        max_turn = Model._angle_limit()
        angle_norm = float(np.clip(pair[0] / max_turn, -1.0, 1.0))
        speed_norm = float(np.clip(pair[1], 0.0, 1.0) * 2.0 - 1.0)
        return np.array([angle_norm, speed_norm], dtype=np.float32)
    
    @staticmethod
    def to_pre_tanh(action_normalized):
        clipped = np.clip(action_normalized, -0.999999, 0.999999)
        return np.arctanh(clipped).astype(np.float32)
    
    @staticmethod
    def from_normalized(action_normalized):
        max_turn = Model._angle_limit()
        angle = float(np.clip(action_normalized[0], -1.0, 1.0) * max_turn)
        speed = float(np.clip((action_normalized[1] + 1.0) * 0.5, 0.0, 1.0))
        return angle, speed
    
    def __init__(self, device, global_model=False):
        self.device = device
        self.network = DefenderNetMLP().to(device)
        
        if global_model:
            self.net_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=TrainingParameters.lr
            )
        else:
            self.net_optimizer = None
            
        self.network.train()
        self.current_lr = TrainingParameters.lr
    
    def get_weights(self):
        return {name: param.cpu() for name, param in self.network.state_dict().items()}
    
    def set_weights(self, weights):
        self.network.load_state_dict(weights)
    
    def _to_tensor(self, vector):
        if isinstance(vector, np.ndarray):
            input_vector = torch.from_numpy(vector).float().to(self.device)
        elif torch.is_tensor(vector):
            input_vector = vector.to(self.device).float()
        else:
            input_vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        if input_vector.dim() == 1:
            input_vector = input_vector.unsqueeze(0)
        return torch.nan_to_num(input_vector)
    
    @staticmethod
    def _log_prob_from_pre_tanh(pre_tanh, mean, log_std):
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        base_log_prob = dist.log_prob(pre_tanh)
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        return (base_log_prob - log_det_jac).sum(dim=-1)
    
    @torch.no_grad()
    def step(self, actor_obs, critic_obs):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        
        mean, value, log_std = self.network(actor_tensor, critic_tensor)
        
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        
        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(), \
               float(value.item()), float(log_prob.item())
    
    @torch.no_grad()
    def evaluate(self, actor_obs, critic_obs, greedy=True):
        actor_tensor = self._to_tensor(actor_obs)
        critic_tensor = self._to_tensor(critic_obs)
        
        mean, value, log_std = self.network(actor_tensor, critic_tensor)
        
        pre_tanh = mean if greedy else mean + torch.exp(log_std) * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, log_std)
        
        return action[0].cpu().numpy(), pre_tanh[0].cpu().numpy(), \
               float(value.item()), float(log_prob.item())
    
    def train(self, actor_obs=None, critic_obs=None, returns=None, values=None,
              actions=None, old_log_probs=None, mask=None,
              writer=None, global_step=None, perf_dict=None):
        """
        纯RL训练（PPO）
        """
        self.net_optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        policy_loss = 0.0
        entropy_loss = 0.0
        value_loss = 0.0
        approx_kl = 0.0
        clipfrac = 0.0
        grad_norm = 0.0
        adv_mean = 0.0
        adv_std = 0.0
        
        if actor_obs is not None:
            actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
            critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
            returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
            
            if actor_obs.dim() == 1:
                actor_obs = actor_obs.unsqueeze(0)
            if critic_obs.dim() == 1:
                critic_obs = critic_obs.unsqueeze(0)
            if returns.dim() == 0:
                returns = returns.unsqueeze(0)
            if values.dim() == 0:
                values = values.unsqueeze(0)
            if old_log_probs.dim() == 0:
                old_log_probs = old_log_probs.unsqueeze(0)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            if mask is None:
                mask = torch.ones_like(returns, dtype=torch.float32, device=self.device)
            else:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
                if mask.dim() == 0:
                    mask = mask.unsqueeze(0)
            
            mean_flat, value_flat, log_std_flat = self.network(actor_obs, critic_obs)
            new_values = value_flat.squeeze(-1)
            new_action_log_probs = self._log_prob_from_pre_tanh(actions, mean_flat, log_std_flat)
            
            raw_advantages = returns - values.squeeze(-1)
            valid_mask = mask > 0
            if valid_mask.sum() > 1:
                adv_std = float(raw_advantages[valid_mask].std().item())
                adv_mean = float(raw_advantages[valid_mask].mean().item())
                advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
            else:
                advantages = raw_advantages * 0.0
            
            advantages = advantages * mask
            
            ratio = torch.exp(new_action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                               1.0 + TrainingParameters.CLIP_RANGE) * advantages
            policy_loss_t = -torch.min(surr1, surr2).sum() / mask.sum().clamp_min(1.0)
            policy_loss = policy_loss_t.item()
            
            entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std_flat).sum(dim=-1)
            entropy_loss_t = -(entropy * mask).sum() / mask.sum().clamp_min(1.0)
            entropy_loss = entropy_loss_t.item()
            
            value_clipped = values.squeeze(-1) + torch.clamp(
                new_values - values.squeeze(-1),
                -TrainingParameters.VALUE_CLIP_RANGE,
                TrainingParameters.VALUE_CLIP_RANGE
            )
            v_loss1 = (new_values - returns) ** 2
            v_loss2 = (value_clipped - returns) ** 2
            value_loss_t = (torch.max(v_loss1, v_loss2) * mask).sum() / mask.sum().clamp_min(1.0)
            value_loss = value_loss_t.item()
            
            total_loss_t = (policy_loss_t + 
                          TrainingParameters.EX_VALUE_COEF * value_loss_t + 
                          TrainingParameters.ENTROPY_COEF * entropy_loss_t)
            total_loss_t.backward()
            total_loss = total_loss_t.item()
            
            with torch.no_grad():
                approx_kl = (old_log_probs - new_action_log_probs).mean().item()
                clipfrac = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()
            
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                TrainingParameters.MAX_GRAD_NORM
            ).item())
            
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad)
            
            self.net_optimizer.step()
        
        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, 0.0, clipfrac, grad_norm, adv_mean]
        return {'losses': losses, 'il_loss': None, 'il_filter_ratio': None}
    
    def train_mixed(self, actor_obs, critic_obs, actions, old_log_probs, 
                    returns, values, expert_actions, il_weight,
                    mask=None, writer=None, global_step=None):
        """
        IL+RL混合训练（加权组合，不使用梯度投影）
        
        total_loss = il_weight * il_loss + (1 - il_weight) * rl_loss
        
        Args:
            actor_obs: Actor观测
            critic_obs: Critic观测
            actions: 采样的动作（pre-tanh）
            old_log_probs: 旧的log概率
            returns: 回报
            values: 值估计
            expert_actions: 专家动作（归一化后，tanh后的空间）
            il_weight: IL损失权重（0~1）
            mask: 可选的mask
            writer: TensorBoard writer
            global_step: 当前步数
            
        Returns:
            损失字典
        """
        self.net_optimizer.zero_grad(set_to_none=True)
        
        # 转换为tensor
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        expert_actions = torch.as_tensor(expert_actions, dtype=torch.float32, device=self.device)
        
        # 确保维度正确
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if old_log_probs.dim() == 0:
            old_log_probs = old_log_probs.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if expert_actions.dim() == 1:
            expert_actions = expert_actions.unsqueeze(0)
        
        if mask is None:
            mask = torch.ones_like(returns, dtype=torch.float32, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)
        
        # 前向传播
        mean_flat, value_flat, log_std_flat = self.network(actor_obs, critic_obs)
        new_values = value_flat.squeeze(-1)
        new_action_log_probs = self._log_prob_from_pre_tanh(actions, mean_flat, log_std_flat)
        
        # ========== IL Loss ==========
        pred_actions = torch.tanh(mean_flat)
        il_mse = ((pred_actions - expert_actions) ** 2).sum(dim=-1)
        il_loss_t = (il_mse * mask).sum() / mask.sum().clamp_min(1.0)
        il_loss_value = float(il_loss_t.item())
        
        # ========== RL Loss (PPO) ==========
        # Advantage
        raw_advantages = returns - values.squeeze(-1)
        valid_mask = mask > 0
        if valid_mask.sum() > 1:
            adv_std = float(raw_advantages[valid_mask].std().item())
            adv_mean = float(raw_advantages[valid_mask].mean().item())
            advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
        else:
            adv_std = 0.0
            adv_mean = 0.0
            advantages = raw_advantages * 0.0
        
        advantages = advantages * mask
        
        # Policy loss (clipped surrogate)
        ratio = torch.exp(new_action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                           1.0 + TrainingParameters.CLIP_RANGE) * advantages
        policy_loss_t = -torch.min(surr1, surr2).sum() / mask.sum().clamp_min(1.0)
        policy_loss = policy_loss_t.item()
        
        # Entropy loss
        entropy = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std_flat).sum(dim=-1)
        entropy_loss_t = -(entropy * mask).sum() / mask.sum().clamp_min(1.0)
        entropy_loss = entropy_loss_t.item()
        
        # Value loss (clipped)
        value_clipped = values.squeeze(-1) + torch.clamp(
            new_values - values.squeeze(-1),
            -TrainingParameters.VALUE_CLIP_RANGE,
            TrainingParameters.VALUE_CLIP_RANGE
        )
        v_loss1 = (new_values - returns) ** 2
        v_loss2 = (value_clipped - returns) ** 2
        value_loss_t = (torch.max(v_loss1, v_loss2) * mask).sum() / mask.sum().clamp_min(1.0)
        value_loss = value_loss_t.item()
        
        # RL total loss
        rl_loss_t = (policy_loss_t + 
                    TrainingParameters.EX_VALUE_COEF * value_loss_t + 
                    TrainingParameters.ENTROPY_COEF * entropy_loss_t)
        
        # ========== 加权组合 ==========
        # total_loss = il_weight * il_loss + (1 - il_weight) * rl_loss
        rl_weight = 1.0 - il_weight
        total_loss_t = il_weight * il_loss_t + rl_weight * rl_loss_t
        total_loss = total_loss_t.item()
        
        # 反向传播
        total_loss_t.backward()
        
        # 计算统计量
        with torch.no_grad():
            approx_kl = (old_log_probs - new_action_log_probs).mean().item()
            clipfrac = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()
        
        # 梯度裁剪
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            TrainingParameters.MAX_GRAD_NORM
        ).item())
        
        # 处理NaN梯度
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad)
        
        # 优化器步进
        self.net_optimizer.step()
        
        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, 0.0, clipfrac, grad_norm, adv_mean]
        
        return {
            'losses': losses,
            'il_loss': il_loss_value,
            'il_weight': il_weight,
            'rl_loss': rl_loss_t.item()
        }
    
    def imitation_train(self, actor_obs, critic_obs, optimal_actions, writer=None, global_step=None):
        """
        纯模仿学习训练
        """
        self.net_optimizer.zero_grad(set_to_none=True)
        actor_obs = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
        optimal_actions = torch.as_tensor(optimal_actions, dtype=torch.float32, device=self.device)
        
        if actor_obs.dim() == 1:
            actor_obs = actor_obs.unsqueeze(0)
        if critic_obs.dim() == 1:
            critic_obs = critic_obs.unsqueeze(0)
        if optimal_actions.dim() == 1:
            optimal_actions = optimal_actions.unsqueeze(0)
        
        gate_mask = torch.ones(optimal_actions.shape[0], device=self.device)
        gate_ratio = 1.0
        if gate_mask.sum() <= 0:
            if writer is not None and global_step is not None:
                writer.add_scalar('IL/filter_ratio', gate_ratio, int(global_step))
            return [None, 0.0]
        
        mean, _, _ = self.network(actor_obs, critic_obs)
        pred_actions = torch.tanh(mean)
        il_loss = (((pred_actions - optimal_actions) ** 2).sum(dim=-1) * gate_mask).sum() / gate_mask.sum()
        
        if torch.isfinite(il_loss):
            il_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                TrainingParameters.MAX_GRAD_NORM
            )
            self.net_optimizer.step()
        else:
            grad_norm = 0.0
        
        return [float(il_loss.item()), float(grad_norm)]
    
    def update_learning_rate(self, new_lr):
        for group in self.net_optimizer.param_groups:
            group['lr'] = new_lr
            
    def save(self, path, step=None, reward=None):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            step: 当前训练步数 (用于RETRAIN恢复)
            reward: 当前最佳奖励 (用于RETRAIN恢复)
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            'model': self.get_weights(),
            'step': step if step is not None else 0,
            'reward': reward if reward is not None else -float('inf')
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint)
