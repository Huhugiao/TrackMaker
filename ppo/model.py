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
from ppo.nets import DefenderNetMLP, DefenderNetNMN, create_network


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
    
    def __init__(self, device, global_model=False, network_type='nmn'):
        self.device = device
        self.network_type = network_type
        self.network = create_network(network_type).to(device)
        
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
        纯RL训练（PPO）- Mini-batch多轮更新
        
        标准PPO流程:
        1. 在full batch上计算并标准化advantage
        2. 对数据做N_EPOCHS轮随机shuffle
        3. 每轮内按MINIBATCH_SIZE切分，每个mini-batch做一次梯度更新
        """
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
            
            # ====== 在full batch上计算并标准化advantage ======
            raw_advantages = returns - values.squeeze(-1)
            valid_mask = mask > 0
            if valid_mask.sum() > 1:
                adv_std = float(raw_advantages[valid_mask].std().item())
                adv_mean = float(raw_advantages[valid_mask].mean().item())
                advantages = ((raw_advantages - adv_mean) / (adv_std + 1e-8))
            else:
                advantages = raw_advantages * 0.0
            advantages = advantages * mask
            
            # ====== Mini-batch多轮更新 ======
            dataset_size = actor_obs.shape[0]
            minibatch_size = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
            n_epochs = TrainingParameters.N_EPOCHS_INITIAL
            
            # 累计统计量
            sum_policy_loss = 0.0
            sum_entropy_loss = 0.0
            sum_value_loss = 0.0
            sum_approx_kl = 0.0
            sum_clipfrac = 0.0
            sum_grad_norm = 0.0
            n_updates = 0
            
            for epoch in range(n_epochs):
                indices = torch.randperm(dataset_size, device=self.device)
                
                for start in range(0, dataset_size, minibatch_size):
                    end = min(start + minibatch_size, dataset_size)
                    mb_idx = indices[start:end]
                    
                    mb_actor_obs = actor_obs[mb_idx]
                    mb_critic_obs = critic_obs[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_values = values[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_mask = mask[mb_idx]
                    
                    self.net_optimizer.zero_grad(set_to_none=True)
                    
                    mean_flat, value_flat, log_std_flat = self.network(mb_actor_obs, mb_critic_obs)
                    new_values = value_flat.squeeze(-1)
                    new_log_probs = self._log_prob_from_pre_tanh(mb_actions, mean_flat, log_std_flat)
                    
                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                       1.0 + TrainingParameters.CLIP_RANGE) * mb_advantages
                    mb_mask_sum = mb_mask.sum().clamp_min(1.0)
                    policy_loss_t = -torch.min(surr1, surr2).sum() / mb_mask_sum
                    
                    # Entropy loss
                    ent = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std_flat).sum(dim=-1)
                    entropy_loss_t = -(ent * mb_mask).sum() / mb_mask_sum
                    
                    # Value loss (clipped)
                    value_clipped = mb_values.squeeze(-1) + torch.clamp(
                        new_values - mb_values.squeeze(-1),
                        -TrainingParameters.VALUE_CLIP_RANGE,
                        TrainingParameters.VALUE_CLIP_RANGE
                    )
                    v_loss1 = (new_values - mb_returns) ** 2
                    v_loss2 = (value_clipped - mb_returns) ** 2
                    value_loss_t = (torch.max(v_loss1, v_loss2) * mb_mask).sum() / mb_mask_sum
                    
                    total_loss_t = (policy_loss_t +
                                   TrainingParameters.EX_VALUE_COEF * value_loss_t +
                                   TrainingParameters.ENTROPY_COEF * entropy_loss_t)
                    total_loss_t.backward()
                    
                    with torch.no_grad():
                        mb_kl = (mb_old_log_probs - new_log_probs).mean().item()
                        mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()
                    
                    gn = float(torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        TrainingParameters.MAX_GRAD_NORM
                    ).item())
                    
                    for name, param in self.network.named_parameters():
                        if param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad)
                    
                    self.net_optimizer.step()
                    
                    sum_policy_loss += policy_loss_t.item()
                    sum_entropy_loss += entropy_loss_t.item()
                    sum_value_loss += value_loss_t.item()
                    sum_approx_kl += mb_kl
                    sum_clipfrac += mb_cf
                    sum_grad_norm += gn
                    n_updates += 1
                
                # Early stopping: 如果KL散度过大，停止后续epoch
                avg_kl_so_far = sum_approx_kl / max(n_updates, 1)
                if abs(avg_kl_so_far) > 0.03:
                    break
            
            # 计算平均统计量
            if n_updates > 0:
                policy_loss = sum_policy_loss / n_updates
                entropy_loss = sum_entropy_loss / n_updates
                value_loss = sum_value_loss / n_updates
                approx_kl = sum_approx_kl / n_updates
                clipfrac = sum_clipfrac / n_updates
                grad_norm = sum_grad_norm / n_updates
                total_loss = (policy_loss +
                             TrainingParameters.EX_VALUE_COEF * value_loss +
                             TrainingParameters.ENTROPY_COEF * entropy_loss)
        
        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl, 0.0, clipfrac, grad_norm, adv_mean]
        return {'losses': losses, 'il_loss': None, 'il_filter_ratio': None}
    
    def train_mixed(self, actor_obs, critic_obs, actions, old_log_probs, 
                    returns, values, expert_actions, il_weight,
                    mask=None, writer=None, global_step=None):
        """
        IL+RL混合训练（加权组合）- Mini-batch多轮更新
        
        total_loss = il_weight * il_loss + (1 - il_weight) * rl_loss
        """
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
        
        # ====== 在full batch上计算并标准化advantage ======
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
        
        # ====== Mini-batch多轮更新 ======
        dataset_size = actor_obs.shape[0]
        minibatch_size = min(TrainingParameters.MINIBATCH_SIZE, dataset_size)
        n_epochs = TrainingParameters.N_EPOCHS_INITIAL
        rl_weight = 1.0 - il_weight
        
        sum_policy_loss = 0.0
        sum_entropy_loss = 0.0
        sum_value_loss = 0.0
        sum_il_loss = 0.0
        sum_total_loss = 0.0
        sum_rl_loss = 0.0
        sum_approx_kl = 0.0
        sum_clipfrac = 0.0
        sum_grad_norm = 0.0
        n_updates = 0
        
        for epoch in range(n_epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            
            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                mb_idx = indices[start:end]
                
                mb_actor_obs = actor_obs[mb_idx]
                mb_critic_obs = critic_obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values = values[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_mask = mask[mb_idx]
                mb_expert_actions = expert_actions[mb_idx]
                
                self.net_optimizer.zero_grad(set_to_none=True)
                
                mean_flat, value_flat, log_std_flat = self.network(mb_actor_obs, mb_critic_obs)
                new_values = value_flat.squeeze(-1)
                new_log_probs = self._log_prob_from_pre_tanh(mb_actions, mean_flat, log_std_flat)
                mb_mask_sum = mb_mask.sum().clamp_min(1.0)
                
                # ========== IL Loss ==========
                pred_actions = torch.tanh(mean_flat)
                il_mse = ((pred_actions - mb_expert_actions) ** 2).sum(dim=-1)
                il_loss_t = (il_mse * mb_mask).sum() / mb_mask_sum
                
                # ========== RL Loss (PPO) ==========
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                ratio = torch.clamp(ratio, 0.0, TrainingParameters.RATIO_CLAMP_MAX)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                   1.0 + TrainingParameters.CLIP_RANGE) * mb_advantages
                policy_loss_t = -torch.min(surr1, surr2).sum() / mb_mask_sum
                
                ent = (0.5 * (1.0 + np.log(2 * np.pi)) + log_std_flat).sum(dim=-1)
                entropy_loss_t = -(ent * mb_mask).sum() / mb_mask_sum
                
                value_clipped = mb_values.squeeze(-1) + torch.clamp(
                    new_values - mb_values.squeeze(-1),
                    -TrainingParameters.VALUE_CLIP_RANGE,
                    TrainingParameters.VALUE_CLIP_RANGE
                )
                v_loss1 = (new_values - mb_returns) ** 2
                v_loss2 = (value_clipped - mb_returns) ** 2
                value_loss_t = (torch.max(v_loss1, v_loss2) * mb_mask).sum() / mb_mask_sum
                
                rl_loss_t = (policy_loss_t +
                            TrainingParameters.EX_VALUE_COEF * value_loss_t +
                            TrainingParameters.ENTROPY_COEF * entropy_loss_t)
                
                # ========== 加权组合 ==========
                total_loss_t = il_weight * il_loss_t + rl_weight * rl_loss_t
                total_loss_t.backward()
                
                with torch.no_grad():
                    mb_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    mb_cf = (torch.abs(ratio - 1.0) > TrainingParameters.CLIP_RANGE).float().mean().item()
                
                gn = float(torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    TrainingParameters.MAX_GRAD_NORM
                ).item())
                
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad)
                
                self.net_optimizer.step()
                
                sum_policy_loss += policy_loss_t.item()
                sum_entropy_loss += entropy_loss_t.item()
                sum_value_loss += value_loss_t.item()
                sum_il_loss += il_loss_t.item()
                sum_rl_loss += rl_loss_t.item()
                sum_total_loss += total_loss_t.item()
                sum_approx_kl += mb_kl
                sum_clipfrac += mb_cf
                sum_grad_norm += gn
                n_updates += 1
            
            # Early stopping
            avg_kl_so_far = sum_approx_kl / max(n_updates, 1)
            if abs(avg_kl_so_far) > 0.03:
                break
        
        # 计算平均统计量
        if n_updates > 0:
            policy_loss = sum_policy_loss / n_updates
            entropy_loss = sum_entropy_loss / n_updates
            value_loss = sum_value_loss / n_updates
            il_loss_value = sum_il_loss / n_updates
            total_loss = sum_total_loss / n_updates
            approx_kl_avg = sum_approx_kl / n_updates
            clipfrac_avg = sum_clipfrac / n_updates
            grad_norm_avg = sum_grad_norm / n_updates
            rl_loss_avg = sum_rl_loss / n_updates
        else:
            policy_loss = 0.0
            entropy_loss = 0.0
            value_loss = 0.0
            il_loss_value = 0.0
            total_loss = 0.0
            approx_kl_avg = 0.0
            clipfrac_avg = 0.0
            grad_norm_avg = 0.0
            rl_loss_avg = 0.0
        
        losses = [total_loss, policy_loss, entropy_loss, value_loss,
                  adv_std, approx_kl_avg, 0.0, clipfrac_avg, grad_norm_avg, adv_mean]
        
        return {
            'losses': losses,
            'il_loss': il_loss_value,
            'il_weight': il_weight,
            'rl_loss': rl_loss_avg
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.network.load_state_dict(checkpoint['model'])
        else:
            self.network.load_state_dict(checkpoint)
