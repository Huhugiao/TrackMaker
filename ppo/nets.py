"""
TAD PPO 网络结构

网络架构:
- RadarEncoder: 将64维雷达观测编码为8维向量
- DefenderNetMLP: Defender的Actor-Critic网络 (MLP版本)
  - Actor: 输入Defender观测，输出动作分布
  - Critic: 输入Defender观测 + Attacker特权观测 (CTDE)
"""

import torch
import torch.nn as nn
import numpy as np
from ppo.alg_parameters import NetParameters


class RadarEncoder(nn.Module):
    """
    雷达编码器: 将64维雷达信号编码为低维向量
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NetParameters.RADAR_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, NetParameters.RADAR_EMBED_DIM),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)


class DefenderNetMLP(nn.Module):
    """
    Defender Actor-Critic 网络 (MLP版本)
    
    Actor输入: Defender观测 (71维)
      - attacker_info: 5维 [distance, bearing, fov_edge, is_visible, unobserved_time]
      - radar: 64维
      - target_info: 2维 [target_distance, target_bearing]
      
    Critic输入: Defender观测 + Attacker特权观测 (142维, CTDE)
    
    输出:
      - mean: 动作均值 (2维)
      - value: 状态价值
      - log_std: 动作标准差对数
    """
    def __init__(self):
        super(DefenderNetMLP, self).__init__()
        
        self.hidden_dim = NetParameters.HIDDEN_DIM
        self.num_layers = getattr(NetParameters, 'NUM_HIDDEN_LAYERS', 3)
        
        # Radar Encoder (shared)
        self.radar_encoder = RadarEncoder()
        
        # --- Actor Network ---
        # Input: Scalar (7) + Embedded Radar (8) = 15 (Encoded)
        self.actor_backbone = self._build_mlp(
            NetParameters.ACTOR_VECTOR_LEN,  # 16
            self.hidden_dim,
            self.num_layers
        )
        
        self.policy_mean = nn.Linear(self.hidden_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))
        
        # --- Critic Network (CTDE: 训练时使用完整观测) ---
        # Input: Defender (15) + Attacker (15) = 30 (Encoded)
        self.critic_backbone = self._build_mlp(
            NetParameters.CRITIC_VECTOR_LEN,  # 32
            self.hidden_dim,
            self.num_layers
        )
        
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        """构建MLP网络"""
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def _encode_observation(self, obs, is_critic=False):
        """
        编码观测
        
        Args:
            obs: 原始观测 [Batch, RAW_LEN]
            is_critic: 是否为Critic观测
            
        Returns:
            encoded: 编码后的观测 [Batch, ENCODED_LEN]
        """
        if is_critic:
            # Critic观测: Defender(71) + Attacker(71) = 142
            defender_end = NetParameters.ACTOR_RAW_LEN  # 71
            
            # Defender部分
            defender_scalar = obs[:, :NetParameters.ACTOR_SCALAR_LEN]  # 7
            defender_radar = obs[:, NetParameters.ACTOR_SCALAR_LEN:defender_end]  # 64
            defender_radar_emb = self.radar_encoder(defender_radar)  # 8
            defender_part = torch.cat([defender_scalar, defender_radar_emb], dim=-1)  # 15
            
            # Attacker部分
            attacker_start = defender_end
            attacker_scalar = obs[:, attacker_start:attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN]  # 7
            attacker_radar = obs[:, attacker_start + NetParameters.PRIVILEGED_SCALAR_LEN:]  # 64
            attacker_radar_emb = self.radar_encoder(attacker_radar)  # 8
            attacker_part = torch.cat([attacker_scalar, attacker_radar_emb], dim=-1)  # 15
            
            return torch.cat([defender_part, attacker_part], dim=-1)  # 30
        else:
            # Actor观测: 71维
            scalar = obs[:, :NetParameters.ACTOR_SCALAR_LEN]  # 7
            radar = obs[:, NetParameters.ACTOR_SCALAR_LEN:]  # 64
            radar_emb = self.radar_encoder(radar)  # 8
            return torch.cat([scalar, radar_emb], dim=-1)  # 15
    
    def forward(self, actor_obs, critic_obs):
        """
        前向传播
        
        Args:
            actor_obs: [Batch, 71] Defender观测
            critic_obs: [Batch, 142] Defender观测 + Attacker观测
            
        Returns:
            mean: 动作均值 [Batch, 2]
            value: 状态价值 [Batch, 1]
            log_std: 动作标准差对数 [Batch, 2]
        """
        # 编码观测
        actor_in = self._encode_observation(actor_obs, is_critic=False)  # [Batch, 15]
        critic_in = self._encode_observation(critic_obs, is_critic=True)  # [Batch, 30]
        
        # --- Actor Forward ---
        a_out = self.actor_backbone(actor_in)
        mean = self.policy_mean(a_out)
        log_std = self.log_std.expand_as(mean)
        
        # --- Critic Forward ---
        c_out = self.critic_backbone(critic_in)
        value = self.value_head(c_out)
        
        return mean, value, log_std
    
    def act(self, actor_obs, critic_obs):
        """采样动作 (用于训练)"""
        mean, value, log_std = self.forward(actor_obs, critic_obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        # 也可以手动采样以保持 consistent with pre-tanh logic if needed
        # but here we follow standard PPO sampling
        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        
        # log_prob from pre_tanh
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        log_prob = (dist.log_prob(pre_tanh) - log_det_jac).sum(dim=-1)
        
        return action, log_prob, pre_tanh, value
    
    def critic_value(self, critic_obs):
        """计算状态价值 (用于Critic)"""
        critic_in = self._encode_observation(critic_obs, is_critic=True)
        c_out = self.critic_backbone(critic_in)
        return self.value_head(c_out)
