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


class DefenderNetNMN(nn.Module):
    """
    Defender Neural Modular Network (NMN) Actor-Critic

    基于神经模块网络(NMN)融合方法:
    "基于任务分解的状态并行提取与特征拼接融合"

    Actor (NMN架构):
      1. 输入解耦: 将观测分解为跟踪子集和避障子集
         - 跟踪分支: attacker_info(5) + target_info(2) = 7
         - 避障分支: radar(64) → RadarEncoder → 8
      2. 并行特征提取:
         - h1 跟踪分支: FC(7→32) + Tanh → 32维跟踪特征
         - h2 避障分支: FC(8→32) + Tanh → 32维避障特征
      3. 特征拼接融合: Concatenate → 64维复合特征向量
      4. 深度交互合并: h3 = FC(64→64) + Tanh (平衡两个任务优先级)
      5. 动作映射输出: FC(64→2) → π(a_t|s_t)

    Critic (普通MLP，与Actor相同输入，非CTDE):
      - 输入: encoded actor obs (15维)
      - MLP: FC(15→64) + Tanh → FC(64→64) + Tanh
      - 输出: value head FC(64→1)
    """
    def __init__(self):
        super().__init__()

        # Shared radar encoder
        self.radar_encoder = RadarEncoder()

        # Dimension aliases
        tracking_dim = NetParameters.ACTOR_SCALAR_LEN   # 7 (attacker 5 + target 2)
        obstacle_dim = NetParameters.RADAR_EMBED_DIM    # 8 (encoded radar)
        branch_dim   = NetParameters.NMN_BRANCH_DIM     # 32
        merged_dim   = NetParameters.NMN_MERGED_DIM     # 64

        # --- NMN Actor ---
        # h1: 跟踪分支 (S_uav + S_tar → FC32)
        self.tracking_branch = nn.Sequential(
            nn.Linear(tracking_dim, branch_dim),
            nn.Tanh()
        )

        # h2: 避障分支 (S_uav + S_obs → FC32)
        self.obstacle_branch = nn.Sequential(
            nn.Linear(obstacle_dim, branch_dim),
            nn.Tanh()
        )

        # h3: 合并隐藏层 (拼接64 → FC64, 非线性加权整合)
        self.merged_layer = nn.Sequential(
            nn.Linear(branch_dim * 2, merged_dim),
            nn.Tanh()
        )

        # 动作映射输出
        self.policy_mean = nn.Linear(merged_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))

        # --- Critic (普通MLP, 与Actor相同输入, 非CTDE) ---
        critic_input = NetParameters.ACTOR_VECTOR_LEN   # 15 (scalar 7 + radar_emb 8)
        critic_hidden = NetParameters.NMN_CRITIC_HIDDEN 
        critic_layers = NetParameters.NMN_CRITIC_LAYERS
        self.critic_backbone = self._build_mlp(
            critic_input, critic_hidden, critic_layers
        )
        self.value_head = nn.Linear(critic_hidden, 1)

        # Initialize weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_mlp(input_dim, hidden_dim, num_layers):
        """构建MLP网络"""
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(module):
        """正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    # ------------------------------------------------------------------
    def _encode_actor_obs(self, obs):
        """
        编码Actor观测: raw(71) → scalar(7), radar_emb(8)
        """
        scalar = obs[:, :NetParameters.ACTOR_SCALAR_LEN]      # [B, 7]
        radar  = obs[:, NetParameters.ACTOR_SCALAR_LEN:]       # [B, 64]
        radar_emb = self.radar_encoder(radar)                  # [B, 8]
        return scalar, radar_emb

    # ------------------------------------------------------------------
    def forward(self, actor_obs, critic_obs):
        """
        前向传播

        Args:
            actor_obs:  [B, 71] Defender观测
            critic_obs: [B, *]  (NMN Critic不使用CTDE, 忽略此参数, 使用actor_obs)

        Returns:
            mean:    动作均值 [B, 2]
            value:   状态价值 [B, 1]
            log_std: 动作标准差对数 [B, 2]
        """
        # 编码Actor观测
        scalar, radar_emb = self._encode_actor_obs(actor_obs)

        # --- NMN Actor 前向 ---
        h1 = self.tracking_branch(scalar)          # [B, 32] 跟踪特征
        h2 = self.obstacle_branch(radar_emb)       # [B, 32] 避障特征
        h_cat = torch.cat([h1, h2], dim=-1)        # [B, 64] 特征拼接
        h3 = self.merged_layer(h_cat)              # [B, 64] 深度交互
        mean = self.policy_mean(h3)                # [B, 2]  动作均值
        log_std = self.log_std.expand_as(mean)

        # --- Critic 前向 (使用actor_obs, 非CTDE) ---
        critic_in = torch.cat([scalar, radar_emb], dim=-1)  # [B, 15]
        c_out = self.critic_backbone(critic_in)
        value = self.value_head(c_out)

        return mean, value, log_std

    # ------------------------------------------------------------------
    def act(self, actor_obs, critic_obs):
        """采样动作 (用于训练)"""
        mean, value, log_std = self.forward(actor_obs, critic_obs)
        std = torch.exp(log_std)

        eps = torch.randn_like(mean)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)

        # log_prob with tanh squashing correction
        log_det_jac = torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)
        log_prob = (torch.distributions.Normal(mean, std).log_prob(pre_tanh)
                    - log_det_jac).sum(dim=-1)

        return action, log_prob, pre_tanh, value

    # ------------------------------------------------------------------
    def critic_value(self, critic_obs):
        """
        计算状态价值

        兼容CTDE obs(143维)和actor obs(71维):
        只取前ACTOR_RAW_LEN维作为actor观测
        """
        actor_obs = critic_obs[:, :NetParameters.ACTOR_RAW_LEN]
        scalar, radar_emb = self._encode_actor_obs(actor_obs)
        critic_in = torch.cat([scalar, radar_emb], dim=-1)
        c_out = self.critic_backbone(critic_in)
        return self.value_head(c_out)


# ======================================================================
# Factory function: 根据 network_type 创建对应网络
# ======================================================================
def create_network(network_type='nmn'):
    """
    网络工厂函数

    Args:
        network_type: 'nmn' → DefenderNetNMN (底层技能训练)
                      'mlp' → DefenderNetMLP (HRL顶层, CTDE)
    Returns:
        nn.Module
    """
    if network_type == 'nmn':
        return DefenderNetNMN()
    elif network_type == 'mlp':
        return DefenderNetMLP()
    else:
        raise ValueError(f"Unknown network_type: {network_type!r}. "
                         f"Choose 'nmn' or 'mlp'.")
