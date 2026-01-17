import torch
import torch.nn as nn
import numpy as np
from mlp.alg_parameters_mlp import NetParameters

class RadarEncoder(nn.Module):
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

class ProtectingNetMLP(nn.Module):
    def __init__(self):
        super(ProtectingNetMLP, self).__init__()
        
        self.hidden_dim = NetParameters.HIDDEN_DIM
        self.num_layers = getattr(NetParameters, 'NUM_HIDDEN_LAYERS', 3)
        
        # Radar Encoder
        self.radar_encoder = RadarEncoder()
        
        # --- Actor Network ---
        # Input: Scalar (8) + Embedded Radar (8) = 16 (Encoded)
        # Note: The backbone takes the ENCODED vector, but the forward method takes the RAW vector and encodes it.
        self.actor_backbone = self._build_mlp(NetParameters.ACTOR_VECTOR_LEN, self.hidden_dim, self.num_layers)

        self.policy_mean = nn.Linear(self.hidden_dim, NetParameters.ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(NetParameters.ACTION_DIM))

        # --- Critic Network (CTDE) ---
        # Input: Tracker (16) + Target (16) = 32 (Encoded)
        self.critic_backbone = self._build_mlp(NetParameters.CRITIC_VECTOR_LEN, self.hidden_dim, self.num_layers)
        
        self.value_head = nn.Linear(self.hidden_dim, 1)
        

            
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, actor_obs, critic_obs):
        # actor_obs: [Batch, 72] -> Scalar(8) + Radar(64)
        # critic_obs: [Batch, 144] -> Tracker(72) + Target(72)

        # 1. Process Actor Radar
        actor_scalar = actor_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        actor_radar = actor_obs[:, NetParameters.ACTOR_SCALAR_LEN:]
        actor_radar_emb = self.radar_encoder(actor_radar)
        actor_in = torch.cat([actor_scalar, actor_radar_emb], dim=-1) # [Batch, 16]

        # 2. Process Critic Radar
        # Tracker Part (0 to 72)
        tracker_end = NetParameters.ACTOR_RAW_LEN
        tracker_scalar = critic_obs[:, :NetParameters.ACTOR_SCALAR_LEN]
        tracker_radar = critic_obs[:, NetParameters.ACTOR_SCALAR_LEN:tracker_end]
        tracker_radar_emb = self.radar_encoder(tracker_radar) # Share encoder
        tracker_part = torch.cat([tracker_scalar, tracker_radar_emb], dim=-1) # [Batch, 16]

        # Target Part (72 to 144)
        target_start = tracker_end
        target_scalar = critic_obs[:, target_start:target_start+NetParameters.PRIVILEGED_SCALAR_LEN]
        target_radar = critic_obs[:, target_start+NetParameters.PRIVILEGED_SCALAR_LEN:]
        target_radar_emb = self.radar_encoder(target_radar) # Share encoder
        target_part = torch.cat([target_scalar, target_radar_emb], dim=-1) # [Batch, 16]

        critic_in = torch.cat([tracker_part, target_part], dim=-1) # [Batch, 32]

        # --- Actor Forward ---
        a_out = self.actor_backbone(actor_in)
        mean = self.policy_mean(a_out)
        log_std = self.log_std.expand_as(mean)
        
        # --- Critic Forward ---
        c_out = self.critic_backbone(critic_in)
        value = self.value_head(c_out)
        
        return mean, value, log_std
