"""NMN CTDE task-shared network with privileged latent distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.skill_config import NetParameters, TrainingParameters

from .nmn_ctde_task_shared import DefenderNetNMNCTDETaskShared


class DefenderNetNMNCTDETaskSharedDistill(DefenderNetNMNCTDETaskShared):
    """Task-shared NMN with a belief head that predicts privileged latent features."""

    def __init__(self):
        super().__init__()
        merged_dim = int(NetParameters.NMN_MERGED_DIM)
        privileged_branch_dim = int(NetParameters.NMN_BRANCH_DIM)
        self.belief_predictor = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.Tanh(),
            nn.Linear(merged_dim, privileged_branch_dim),
            nn.Tanh(),
        )
        self.belief_predictor.apply(self._init_weights)
        self.auxiliary_loss_coef = float(getattr(TrainingParameters, 'AUX_DISTILL_COEF', 0.1))
        self.auxiliary_loss_name = 'Privileged_Latent_Distill'

    def _compute_actor_feature(self, actor_obs: torch.Tensor) -> torch.Tensor:
        actor_scalar, actor_radar_emb = self._encode_actor_obs(actor_obs)
        actor_tracking_feat, actor_obstacle_feat = self._extract_task_features(actor_scalar, actor_radar_emb)
        return self.actor_merge_layer(torch.cat([actor_tracking_feat, actor_obstacle_feat], dim=-1))

    def _compute_privileged_latent(self, critic_obs: torch.Tensor) -> torch.Tensor:
        attacker_scalar, attacker_radar_emb = self._encode_privileged_obs(critic_obs)
        privileged_input = torch.cat([attacker_scalar, attacker_radar_emb], dim=-1)
        return self.critic_privileged_branch(privileged_input)

    def compute_auxiliary_loss(self, actor_obs, critic_obs, mask=None):
        actor_feat = self._compute_actor_feature(actor_obs)
        pred_latent = self.belief_predictor(actor_feat)
        target_latent = self._compute_privileged_latent(critic_obs).detach()
        per_sample_loss = F.mse_loss(pred_latent, target_latent, reduction='none').mean(dim=-1)
        if mask is None:
            return per_sample_loss.mean()
        mask = mask.reshape(-1).to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        return (per_sample_loss * mask).sum() / mask.sum().clamp_min(1.0)


__all__ = ["DefenderNetNMNCTDETaskSharedDistill"]
