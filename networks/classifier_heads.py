"""HRL classifier heads."""

import torch
import torch.nn as nn


class HRLTopSkillClassifier(nn.Module):
    """2-layer MLP skill classifier with dropout and temporal statistics."""

    def __init__(self, hidden_dim: int, num_skills: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temporal_proj = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, int(num_skills)),
        )
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.temporal_proj.weight, gain=1.0)
        nn.init.constant_(self.temporal_proj.bias, 0.0)

    def forward(self, x):
        if x.dim() == 2:
            return self.classifier(self.drop(x))
        seq_mean = x.mean(dim=1, keepdim=True).expand_as(x)
        seq_std = x.std(dim=1, keepdim=True).clamp_min(1e-6).expand_as(x)
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        delta_mean = delta.mean(dim=1, keepdim=True).expand_as(x)
        enriched = torch.cat([x, seq_mean, seq_std, delta_mean], dim=-1)
        projected = torch.relu(self.temporal_proj(enriched))
        return self.classifier(self.drop(projected))


class HRLTopSkillClassifierWithContext(nn.Module):
    """Fuse actor hidden state with behavior context before skill classification."""

    def __init__(self, hidden_dim: int, context_dim: int, num_skills: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.context_proj = nn.Linear(self.context_dim, self.hidden_dim // 2)
        self.fuse_proj = nn.Linear(self.hidden_dim + self.hidden_dim // 2, self.hidden_dim)
        self.base_classifier = HRLTopSkillClassifier(self.hidden_dim, num_skills)
        nn.init.orthogonal_(self.context_proj.weight, gain=1.0)
        nn.init.constant_(self.context_proj.bias, 0.0)
        nn.init.orthogonal_(self.fuse_proj.weight, gain=1.0)
        nn.init.constant_(self.fuse_proj.bias, 0.0)

    def forward(self, actor_feat, behavior_context):
        if behavior_context is None:
            return self.base_classifier(actor_feat)
        if actor_feat.dim() != behavior_context.dim():
            raise ValueError(
                f"actor/context rank mismatch: {actor_feat.dim()} vs {behavior_context.dim()}"
            )
        if actor_feat.shape[:-1] != behavior_context.shape[:-1]:
            raise ValueError(
                f"actor/context shape mismatch: {tuple(actor_feat.shape)} vs {tuple(behavior_context.shape)}"
            )
        ctx = torch.tanh(self.context_proj(behavior_context))
        fused = torch.cat([actor_feat, ctx], dim=-1)
        fused = torch.tanh(self.fuse_proj(fused))
        return self.base_classifier(fused)


__all__ = ["HRLTopSkillClassifier", "HRLTopSkillClassifierWithContext"]
