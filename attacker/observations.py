"""Observation projections shared by active Attacker policies."""

from __future__ import annotations

import numpy as np
import torch

from configs.attacker_config import NetParameters


def build_decentralized_actor_observation(privileged_obs):
    """Project the 72-D state into the 70-D attacker-centric actor view."""
    is_tensor = torch.is_tensor(privileged_obs)
    obs = privileged_obs if is_tensor else torch.as_tensor(privileged_obs, dtype=torch.float32)
    if obs.shape[-1] != int(NetParameters.OBS_DIM):
        raise ValueError(
            "privileged attacker observation must have last dimension "
            f"{NetParameters.OBS_DIM}, got {tuple(obs.shape)}"
        )

    scalar = obs[..., :NetParameters.SCALAR_LEN]
    attacker_xy = scalar[..., 0:2]
    defender_world = 0.5 * (scalar[..., 3:5] - attacker_xy)
    target_world = 0.5 * (scalar[..., 6:8] - attacker_xy)
    heading = (scalar[..., 2:3] + 1.0) * np.pi
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)

    def _world_to_body(relative_xy):
        x = relative_xy[..., 0:1]
        y = relative_xy[..., 1:2]
        return torch.cat(
            [
                cos_heading * x + sin_heading * y,
                -sin_heading * x + cos_heading * y,
            ],
            dim=-1,
        )

    defender_body = _world_to_body(defender_world)
    target_body = _world_to_body(target_world)
    defender_dist = torch.linalg.vector_norm(defender_world, dim=-1, keepdim=True) / np.sqrt(2.0)
    target_dist = torch.linalg.vector_norm(target_world, dim=-1, keepdim=True) / np.sqrt(2.0)
    actor_obs = torch.cat(
        [
            defender_body,
            target_body,
            defender_dist,
            target_dist,
            obs[..., NetParameters.SCALAR_LEN:NetParameters.OBS_DIM],
        ],
        dim=-1,
    )
    if is_tensor:
        return actor_obs
    return actor_obs.cpu().numpy().astype(np.float32, copy=False)


__all__ = ["build_decentralized_actor_observation"]
