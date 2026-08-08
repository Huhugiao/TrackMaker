"""Shared network helpers."""

import torch
import torch.nn as nn

from configs.skill_config import NetParameters


def _unroll_gru_with_dones(
    gru: nn.GRU,
    seq_input: torch.Tensor,
    dones: torch.Tensor,
    initial_hidden: torch.Tensor = None,
) -> torch.Tensor:
    """Unroll a GRU and reset hidden state after done steps."""
    bsz, tlen, _ = seq_input.shape
    outputs = []
    hidden = initial_hidden
    dones = dones.float()
    for t in range(tlen):
        if t > 0 and hidden is not None:
            keep_mask = (1.0 - dones[:, t - 1]).view(1, bsz, 1)
            hidden = hidden * keep_mask
        out_t, hidden = gru(seq_input[:, t:t + 1, :], hidden)
        outputs.append(out_t)
    return torch.cat(outputs, dim=1)


class _NMNStageMixin:
    """Utility mixin for NMN-CL stage-dependent obstacle inputs."""

    def _init_nmn_stage_control(self):
        self._nmn_stage = 2
        self._nmn_dummy_obs_value = 1.0

    def set_nmn_stage(self, stage: int):
        stage_idx = int(stage)
        if stage_idx not in (1, 2):
            raise ValueError(f"NMN stage must be 1 or 2, got {stage!r}")
        self._nmn_stage = stage_idx

    def get_nmn_stage(self) -> int:
        return int(getattr(self, "_nmn_stage", 2))

    def _use_dummy_obstacle_input(self) -> bool:
        return self.get_nmn_stage() == 1

    def _prepare_nmn_radar(self, radar: torch.Tensor) -> torch.Tensor:
        if not self._use_dummy_obstacle_input():
            return radar
        return torch.ones_like(radar) * float(self._nmn_dummy_obs_value)


def _mark_hrl_top_discrete_policy(net: nn.Module) -> nn.Module:
    net.is_discrete_policy = True
    net.discrete_action_dim = int(
        getattr(
            NetParameters,
            "HRL_TOP_DISCRETE_ACTION_DIM",
            getattr(NetParameters, "HRL_NUM_SKILLS", 2),
        )
    )
    if not hasattr(net, "hrl_top_marker"):
        net.register_buffer("hrl_top_marker", torch.ones((), dtype=torch.int8))
    if not hasattr(net, "discrete_policy_marker"):
        net.register_buffer("discrete_policy_marker", torch.ones((), dtype=torch.int8))
    return net
