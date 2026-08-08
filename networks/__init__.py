"""Compatibility facade for network definitions and factory helpers."""

from .classifier_heads import HRLTopSkillClassifier, HRLTopSkillClassifierWithContext
from .factory import create_network
from .hrl_top_dual_gru_raw import DefenderNetHRLTopDualGRURaw
from .hrl_top_gru import DefenderNetHRLTopGRU
from .mlp import DefenderNetMLP
from .mlp_gru import DefenderNetMLPGRU
from .mlp_noctde import DefenderNetMLPNoCTDE
from .nmn import DefenderNetNMN
from .nmn_ctde import DefenderNetNMNCTDE
from .nmn_ctde_shared import DefenderNetNMNCTDEShared
from .nmn_ctde_task_shared import DefenderNetNMNCTDETaskShared
from .nmn_ctde_task_shared_distill import DefenderNetNMNCTDETaskSharedDistill
from .nmn_dual_gru_raw import DefenderNetNMNDualGRURaw
from .nmn_dual_gru_raw_ctde import DefenderNetNMNDualGRURawCTDE
from .nmn_gru import DefenderNetNMNGRU
from .nmn_no_shared_radar import DefenderNetNMNNoSharedRadar
from .radar_encoder import RadarEncoder

__all__ = [
    "RadarEncoder",
    "DefenderNetMLP",
    "DefenderNetMLPNoCTDE",
    "DefenderNetMLPGRU",
    "DefenderNetNMN",
    "DefenderNetNMNCTDE",
    "DefenderNetNMNCTDEShared",
    "DefenderNetNMNCTDETaskShared",
    "DefenderNetNMNCTDETaskSharedDistill",
    "DefenderNetNMNDualGRURaw",
    "DefenderNetNMNDualGRURawCTDE",
    "DefenderNetNMNGRU",
    "DefenderNetNMNNoSharedRadar",
    "DefenderNetHRLTopGRU",
    "DefenderNetHRLTopDualGRURaw",
    "HRLTopSkillClassifier",
    "HRLTopSkillClassifierWithContext",
    "create_network",
]
