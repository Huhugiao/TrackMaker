"""Network factory."""

from configs.skill_config import NetParameters

from .common import _mark_hrl_top_discrete_policy
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


def create_network(network_type: str = "nmn"):
    network_type = str(network_type).strip().lower()
    if network_type == "nmn":
        return DefenderNetNMN()
    if network_type == "nmn_ctde":
        return DefenderNetNMNCTDE()
    if network_type == "nmn_ctde_shared":
        return DefenderNetNMNCTDEShared()
    if network_type == "nmn_ctde_task_shared":
        return DefenderNetNMNCTDETaskShared()
    if network_type == "nmn_ctde_task_shared_distill":
        return DefenderNetNMNCTDETaskSharedDistill()
    if network_type == "nmn_dual_gru_raw":
        return DefenderNetNMNDualGRURaw()
    if network_type == "nmn_dual_gru_raw_ctde":
        return DefenderNetNMNDualGRURawCTDE()
    if network_type == "nmn_gru":
        return DefenderNetNMNGRU()
    if network_type == "nmn_no_shared_radar":
        return DefenderNetNMNNoSharedRadar()
    if network_type == "mlp":
        return DefenderNetMLP()
    if network_type == "mlp_ctde":
        return DefenderNetMLP()
    if network_type == "mlp_gru":
        return DefenderNetMLPGRU()
    if network_type == "mlp_noctde":
        return DefenderNetMLPNoCTDE()
    if network_type == "hrl_top":
        return _mark_hrl_top_discrete_policy(
            DefenderNetMLP(action_dim=int(getattr(NetParameters, "HRL_TOP_ACTION_DIM", 3)))
        )
    if network_type == "hrl_top_noctde":
        return _mark_hrl_top_discrete_policy(
            DefenderNetMLPNoCTDE(action_dim=int(getattr(NetParameters, "HRL_TOP_ACTION_DIM", 3)))
        )
    if network_type == "hrl_top_gru":
        return _mark_hrl_top_discrete_policy(
            DefenderNetHRLTopGRU(action_dim=int(getattr(NetParameters, "HRL_TOP_ACTION_DIM", 3)))
        )
    if network_type == "hrl_top_dual_gru_raw":
        return _mark_hrl_top_discrete_policy(
            DefenderNetHRLTopDualGRURaw(action_dim=int(getattr(NetParameters, "HRL_TOP_ACTION_DIM", 3)))
        )
    raise ValueError(
        "Unknown network_type: "
        f"{network_type!r}. Choose 'nmn', 'nmn_ctde', 'nmn_ctde_shared', 'nmn_ctde_task_shared', 'nmn_ctde_task_shared_distill', "
        "'nmn_no_shared_radar', 'nmn_dual_gru_raw', 'nmn_dual_gru_raw_ctde', "
        "'nmn_gru', 'mlp', 'mlp_ctde', "
        "'mlp_gru', 'mlp_noctde', "
        "'hrl_top', 'hrl_top_noctde', 'hrl_top_gru', or 'hrl_top_dual_gru_raw'."
    )
