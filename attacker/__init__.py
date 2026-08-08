"""Attacker training and deployment package with lazy public imports."""

__all__ = [
    "AttackerMultiStylePPO",
    "MultiStylePPORunner",
    "LearnedAttackerPolicy",
    "SUPPORTED_DEFENDER_STRATEGIES",
]


def __getattr__(name):
    if name == "AttackerMultiStylePPO":
        from .ppo_model import AttackerMultiStylePPO

        return AttackerMultiStylePPO
    if name == "MultiStylePPORunner":
        from .ppo_runner import MultiStylePPORunner

        return MultiStylePPORunner
    if name == "LearnedAttackerPolicy":
        from .learned_policy import LearnedAttackerPolicy

        return LearnedAttackerPolicy
    if name == "SUPPORTED_DEFENDER_STRATEGIES":
        from .defender_pool import SUPPORTED_DEFENDER_STRATEGIES

        return SUPPORTED_DEFENDER_STRATEGIES
    raise AttributeError(name)
