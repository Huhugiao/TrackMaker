"""Skill-level PPO package."""

from configs.skill_config import SetupParameters, TrainingParameters, NetParameters, RecordingParameters
from networks import DefenderNetMLP, DefenderNetNMN, RadarEncoder, create_network
from skill.model import Model
from skill.util import set_global_seeds, write_to_tensorboard, make_gif


def __getattr__(name):
    """Keep training-only Ray imports out of lightweight deployment processes."""
    if name == 'Runner':
        from skill.runner import Runner
        return Runner
    raise AttributeError(name)

__all__ = [
    'SetupParameters',
    'TrainingParameters', 
    'NetParameters',
    'RecordingParameters',
    'DefenderNetMLP',
    'DefenderNetNMN',
    'create_network',
    'RadarEncoder',
    'Model',
    'Runner',
    'set_global_seeds',
    'write_to_tensorboard',
    'make_gif'
]
