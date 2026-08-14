"""CoppeliaSim-native TrackMaker interfaces with lazy optional imports."""

from .skill_adapter import (
    CoppeliaSkillObservationProvider,
    NormalizedCmdVelAdapter,
    build_critic_observation,
    compute_gae,
    load_skill_model,
)


def __getattr__(name):
    """Avoid loading Gym/Coppelia dependencies in the ROS policy process."""
    if name in {"CoppeliaTrackEnv", "VelocityCommand"}:
        from .track_env import CoppeliaTrackEnv, VelocityCommand

        return {"CoppeliaTrackEnv": CoppeliaTrackEnv, "VelocityCommand": VelocityCommand}[name]
    raise AttributeError(name)

__all__ = [
    "CoppeliaSkillObservationProvider",
    "CoppeliaTrackEnv",
    "NormalizedCmdVelAdapter",
    "VelocityCommand",
    "build_critic_observation",
    "compute_gae",
    "load_skill_model",
]
