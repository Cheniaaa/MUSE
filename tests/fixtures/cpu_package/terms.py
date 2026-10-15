from muse_sdk.config import DataModel
from pydantic import Field


class EmptyConfig(DataModel):
    pass


class ProgressConfig(DataModel):
    scale: float = Field(gt=0)


class LimitConfig(DataModel):
    episode_max_steps: int = Field(gt=0)


class ProgressReward:
    def __init__(self, cfg, env):
        self.cfg = cfg

    def __call__(self, value, *, role_id, scale):
        from .deferred import transform
        return transform(value) * scale


def collision(env):
    return False


def time_limit(env, *, episode_max_steps):
    return False
