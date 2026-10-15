"""只读Task字段；无线history由selection manager拥有。"""
from muse_sdk.config import DataModel
from muse_sdk.registration import TermRegistration

from .task import tensor


class EmptyConfig(DataModel):
    pass


def mission_reward(env):
    return env.task.reward


def team_reward(env, *, role_id):
    return env.task.reward[:, None].expand(-1, env.experiment.bindings[role_id].role_slot_count)


def remaining_time(env, *, role_id):
    return env.task.remaining_fraction[:, None].expand(-1, env.experiment.bindings[role_id].role_slot_count, -1)


def mission_done(env):
    return env.task.done


def registrations():
    terms = {}
    for name, implementation, kind, scope, field, output, dtype in (
        ('mission_reward', mission_reward, 'reward', 'world', 'reward', ['B'], 'float32'),
        ('team_reward', team_reward, 'reward', 'role', 'reward', ['B', 'N_role'], 'float32'),
        ('remaining_time', remaining_time, 'observation', 'role', 'remaining_fraction', ['B', 'N_role', 1], 'float32'),
        ('mission_done', mission_done, 'termination', 'world', 'done', ['B'], 'bool')):
        terms[name] = TermRegistration(implementation, kind, scope, EmptyConfig,
            (dict(path=['task', field], phase='post_task', tensor=tensor(['B', 1] if field == 'remaining_fraction' else ['B'], dtype=dtype)),),
            tensor(output, dtype=dtype))
    return terms
