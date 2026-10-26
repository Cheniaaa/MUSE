"""共享CPU轨迹样本；数值oracle由测试显式给定。"""
import numpy as np


def sample_values(fields, *, step=0, terminal=False):
    values = []
    for field in fields:
        value = np.zeros(field.shape, dtype=field.dtype)
        if field.name == 'xpos':
            value[:, 0] = step
            value[:, 2] = 1
        elif field.name == 'xmat':
            value[:] = np.eye(3).reshape(9)
        elif field.name in ('present', 'alive', 'controllable'):
            value[...] = True
        elif field.name == 'action_channel':
            value[...] = -7
        elif field.name == 'world_end_reason':
            value[...] = 2**40 if terminal else 0
        elif field.name == 'world_truncated':
            value[...] = terminal
        elif field.name in ('reward', 'role_cost', 'world_cost', 'action_motion'):
            value[...] = step + .25
        values.append(value)
    return values
