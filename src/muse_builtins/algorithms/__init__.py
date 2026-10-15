"""内置学习算法；CPU metadata 不加载学习框架。"""
"""内置算法的唯一 CPU 轻量登记入口。"""


def registrations():
    from .ippo import IPPO
    from .mappo import MAPPO
    from ..rules import ConstantVelocity
    return {'ippo': IPPO, 'mappo': MAPPO, 'constant_velocity': ConstantVelocity}
