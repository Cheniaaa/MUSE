"""参考规则的通信速率奖励；科学消费者由builtin提供。"""
from muse_sdk.config import DataModel
from muse_sdk.registration import TermRegistration


class EmptyConfig(DataModel):
    pass


def radio_rate(env):
    import torch
    rate = env.network.radio.rate
    return torch.where(rate.valid, rate.rate_bps, 0.).sum(dim=(1, 2))


def registrations():
    rate_inputs = tuple(dict(path=['network', 'radio', 'rate', name], phase='post_physics', tensor=dict(
        shape=['B', 'L', 'F'], dtype=dtype, units=units, frame='none', rate='radio_epoch'))
        for name, dtype, units in [('rate_bps', 'float32', 'bit/s'), ('valid', 'bool', '1')])
    return {
        'radio_rate': TermRegistration(radio_rate, 'reward', 'world', EmptyConfig, rate_inputs,
            dict(shape=['B'], dtype='float32', units='bit/s', frame='none', rate='decision')),
    }
