"""评分/终止只读结果；服务状态由WirelessRuntime唯一推进。"""
from muse_sdk.config import DataModel
from muse_sdk.registration import TermRegistration


class Config(DataModel):
    pass


def radio_rate(env):
    import torch
    rate = env.network.radio.rate
    return torch.where(rate.valid, rate.rate_bps, 0.).sum(dim=(1, 2))


def upload_done(env):
    return (env.network.services['upload'].remaining_bits <= 0).all(-1)


def registrations():
    return {'radio_rate': TermRegistration(radio_rate, 'reward', 'world', Config,
        tuple(dict(path=['network', 'radio', 'rate', name], phase='post_physics', tensor=dict(
            shape=['B', 'L', 'F'], dtype=dtype, units=units, frame='none', rate='radio_epoch'))
            for name, dtype, units in [('rate_bps', 'float32', 'bit/s'), ('valid', 'bool', '1')]),
        dict(shape=['B'], dtype='float32', units='bit/s', frame='none', rate='decision')),
        'upload_done': TermRegistration(upload_done, 'termination', 'world', Config,
            (dict(path=['network', 'services', 'upload', 'remaining_bits'], phase='post_physics', tensor=dict(
                shape=['B', 'N_service_links'], dtype='float64', units='bit', frame='none', rate='decision')),),
            dict(shape=['B'], dtype='bool', units='1', frame='none', rate='decision'))}
