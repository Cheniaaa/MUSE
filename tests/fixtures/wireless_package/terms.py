from muse_sdk.config import DataModel
from muse_sdk.registration import TermRegistration


class Config(DataModel):
    pass


def upload_done(env):
    return (env.network.services['upload'].remaining_bits <= 0).all(-1)


def radio_valid(env):
    return env.network.radio.rate.valid.any(dim=(1, 2)).float()


def sensing_valid(env):
    return env.sensing_outputs['measurement'].valid.flatten(1).all(-1).float()


def registrations():
    return {'upload_done': TermRegistration(upload_done, 'termination', 'world', Config,
        (dict(path=['network', 'services', 'upload', 'remaining_bits'], phase='post_physics',
            tensor=dict(shape=['B', 'N_service_links'], dtype='float64', units='bit', frame='none', rate='decision')),),
        dict(shape=['B'], dtype='bool', units='1', frame='none', rate='decision')),
        'radio_valid': TermRegistration(radio_valid, 'reward', 'world', Config,
            (dict(path=['network', 'radio', 'rate', 'valid'], phase='post_physics', tensor=dict(
                shape=['B', 'L', 'F'], dtype='bool', units='1', frame='none', rate='radio_epoch')),),
            dict(shape=['B'], dtype='float32', units='1', frame='none', rate='decision')),
        'sensing_valid': TermRegistration(sensing_valid, 'reward', 'world', Config,
            (dict(path=['sensing_outputs', 'measurement', 'valid'], phase='post_physics', tensor=dict(
                shape=['B', 'S', 2], dtype='bool', units='1', frame='none', rate='radio_epoch')),),
            dict(shape=['B'], dtype='float32', units='1', frame='none', rate='decision'))}
