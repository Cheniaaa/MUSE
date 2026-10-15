"""测量局部可见；位置界只作为完整decision的信息评分。"""
from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.registration import TermRegistration


class EmptyConfig(DataModel):
    pass


class LocalPathLengthConfig(DataModel):
    sensing_id: str = Field(min_length=1)
    range_scale_m: float = Field(gt=0)


class PositionBoundConfig(DataModel):
    position_bound_limit_m: float = Field(gt=0)


class LocalPathLengthObservation:
    def __init__(self, cfg, env):
        import torch
        binding = env.experiment.bindings[cfg.params['role_id']]
        layout = env.experiment.geometry.layout
        indices = []
        for slot in binding.role_slot_to_group_slot:
            matches = []
            for index, relation in enumerate(layout.sensing):
                outbound, inbound = layout.paths[relation.outbound], layout.paths[relation.inbound]
                point = layout.points[outbound.tx]
                if (relation.sensing_id == cfg.params['sensing_id'] and outbound.tx == inbound.rx
                        and point.group_id == binding.group_id and point.slot == slot):
                    matches.append(index)
            if len(matches) != 1:
                raise ValueError('LocalPathLengthObservation requires exactly one monostatic relation per role slot')
            indices.append(matches[0])
        self.indices = torch.tensor(indices, dtype=torch.int64, device=env.device)
        self.control_substeps = env.experiment.scenario.scene.control_substeps

    def __call__(self, env, *, role_id, sensing_id, range_scale_m):
        import torch
        view = env.sensing_outputs['measurement']
        valid = view.valid[:, self.indices]
        sample = torch.where(valid, view.data[:, self.indices], 0.).sum(-1) / range_scale_m
        window = ((view.interval_end_tick - view.interval_start_tick) / self.control_substeps)
        window = window[:, None].expand_as(sample)
        return torch.stack((sample, valid.any(-1), window), dim=-1).to(torch.float32)


class PositionBoundViolation:
    def __init__(self, cfg, env):
        import torch
        layout = env.experiment.geometry.layout
        groups = {}
        for index, relation in enumerate(layout.sensing):
            target = layout.paths[relation.outbound].rx
            groups.setdefault(target, []).append(index)
        self.groups = tuple(torch.tensor(indices, dtype=torch.int64, device=env.device)
                            for indices in groups.values())

    def __call__(self, env, *, position_bound_limit_m):
        import torch
        view = env.sensing_outputs['fim']
        violated = torch.zeros(view.data.shape[0], dtype=torch.bool, device=view.data.device)
        for indices in self.groups:
            complete = view.valid[:, indices].all(dim=(1, 2))
            information = view.data[:, indices].sum(dim=(1, 2))
            eigenvalues = torch.linalg.eigvalsh(information[complete])
            positive = ((eigenvalues[:, 1] > 0) &
                        (eigenvalues[:, 0] > 2*torch.finfo(torch.float64).eps*eigenvalues[:, 1]))
            accepted = torch.zeros_like(complete)
            complete_worlds = complete.nonzero().flatten()
            bound = eigenvalues[positive].reciprocal().sum(-1).sqrt()
            accepted[complete_worlds[positive]] = bound <= position_bound_limit_m
            violated |= ~accepted
        return violated.to(torch.float32)


def radio_rate(env):
    import torch
    rate = env.network.radio.rate
    return torch.where(rate.valid, rate.rate_bps, 0.).sum(dim=(1, 2))


def registrations():
    measurement_inputs = tuple(dict(path=['sensing_outputs', 'measurement', name], phase='post_physics',
        tensor=dict(shape=shape, dtype=dtype, units=units, frame='none', rate=rate))
        for name, shape, dtype, units, rate in (
            ('data', ['B', 'S', 'F'], 'float64', 'm', 'radio_epoch'),
            ('valid', ['B', 'S', 'F'], 'bool', '1', 'radio_epoch'),
            ('interval_start_tick', ['B'], 'int64', 'tick', 'decision'),
            ('interval_end_tick', ['B'], 'int64', 'tick', 'decision')))
    fim_inputs = (
        dict(path=['sensing_outputs', 'fim', 'data'], phase='post_physics', tensor=dict(
            shape=['B', 'S', 'F', 2, 2], dtype='float64', units='m^-2', frame='world_z_up', rate='radio_epoch')),
        dict(path=['sensing_outputs', 'fim', 'valid'], phase='post_physics', tensor=dict(
            shape=['B', 'S', 'F'], dtype='bool', units='1', frame='none', rate='radio_epoch')))
    rate_inputs = tuple(dict(path=['network', 'radio', 'rate', name], phase='post_physics', tensor=dict(
        shape=['B', 'L', 'F'], dtype=dtype, units=units, frame='none', rate='radio_epoch'))
        for name, dtype, units in [('rate_bps', 'float32', 'bit/s'), ('valid', 'bool', '1')])
    return {
        'local_path_length': TermRegistration(LocalPathLengthObservation, 'observation', 'role',
            LocalPathLengthConfig, measurement_inputs,
            dict(shape=['B', 'N_role', 3], dtype='float32', units='1', frame='none', rate='decision')),
        'position_bound_violation': TermRegistration(PositionBoundViolation, 'cost', 'world',
            PositionBoundConfig, fim_inputs,
            dict(shape=['B'], dtype='float32', units='1', frame='none', rate='decision'),
            cost_id='position_bound_violation'),
        'radio_rate': TermRegistration(radio_rate, 'reward', 'world', EmptyConfig, rate_inputs,
            dict(shape=['B'], dtype='float32', units='bit/s', frame='none', rate='decision')),
    }
