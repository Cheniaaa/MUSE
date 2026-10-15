"""局部无线观测与逐目标科学评分；登记不加载数值后端。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.geometry import PropagationPathLayout
from muse_sdk.network import SensingFieldView
from muse_sdk.registration import TermRegistration

if TYPE_CHECKING:
    from torch import Tensor


class LocalPathLengthConfig(DataModel):
    sensing_id: str = Field(min_length=1)
    range_scale_m: float = Field(gt=0)


class PositionBoundConfig(DataModel):
    position_bound_limit_m: float = Field(gt=0)


class SelectedLinkFeedbackConfig(DataModel):
    link_id: str = Field(min_length=1)
    rate_scale_bps: float = Field(gt=0)
    power_scale_w: float = Field(gt=0)


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


def position_target_groups(layout: PropagationPathLayout, *, device) -> tuple[Tensor, ...]:
    """按target point聚合关系，不把endpoint或逻辑sensing ID当成target。"""
    import torch
    groups = {}
    for index, relation in enumerate(layout.sensing):
        target = layout.paths[relation.outbound].rx
        groups.setdefault(target, []).append(index)
    return tuple(torch.tensor(indices, dtype=torch.int64, device=device) for indices in groups.values())


def position_bound_violation(view: SensingFieldView, groups: tuple[Tensor, ...], *, limit_m: float) -> Tensor:
    """完整decision的逐目标位置界；缺测、不可逆或超限均视为违反。"""
    import torch
    violated = torch.zeros(view.data.shape[0], dtype=torch.bool, device=view.data.device)
    for indices in groups:
        complete = view.valid[:, indices].all(dim=(1, 2))
        information = view.data[:, indices].sum(dim=(1, 2))
        eigenvalues = torch.linalg.eigvalsh(information[complete])
        positive = ((eigenvalues[:, 1] > 0) &
                    (eigenvalues[:, 0] > 2*torch.finfo(torch.float64).eps*eigenvalues[:, 1]))
        accepted = torch.zeros_like(complete)
        complete_worlds = complete.nonzero().flatten()
        bound = eigenvalues[positive].reciprocal().sum(-1).sqrt()
        accepted[complete_worlds[positive]] = bound <= limit_m
        violated |= ~accepted
    return violated.to(torch.float32)


class PositionBoundViolation:
    def __init__(self, cfg, env):
        self.groups = position_target_groups(env.experiment.geometry.layout, device=env.device)

    def __call__(self, env, *, position_bound_limit_m):
        return position_bound_violation(env.sensing_outputs['fim'], self.groups, limit_m=position_bound_limit_m)


class SelectedLinkFeedback:
    def __init__(self, cfg, env):
        import torch
        network = env.experiment.network
        layout = env.experiment.geometry.layout
        control = next(binding for binding in network.role_controls if binding.role_id == cfg.params['role_id'])
        links = []
        # role control的endpoint顺序已由resolver保证与role slot顺序一致。
        for endpoint in control.endpoint_indices:
            point = network.endpoint_to_point[endpoint]
            matches = [index for index, relation in enumerate(layout.desired)
                       if relation.link_id == cfg.params['link_id'] and layout.paths[relation.path].tx == point]
            if len(matches) != 1:
                raise ValueError('SelectedLinkFeedback requires exactly one matching link per role slot')
            links.append(matches[0])
        self.link_indices = torch.tensor(links, dtype=torch.int64, device=env.device)
        self.endpoint_indices = torch.tensor(control.endpoint_indices, dtype=torch.int64, device=env.device)
        self.control_substeps = env.experiment.scenario.scene.control_substeps

    def __call__(self, env, *, role_id, link_id, rate_scale_bps, power_scale_w):
        import torch
        network = env.network
        transmission = network.simultaneous_transmissions
        selected = (network.allocations.link_active[:, self.link_indices]
                    & transmission.active[:, self.endpoint_indices])
        valid = selected & network.radio.rate.valid[:, self.link_indices]
        rate = torch.where(valid, network.radio.rate.rate_bps[:, self.link_indices], 0.) / rate_scale_bps
        power = torch.where(selected, transmission.power_w[:, self.endpoint_indices], 0.) / power_scale_w
        age = ((env.physics_outputs.episode_decision * self.control_substeps - network.radio.sample_tick)
               / self.control_substeps)
        age = torch.where(valid, age[:, None, None], 0.)
        return torch.stack((rate, valid, power, age), dim=-1).to(torch.float32)


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
    feedback_inputs = tuple(dict(path=path, phase='post_physics', tensor=dict(
        shape=shape, dtype=dtype, units=units, frame='none', rate=rate))
        for path, shape, dtype, units, rate in (
            (['network', 'radio', 'rate', 'rate_bps'], ['B', 'L', 'F'], 'float32', 'bit/s', 'radio_epoch'),
            (['network', 'radio', 'rate', 'valid'], ['B', 'L', 'F'], 'bool', '1', 'radio_epoch'),
            (['network', 'radio', 'sample_tick'], ['B'], 'int64', 'tick', 'radio_epoch'),
            (['network', 'allocations', 'link_active'], ['B', 'L', 'F'], 'bool', '1', 'radio_epoch'),
            (['network', 'simultaneous_transmissions', 'active'], ['B', 'E', 'F'], 'bool', '1', 'radio_epoch'),
            (['network', 'simultaneous_transmissions', 'power_w'], ['B', 'E', 'F'], 'float32', 'W', 'radio_epoch'),
            (['physics_outputs', 'episode_decision'], ['B'], 'int64', '1', 'decision')))
    return {
        'local_path_length': TermRegistration(LocalPathLengthObservation, 'observation', 'role',
            LocalPathLengthConfig, measurement_inputs,
            dict(shape=['B', 'N_role', 3], dtype='float32', units='1', frame='none', rate='decision')),
        'position_bound_violation': TermRegistration(PositionBoundViolation, 'cost', 'world',
            PositionBoundConfig, fim_inputs,
            dict(shape=['B'], dtype='float32', units='1', frame='none', rate='decision'),
            cost_id='position_bound_violation'),
        'selected_link_feedback': TermRegistration(SelectedLinkFeedback, 'observation', 'role',
            SelectedLinkFeedbackConfig, feedback_inputs,
            dict(shape=['B', 'N_role', 'F', 4], dtype='float32', units='1', frame='none', rate='decision')),
    }
