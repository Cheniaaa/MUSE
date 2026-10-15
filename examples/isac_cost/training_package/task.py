"""ISAC有限任务的唯一计分入口；无线科学判据由builtin拥有。"""
from pydantic import Field

from muse_builtins.wireless_terms import position_bound_violation, position_target_groups, registrations
from muse_sdk.config import DataModel
from muse_sdk.runtime_types import TaskUpdateResult


class Workspace(DataModel):
    minimum_m: list[float] = Field(min_length=3, max_length=3)
    maximum_m: list[float] = Field(min_length=3, max_length=3)


class TaskConfig(DataModel):
    horizon_decisions: int = Field(gt=0)
    workspace_by_role: dict[str, Workspace]
    service_bits_scale: float = Field(gt=0)
    rf_energy_scale_j: float = Field(gt=0)
    rf_penalty_weight: float = Field(gt=0)
    position_bound_limit_m: float = Field(gt=0)


def tensor(shape, *, dtype='float32', units='1', frame='none', rate='decision'):
    return dict(shape=shape, dtype=dtype, units=units, frame=frame, rate=rate)


class IsacTask:
    @classmethod
    def describe(cls):
        inputs = [dict(path=['physics_outputs', 'episode_decision'], phase='post_physics',
                       tensor=tensor(['B'], dtype='int64'))]
        for role in ('alpha', 'beta'):
            inputs.extend((
                dict(path=['role_positions_world', role], phase='post_physics',
                     tensor=tensor(['B', 'N_role', 3], units='m', frame='world_z_up')),
                dict(path=['physics_outputs', 'crashed', role], phase='post_physics',
                     tensor=tensor(['B', 'N_role'], dtype='bool'))))
        for service in ('alpha_upload', 'beta_upload'):
            for name, dtype, units in (('served_bits', 'float64', 'bit'), ('valid', 'bool', '1')):
                inputs.append(dict(path=['network', 'services', service, name], phase='post_physics',
                                   tensor=tensor(['B', 'N_service_links'], dtype=dtype, units=units)))
        inputs.append(dict(path=['network', 'simultaneous_transmissions', 'power_w'], phase='post_physics',
                           tensor=tensor(['B', 'E', 'F'], units='W', rate='radio_epoch')))
        inputs.extend(registrations()['position_bound_violation'].input_fields)
        return dict(task_id='isac_task', config_schema=TaskConfig, input_fields=inputs,
                    published_fields=dict(reward=tensor(['B']), done=tensor(['B'], dtype='bool'),
                                          remaining_fraction=tensor(['B', 1]), mission_cost=tensor(['B'])),
                    role_status_bindings={}, learning_done_roles=[])

    def __init__(self, config: dict, env):
        import torch

        self.config = TaskConfig.model_validate(config)
        roles = {'alpha': ('alpha_upload', 'alpha_up'), 'beta': ('beta_upload', 'beta_up')}
        if set(self.config.workspace_by_role) != set(roles) or set(env.experiment.bindings) != set(roles):
            raise ValueError('workspace roles must match the ISAC composition')
        self.workspace = {}
        self.services = []
        endpoints = set()
        network = env.experiment.network
        for role, (service_id, link_id) in roles.items():
            box = self.config.workspace_by_role[role]
            if any(low >= high for low, high in zip(box.minimum_m, box.maximum_m, strict=True)):
                raise ValueError('workspace minimum_m must be below maximum_m')
            self.workspace[role] = (torch.tensor(box.minimum_m, device=env.device),
                                    torch.tensor(box.maximum_m, device=env.device))
            owned = next(control.endpoint_indices for control in network.role_controls if control.role_id == role)
            endpoints.update(owned)
            matches = [allocation.link_index for allocation in network.communication_allocations
                       if allocation.tx in owned and
                       env.experiment.geometry.layout.desired[allocation.link_index].link_id == link_id]
            service = next((service for service in network.services if service.service_id == service_id), None)
            if (service is None or len(matches) != len(owned) or set(service.link_indices) != set(matches)):
                raise ValueError('service must bind the declared upload link for every owned endpoint')
            self.services.append(service_id)
        self.endpoints = torch.tensor(sorted(endpoints), dtype=torch.int64, device=env.device)
        self.decision_dt = env.experiment.scenario.scene.physics_dt * env.experiment.scenario.scene.control_substeps
        self.gamma = env.experiment.algorithm_config['gamma']
        cost = env.experiment.costs.world.get('mission_position_violation')
        if cost is not None and cost.discount != self.gamma:
            raise ValueError('mission cost discount must equal algorithm gamma')
        self.failure_amplitude = (self.config.rf_penalty_weight * self.decision_dt *
                                  sum(network.endpoint_limits[index].max_power_w for index in endpoints) /
                                  self.config.rf_energy_scale_j)
        self.target_groups = position_target_groups(env.experiment.geometry.layout, device=env.device)
        self.reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self.mission_cost = torch.zeros_like(self.reward)
        self.done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.remaining_fraction = torch.ones((env.num_envs, 1), dtype=torch.float32, device=env.device)

    def reset(self, env_ids):
        self.reward[env_ids] = 0
        self.mission_cost[env_ids] = 0
        self.done[env_ids] = False
        self.remaining_fraction[env_ids] = 1

    def update(self, env):
        import torch

        decision = env.physics_outputs.episode_decision
        served = torch.stack([env.network.services[name].served_bits.sum(-1) for name in self.services]).sum(0)
        valid = torch.stack([env.network.services[name].valid.all(-1) for name in self.services]).all(0)
        energy = env.network.simultaneous_transmissions.power_w[:, self.endpoints].sum((1, 2)) * self.decision_dt
        failed = torch.zeros_like(self.done)
        for role, (minimum, maximum) in self.workspace.items():
            position = env.role_positions_world[role]
            failed |= env.physics_outputs.crashed[role].any(-1) | ((position < minimum) | (position > maximum)).any((1, 2))
        remaining = self.config.horizon_decisions - decision + 1
        tail = remaining if self.gamma == 1 else (1 - self.gamma ** remaining) / (1 - self.gamma)
        normal_reward = served / self.config.service_bits_scale - self.config.rf_penalty_weight * energy / self.config.rf_energy_scale_j
        self.reward.copy_(torch.where(failed, -self.failure_amplitude * tail, normal_reward))
        violation = position_bound_violation(env.sensing_outputs['fim'], self.target_groups,
                                             limit_m=self.config.position_bound_limit_m)
        self.mission_cost.copy_(torch.where(failed, tail, violation))
        self.done.copy_(failed | (decision >= self.config.horizon_decisions))
        self.remaining_fraction.copy_((self.config.horizon_decisions - decision).clamp_min(0)[:, None] /
                                      self.config.horizon_decisions)
        return TaskUpdateResult(role_status={}, learning_done={}, diagnostics=dict(
            episode_decision=decision, served_bits=served, service_valid=valid, rf_energy_j=energy,
            failed=failed, task_done=self.done, task_reward=self.reward,
            position_violation=violation, mission_cost=self.mission_cost))
