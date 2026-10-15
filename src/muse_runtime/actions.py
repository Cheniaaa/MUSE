from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from muse_builtins.controllers import DiffDriveController, VelocityController, VelocityControllerConfig
from muse_runtime.scene import entity_name
from muse_sdk.runtime_types import RadioAction
from .action_codec import encode_role_action


def encode_actions(experiment, role_actions, device):
    if set(role_actions) != set(experiment.action_specs):
        raise ValueError('Action roles do not match the resolved layout')
    encoded = {role: encode_role_action(spec, role_actions[role], device)
               for role, spec in experiment.action_specs.items()}
    pieces = []
    for entry in experiment.action_layout:
        spec = experiment.action_specs[entry.role_id]
        value = encoded[entry.role_id]
        value = value[..., :spec.shape[-1]] if entry.field == 'motion' else value[..., spec.shape[-1]:]
        pieces.append(value.flatten(1))
    return torch.cat(pieces, dim=-1)


@dataclass(kw_only=True)
class MotionTermCfg(ActionTermCfg):
    group_id: str

    def build(self, env):
        return MotionTerm(self, env)


class MotionTerm(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.group = next(g for g in env.experiment.scenario.entity_groups if g.group_id == cfg.group_id)
        self.entries = [e for e in env.experiment.action_layout
                        if e.field == 'motion' and env.experiment.bindings[e.role_id].group_id == cfg.group_id]
        self._raw = torch.zeros((env.num_envs, sum(e.stop-e.start for e in self.entries)), device=env.device)
        spec = env.experiment.action_specs[self.entries[0].role_id]
        self.motion_width = spec.shape[-1]
        self.is_drive = spec.interface == 'diff_drive'
        self.setpoint = torch.zeros((env.num_envs, self.group.count, self.motion_width), device=env.device)
        model = env.sim.mj_model
        reference = env.experiment.entity_models[self.group.model_id]
        self.body_ids = [model.body(f'{entity_name(cfg.group_id, i)}/{reference.root_body}').id
                         for i in range(self.group.count)]
        self.geom_ids = [torch.tensor((model.body_rootid[model.geom_bodyid] == body).nonzero()[0],
                                     device=env.device) for body in self.body_ids]
        self.support_geom_ids = [torch.tensor([model.geom(f'{entity_name(cfg.group_id, i)}/{name}').id
            for name in reference.support_geoms], dtype=torch.long, device=env.device) for i in range(self.group.count)]
        joints = [int(model.body_jntadr[i]) for i in self.body_ids]
        self.qpos_ids = torch.tensor([[int(model.jnt_qposadr[j])+k for k in range(7)] for j in joints], device=env.device)
        self.qvel_ids = torch.tensor([[int(model.jnt_dofadr[j])+k for k in range(6)] for j in joints], device=env.device)
        names = ['forward', 'turn'] if self.is_drive else ['body_thrust', 'x_moment', 'y_moment', 'z_moment']
        ids = [[model.actuator(f'{entity_name(cfg.group_id, i)}/{name}').id for name in names]
               for i in range(self.group.count)]
        self.actuator_ids = torch.tensor(ids, device=env.device)
        self.limits = torch.tensor(model.actuator_ctrlrange[ids], dtype=torch.float32, device=env.device)
        config = env.experiment.controllers[cfg.group_id]['config']
        if self.is_drive:
            self.wheel_dof_ids = torch.tensor([[int(model.jnt_dofadr[model.joint(f'{entity_name(cfg.group_id, i)}/{name}').id])
                for name in ('left', 'right')] for i in range(self.group.count)], device=env.device)
            self.controller = DiffDriveController(config)
        else:
            gears = model.actuator_gear[ids]
            self.gear = torch.tensor(gears[..., 2:].diagonal(axis1=1, axis2=2).copy(), dtype=torch.float32, device=env.device)
            self.controller = VelocityController(VelocityControllerConfig(**config),
                {'mass_kg': float(model.body_mass[self.body_ids[0]]), 'gravity_mps2': float(-model.opt.gravity[2])},
                (env.num_envs, self.group.count), env.device)
        self.saturation_count = torch.zeros_like(self.setpoint)

    @property
    def action_dim(self):
        return self._raw.shape[1]

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw.copy_(actions)
        offset = 0
        for entry in self.entries:
            binding = self._env.experiment.bindings[entry.role_id]
            width = entry.stop-entry.start
            self.setpoint[:, binding.role_slot_to_group_slot] = actions[:, offset:offset+width].reshape(self.num_envs, -1, self.motion_width)
            offset += width

    def state(self):
        return self._env.sim.data.qpos[:, self.qpos_ids], self._env.sim.data.qvel[:, self.qvel_ids]

    def apply_actions(self):
        if self.is_drive:
            control = self.controller.compute(self.setpoint, self._env.sim.data.qvel[:, self.wheel_dof_ids])
        else:
            pose, velocity = self.state()
            wrench = self.controller.compute(self.setpoint[..., :3], self.setpoint[..., 3],
                pose[..., 3:7], velocity[..., :3], velocity[..., 3:], self._env.physics_dt)
            control = wrench / self.gear
        if not torch.isfinite(control).all():
            raise RuntimeError('nonfinite controller output')
        self.saturation_count += (control < self.limits[..., 0]) | (control > self.limits[..., 1])
        self._env.sim.data.ctrl[:, self.actuator_ids] = control.clamp(self.limits[..., 0], self.limits[..., 1])

    def reset(self, env_ids):
        self._raw[env_ids] = 0
        self.setpoint[env_ids] = 0
        self.saturation_count[env_ids] = 0
        if not self.is_drive:
            pose, _ = self.state()
            self.controller.reset(env_ids, pose[env_ids, ..., 3:7])


@dataclass(kw_only=True)
class RadioActionTermCfg(ActionTermCfg):
    role_id: str

    def build(self, env):
        return RadioActionTerm(self, env)


class RadioActionTerm(ActionTerm):
    """已验证动作的唯一held请求；无线调度由Env驱动。"""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        spec = env.experiment.action_specs[cfg.role_id]
        self._initial_channel = spec.radio.channel_ids[0]
        self._raw = torch.zeros((env.num_envs, spec.shape[1]*2), device=env.device)
        self.request = RadioAction(self._raw.reshape(env.num_envs, -1, 2)[..., 0],
            torch.full(spec.shape[:2], self._initial_channel, dtype=torch.int32, device=env.device))

    @property
    def action_dim(self):
        return self._raw.shape[1]

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self.request.channel.copy_(actions.reshape(self.num_envs, -1, 2)[..., 1])

    def apply_actions(self):
        pass

    def reset(self, env_ids):
        self._raw[env_ids] = 0
        self.request.channel[env_ids] = self._initial_channel
