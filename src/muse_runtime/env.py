from dataclasses import dataclass, field
from functools import wraps
from types import SimpleNamespace
from typing import Callable

import torch
import warp as wp
import mujoco
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionManager
from mjlab.managers.command_manager import CommandManager
from mjlab.managers.event_manager import EventManager
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from muse_builtins.terms import registrations
from muse_sdk.runtime_types import Observations, PolicyInput, RoleMask, StepOutput, TaskUpdateResult, TransitionView
from .actions import MotionTermCfg, RadioActionTermCfg, encode_actions
from .managers import (CheckedObservationManager, CheckedObservationTermCfg, CheckedWorldRewardManager,
                       CostTermCfg, FixedTargetCfg, RoleCostManager, RoleRewardManager, WorldCostManager,
                       check_output)
from .scene import assemble_scene, entity_name
from .sensors import SensorSampler


@dataclass(frozen=True)
class RuntimeContext:
    device: str
    resource_paths: dict[str, str]
    term_registry: dict
    task_factory: Callable | None = None
    model_types: dict = field(default_factory=dict)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return type(value)(*(snapshot(item) for item in value))
    return value


def selection_key(selection):
    return '.'.join((*selection.owner_path, selection.term_kind, selection.selection_id))


def on_torch_stream(method):
    @wraps(method)
    def aligned(self, *args, **kwargs):
        with wp.ScopedStream(self._warp_stream):
            return method(self, *args, **kwargs)
    return aligned


class MuseEnv(ManagerBasedRlEnv):
    def __init__(self, experiment, context: RuntimeContext, rng):
        self.experiment, self.context, self.rng = experiment, context, rng
        self.term_registry = {f'builtin:{name}': reg for name, reg in registrations().items()}
        self.term_registry.update(context.term_registry)
        scene, sim = assemble_scene(experiment, context.resource_paths)
        cfg = ManagerBasedRlEnvCfg(scene=scene, sim=sim, decimation=experiment.scenario.scene.control_substeps,
                                  seed=None, events={}, scale_rewards_by_dt=False)
        wp.init()
        self._warp_stream = wp.get_stream(context.device)
        current = torch.cuda.current_stream(context.device)
        if self._warp_stream.cuda_stream != current.cuda_stream:
            self._warp_stream = wp.stream_from_torch(current)
        with wp.ScopedStream(self._warp_stream):
            super().__init__(cfg, context.device)
        self.policy_input = self.reset(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    def load_managers(self):
        self.event_manager = EventManager(self.cfg.events, self)
        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)
        self.command_manager = CommandManager({'target': FixedTargetCfg(resampling_time_range=(0., 0.))}, self)
        self.commands = self.command_manager.get_term('target')
        self.act_commands = SimpleNamespace(target_world=snapshot(self.commands.target_world))
        self._motion_groups = tuple(group.group_id for group in self.experiment.scenario.entity_groups
                                    if group.controller_id != 'none')
        action_terms = {group.group_id: MotionTermCfg(
            entity_name=entity_name(group.group_id, 0), group_id=group.group_id)
            for group in self.experiment.scenario.entity_groups if group.controller_id != 'none'}
        self._manager_slices = [entry for group in self._motion_groups for entry in self.experiment.action_layout
            if entry.field == 'motion' and self.experiment.bindings[entry.role_id].group_id == group]
        for entry in self.experiment.action_layout:
            if entry.field == 'radio':
                binding = self.experiment.bindings[entry.role_id]
                action_terms['radio:' + entry.role_id] = RadioActionTermCfg(role_id=entry.role_id,
                    entity_name=entity_name(binding.group_id, binding.role_slot_to_group_slot[0]))
                self._manager_slices.append(entry)
        self.action_manager = ActionManager(action_terms, self)
        self._radio_requests = {entry.role_id: self.action_manager.get_term('radio:' + entry.role_id).request
            for entry in self.experiment.action_layout if entry.field == 'radio'}
        self.role_positions_world = {}
        self.physics_outputs = SimpleNamespace(crashed={}, episode_decision=self.episode_length_buf,
            velocity_world={}, orientation_wxyz={}, angular_velocity_body={}, previous_action={},
            sim_time=torch.zeros(self.num_envs, dtype=torch.float64, device=self.device))
        self.role_masks = {role: RoleMask(*(torch.ones((self.num_envs, binding.role_slot_count),
            dtype=torch.bool, device=self.device) for _ in range(3))) for role, binding in self.experiment.bindings.items()}
        self.learning_done = {role: torch.zeros_like(mask.alive) for role, mask in self.role_masks.items()}
        self._contact_indices = torch.arange(self.sim.data.contact.worldid.shape[0], device=self.device)
        model = self.sim.mj_model
        horizontal = 1 - 2 * (model.geom_quat[:, 1]**2 + model.geom_quat[:, 2]**2) > 1 - 1e-6
        self._support_ground = torch.tensor(((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            & (model.geom_bodyid == 0) & horizontal).nonzero()[0], device=self.device)
        self._refresh_fields()
        self.sensor_sampler = SensorSampler(self.experiment, self.scene, self.device)
        self.sensor_outputs = self.sensor_sampler.outputs
        self.geometry_query = None
        self.wireless = None
        if self.experiment.network is not None:
            from .network.geometry import GeometryQuery
            from .network.runtime import WirelessRuntime
            if 'geometry' in self.experiment.activation:
                self.geometry_query = GeometryQuery(self.experiment.geometry, self.sim,
                    physics_dt=self.physics_dt, device=self.device)
                self._geometry_present = torch.ones((self.num_envs, len(self.experiment.geometry.layout.points)),
                    dtype=torch.bool, device=self.device)
                self._all_worlds = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.wireless = WirelessRuntime(self.experiment.network, self.experiment.model_bindings,
                self.context.model_types, batch_size=self.num_envs, device=self.device, rng=self.rng,
                geometry_layout=self.experiment.geometry.layout if self.experiment.geometry is not None else None,
                geometry_view=self.geometry_query.view if self.geometry_query is not None else None)
            self.network = self.wireless.network
            if 'sensing' in self.experiment.model_bindings:
                self.sensing_outputs = self.wireless.sensing_outputs
        self.task = self.context.task_factory(self.experiment.task_config, self) if self.context.task_factory else None
        if self.task is not None:
            self._check_task_fields()
        observations, rewards, world_rewards, costs, cost_roles, world_costs, roles, terminations = {}, {}, {}, {}, {}, {}, {}, {}
        self._observation_owners = {}
        self._termination_selections = []
        for selection in self.experiment.selections:
            if not selection.enabled:
                continue
            params = dict(selection.config)
            if selection.binding_role_id is not None:
                params['role_id'] = selection.binding_role_id
            func = self.term_registry[selection.term].implementation
            key = selection_key(selection)
            if selection.term_kind == 'observation':
                owner = tuple(selection.owner_path)
                group = '.'.join(owner)
                self._observation_owners[group] = owner
                observations.setdefault(group, ObservationGroupCfg(terms={}, concatenate_terms=False, nan_policy='error'))
                observations[group].terms[selection.selection_id] = CheckedObservationTermCfg(func=func, params=params,
                    output_shape=tuple(selection.output_spec.shape), output_dtype=getattr(torch, selection.output_spec.dtype))
            elif selection.term_kind == 'reward':
                cfg = RewardTermCfg(func=func, params=params, weight=selection.weight)
                if selection.binding_role_id is None:
                    world_rewards[key] = cfg
                else:
                    rewards[key] = cfg
                    roles[key] = selection.binding_role_id
            elif selection.term_kind == 'cost':
                cfg = CostTermCfg(func=func, params=params, cost_id=selection.cost_id)
                if selection.binding_role_id is None:
                    world_costs[key] = cfg
                else:
                    costs[key] = cfg
                    cost_roles[key] = selection.binding_role_id
            elif selection.term_kind == 'termination':
                terminations[key] = TerminationTermCfg(func=func, params=params, time_out=selection.time_out)
                self._termination_selections.append(selection)
        self.observation_manager = CheckedObservationManager(observations, self)
        self.reward_manager = RoleRewardManager(rewards, roles, self)
        self.world_reward_manager = CheckedWorldRewardManager(world_rewards, self, scale_by_dt=False) if world_rewards else None
        self.cost_manager = RoleCostManager(costs, cost_roles, self) if costs else None
        self.world_cost_manager = WorldCostManager(world_costs, self) if world_costs else None
        self.termination_manager = TerminationManager(terminations, self)

    def _check_task_fields(self):
        for name, spec in self.experiment.task_spec.published_fields.items():
            check_output(getattr(self.task, name), spec.shape, getattr(torch, spec.dtype), self.device, 'task.' + name)

    def _commit_task_update(self, result):
        # 用户计算输出的唯一提交边界，完整验证后再写入平台学习状态。
        spec = self.experiment.task_spec
        if not isinstance(result, TaskUpdateResult) or result.role_status:
            raise ValueError('Task must return TaskUpdateResult with empty role_status; physical fallback is not enabled')
        if set(result.learning_done) != set(spec.learning_done_roles):
            raise ValueError('Task learning_done roles differ from the declared layout')
        self._check_task_fields()
        for role, value in result.learning_done.items():
            check_output(value, self.learning_done[role].shape, torch.bool, self.device, 'task.learning_done.' + role)
        for role, value in result.learning_done.items():
            self.learning_done[role].copy_(value)

    def _refresh_fields(self):
        data = self.sim.data
        contacts = data.contact
        valid = self._contact_indices < data.nacon[0]
        valid &= contacts.dist <= 0
        for role, binding in self.experiment.bindings.items():
            term = self.action_manager.get_term(binding.group_id)
            pose, velocity = term.state()
            slots = binding.role_slot_to_group_slot
            self.role_positions_world[role] = pose[:, slots, :3]
            self.physics_outputs.velocity_world[role] = velocity[:, slots, :3]
            self.physics_outputs.orientation_wxyz[role] = pose[:, slots, 3:7]
            self.physics_outputs.angular_velocity_body[role] = velocity[:, slots, 3:]
            offset = 0
            for entry in term.entries:
                if entry.role_id == role:
                    self.physics_outputs.previous_action[role] = term.raw_action[:, offset:offset+entry.stop-entry.start].reshape(self.num_envs, -1, term.motion_width)
                    break
                offset += entry.stop-entry.start
            quat = pose[:, slots, 3:7]
            crashed = 1 - 2 * (quat[..., 1].square() + quat[..., 2].square()) <= 0
            for role_slot, slot in enumerate(slots):
                owned = torch.isin(contacts.geom, term.geom_ids[slot])
                support = torch.isin(contacts.geom, term.support_geom_ids[slot])
                ground = torch.isin(contacts.geom, self._support_ground)
                allowed = (support[:, 0] & ground[:, 1]) | (support[:, 1] & ground[:, 0])
                contact = valid & owned.any(-1) & ~allowed
                worlds = contacts.worldid[contact].long()
                crashed[worlds, role_slot] = True
            if role not in self.physics_outputs.crashed:
                self.physics_outputs.crashed[role] = crashed
            else:
                self.physics_outputs.crashed[role] |= crashed

    def _check_physics(self):
        data = self.sim.data
        if (data.nacon > data.naconmax).any() or (data.nefc > data.njmax).any() or data.overflow.any():
            raise RuntimeError('physics capacity overflow')
        if any(not torch.isfinite(getattr(data, name)).all() for name in ('qpos', 'qvel', 'qacc', 'ctrl')):
            raise RuntimeError('nonfinite physics')

    def _observe(self, env_ids=None):
        computed = self.observation_manager.compute(update_history=True, env_ids=env_ids)
        actors, critic = {}, {}
        for group, fields in computed.items():
            owner = self._observation_owners[group]
            (actors if owner[0] == 'roles' else critic)[owner[1]] = fields
        self.observations = Observations(actors, critic)
        return self.observations

    @torch.no_grad()
    @on_torch_stream
    def reset(self, world_mask):
        if world_mask.dtype != torch.bool or tuple(world_mask.shape) != (self.num_envs,):
            raise ValueError('reset requires bool world mask [B]')
        ids = world_mask.nonzero().flatten()
        if not len(ids):
            return self.policy_input
        self.sim.reset(ids)
        self.scene.reset(ids)
        self.sim.forward()
        self._check_physics()
        self.action_manager.reset(ids)
        self.episode_length_buf[ids] = 0
        self.physics_outputs.sim_time[ids] = 0
        self.sensor_sampler.reset(ids)
        for role, masks in self.role_masks.items():
            masks.alive[ids] = True
            masks.controllable[ids] = True
            self.learning_done[role][ids] = False
            self.physics_outputs.crashed[role][ids] = False
        self._refresh_fields()
        if self.wireless is not None:
            ticks = self.episode_length_buf * self.cfg.decimation
            if self.geometry_query is not None:
                self.geometry_query.refresh(world_mask, sample_tick=ticks, present=self._geometry_present)
            self.wireless.reset(world_mask, sample_tick=ticks)
        self.command_manager.reset(ids)
        if self.task is not None:
            self.task.reset(ids)
            self._check_task_fields()
        self.extras['log'] = {}
        for manager in (self.observation_manager, self.reward_manager, self.termination_manager,
                        self.cost_manager, self.world_cost_manager):
            if manager is None:
                continue
            self.extras['log'].update(manager.reset(ids))
        if self.world_reward_manager is not None:
            self.extras['log'].update(self.world_reward_manager.reset(ids))
        self.policy_input = PolicyInput(self._observe(ids), self.role_masks)
        return self.policy_input

    @torch.no_grad()
    @on_torch_stream
    def step(self, role_actions):
        self._check_physics()
        packed = encode_actions(self.experiment, role_actions, self.device)
        self.act_role_masks = snapshot(self.role_masks)
        for role, target in self.commands.target_world.items():
            self.act_commands.target_world[role].copy_(target)
        # 原生 manager 按 group 分派，ActionLayout 的 role 顺序不必按 group 连续。
        group_packed = torch.cat([packed[:, e.start:e.stop] for e in self._manager_slices], dim=-1)
        self.action_manager.process_action(group_packed)
        for crashed in self.physics_outputs.crashed.values():
            crashed.zero_()
        offsets = self.experiment.network.epoch_offsets if self.wireless is not None else ()
        intervals = dict(zip(offsets[:-1], offsets[1:]))
        if self.wireless is not None:
            self.wireless.begin_decision(self.episode_length_buf * self.cfg.decimation)
        for substep in range(self.cfg.decimation):
            if substep in intervals:
                ticks = self.episode_length_buf * self.cfg.decimation + substep
                if self.geometry_query is not None:
                    self.geometry_query.refresh(self._all_worlds, sample_tick=ticks, present=self._geometry_present)
                self.wireless.begin_epoch(ticks, interval_ticks=intervals[substep]-substep,
                    radio_requests=self._radio_requests)
            self._sim_step_counter += 1
            self.scene.write_data_to_sim()
            self.action_manager.apply_action()
            self.sim.step()
            self.scene.update(self.physics_dt)
            self._check_physics()
            self._refresh_fields()
            ticks = self.episode_length_buf * self.cfg.decimation + substep + 1
            due = self.sensor_sampler.due(ticks)
            sample_due = any(mask.any() for signals in due.values() for mask in signals.values())
            # RK4内的sensor缓存不代表积分末端；同tick共用一次forward。
            if sample_due or substep == self.cfg.decimation - 1 or (self.geometry_query is not None and substep + 1 in intervals):
                self.sim.forward()
                self._check_physics()
            if sample_due:
                self.sensor_sampler.capture(ticks, due)
            if self.wireless is not None and substep + 1 in offsets[1:]:
                self.wireless.end_epoch(ticks)
        self.episode_length_buf.add_(1)
        self.physics_outputs.sim_time.copy_(self.episode_length_buf.to(torch.float64)*self.step_dt)
        self.sensor_sampler.refresh_age(self.episode_length_buf * self.cfg.decimation)
        self.common_step_counter += 1
        self._refresh_fields()
        self.command_manager.compute(self.step_dt)
        diagnostics = {}
        if self.task is not None:
            result = self.task.update(self)
            self._commit_task_update(result)
            diagnostics['task'] = snapshot(result.diagnostics)
        observations = self._observe()
        rewards = self.reward_manager.compute()
        world_reward = self.world_reward_manager.compute(self.step_dt) if self.world_reward_manager is not None else None
        costs = {
            'roles': self.cost_manager.compute() if self.cost_manager is not None else {},
            'world': self.world_cost_manager.compute() if self.world_cost_manager is not None else {},
        }
        self.termination_manager.compute()
        terminated = self.termination_manager.terminated.clone()
        truncated = self.termination_manager.time_outs & ~terminated
        reset = terminated | truncated
        reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for selection in self._termination_selections:
            hit = self.termination_manager.get_term(selection_key(selection))
            eligible = truncated if selection.time_out else terminated
            identity = ('world', 'termination', selection.selection_id)
            number = next(k for k, v in self.experiment.termination_reasons.items() if tuple(v) == identity)
            reason[(reason == 0) & hit & eligible] = int(number)
        transition = TransitionView(self.act_role_masks, snapshot(observations), snapshot(rewards), snapshot(world_reward),
            terminated, truncated.clone(), reason, snapshot(self.learning_done), reset.clone(), snapshot(costs),
            snapshot(self.wireless.snapshot()) if self.wireless is not None else None)
        self.extras['log'] = {}
        self.policy_input = PolicyInput(observations, self.role_masks)
        if reset.any():
            diagnostics['termination'] = {'terminated': int(terminated.sum()), 'truncated': int(truncated.sum())}
            ticks = self.episode_length_buf[reset, None, None] * self.cfg.decimation
            diagnostics['control_saturation'] = {
                group: float((self.action_manager.get_term(group).saturation_count[reset] / ticks).mean())
                for group in self._motion_groups}
            self.reset(reset)
        diagnostics.update(self.extras['log'])
        return StepOutput(self.policy_input.observations, self.policy_input.role_masks, transition, diagnostics)

    def close(self):
        if self._offline_renderer is not None:
            self._offline_renderer.close()
