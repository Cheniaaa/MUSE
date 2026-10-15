"""逐 role 参数共享的本地 IPPO；模块导入/describe 保持 CPU 轻量。

actor 使用 present/alive/controllable，local critic 使用 present。
环境 reset 不恢复 learner RNG；学习 state 不包含物理、pending 或半段 rollout。
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from typing import Annotated

from pydantic import Field

from muse_sdk.config import DataModel
from muse_sdk.interfaces import Algorithm
from muse_sdk.runtime_types import ActResult, MotionAction


class IPPOConfig(DataModel):
    collection_steps: int = Field(default=24, gt=0, description='每次 driver update 前的成功 transition 数 H')
    hidden_dims: list[Annotated[int, Field(gt=0)]] = Field(default=[64, 64], min_length=1)
    gamma: float = Field(default=.99, ge=0, le=1)
    lam: float = Field(default=.95, ge=0, le=1)
    learning_rate: float = Field(default=.0003, gt=0)
    num_learning_epochs: int = Field(default=4, gt=0)
    num_mini_batches: int = Field(default=4, gt=0)
    clip_param: float = Field(default=.2, gt=0, lt=1)
    value_loss_coef: float = Field(default=1., gt=0)
    entropy_coef: float = Field(default=.01, ge=0)
    max_grad_norm: float = Field(default=1., gt=0)


def compute_returns(storage, gamma, lam):
    """唯一 GAE：terminal 不 bootstrap，truncated bootstrap 但不跨 episode 递推。"""
    import torch
    with torch.no_grad():
        advantage = torch.zeros_like(storage.values[0])
        for step in reversed(range(storage.rewards.shape[0])):
            delta = (storage.rewards[step] + gamma * storage.final_values[step]
                     * ~storage.terminated[step] - storage.values[step])
            advantage = delta + gamma * lam * ~(storage.terminated[step] | storage.truncated[step]) * advantage
            storage.returns[step].copy_(advantage + storage.values[step])
        raw = storage.returns - storage.values
        valid = raw[storage.actor_valid]
        storage.advantages.zero_()
        if valid.numel():
                storage.advantages[storage.actor_valid] = (valid - valid.mean()) / (valid.std(unbiased=False) + 1e-8)


def compute_cost_returns(costs, values, final_values, terminated, truncated, gamma, lam, valid=None):
    """独立named cost的GAE；terminal抑制bootstrap，timeout只截断递推。"""
    import torch
    with torch.no_grad():
        valid = torch.ones_like(terminated) if valid is None else valid
        advantage = torch.zeros_like(values[0])
        advantages = torch.empty_like(values)
        for step in reversed(range(costs.shape[0])):
            delta = costs[step] + gamma * torch.where(terminated[step], 0., final_values[step]) - values[step]
            advantage = torch.where(valid[step], delta + gamma * lam * torch.where(
                terminated[step] | truncated[step], 0., advantage), 0.)
            advantages[step] = advantage
        return torch.where(valid, advantages + values, 0.), advantages


def update_lambda(value, estimate, budget, learning_rate):
    return (value + learning_rate * (estimate - budget)).clamp_min_(0.)


class IPPO(Algorithm):
    def __init__(self, config):
        self.config = config
        self.collection_steps = config['collection_steps']
        self.children = {}
        self.cost_state = {}
        self.fields = {}
        self.bounds = {}
        self.shapes = {}
        self.update_count = 0
        self.mode = 'TRAIN'
        self._sequence = 0

    @classmethod
    def describe(cls):
        return dict(algorithm_id='ippo', config_schema=IPPOConfig,
                    observation_requirements=[], action_requirements=['body_velocity', 'diff_drive'],
                     actor_observation='all_local', critic_inputs={}, reward_inputs=['role'], cost_inputs=['role'],
                     act_aux={}, diagnostics={}, training_schedule={'collection_steps': 'collection_steps'})

    @contextmanager
    def _learner_rng(self):
        import torch
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(self._cpu_rng)
            if devices:
                torch.cuda.set_rng_state(self._cuda_rng, self.device)
            try:
                yield
            finally:
                self._cpu_rng = torch.random.get_rng_state()
                if devices:
                    self._cuda_rng = torch.cuda.get_rng_state(self.device)

    def initialize(self, experiment, context, rng):
        import torch
        from tensordict import TensorDict
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from ._storage import MaskedStorage

        if self.children:
            raise RuntimeError('IPPO is already initialized')
        if experiment.training_schedule.collection_steps != self.collection_steps:
            raise ValueError('TrainingSchedule collection_steps must match IPPO storage H')
        self.device = torch.device(context.device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        seed = int.from_bytes(hashlib.sha256(f'{experiment.root_seed}:ippo:learner'.encode()).digest()[:8], 'little')
        self._cpu_rng = torch.Generator(device='cpu').manual_seed(seed).get_state()
        self._cuda_rng = (torch.Generator(device=self.device).manual_seed(seed).get_state()
                          if self.device.type == 'cuda' else None)
        with self._learner_rng():
            for role, spec in experiment.action_specs.items():
                b, n, width = spec.shape
                self.shapes[role] = (b, n)
                self.fields[role] = {item.path[3]: tuple(item.tensor.shape[2:])
                                    for item in experiment.algorithm_inputs.actors[role]}
                if not self.fields[role]:
                    raise ValueError(f'IPPO requires actor observations for {role}')
                self.bounds[role] = (torch.tensor(spec.minimum, device=self.device),
                                     torch.tensor(spec.maximum, device=self.device))
                obs = TensorDict({key: torch.zeros(b*n, math.prod(shape), device=self.device)
                                  for key, shape in self.fields[role].items()}, [b*n], device=self.device)
                groups = {'actor': list(obs.keys()), 'critic': list(obs.keys())}
                actor = MLPModel(obs, groups, 'actor', width, hidden_dims=self.config['hidden_dims'],
                                 distribution_cfg={'class_name': 'rsl_rl.modules.distribution:BetaDistribution',
                                                   'action_range': (0., 1.)})
                critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=self.config['hidden_dims'])
                storage = MaskedStorage('rl', b*n, self.collection_steps, obs, [width], str(self.device))
                ppo = {key: value for key, value in self.config.items() if key not in ('collection_steps', 'hidden_dims')}
                self.children[role] = PPO(actor, critic, storage, **ppo, schedule='fixed', device=str(self.device))
            self._initialize_costs(experiment.costs.model_dump())
        self.set_mode(self.mode)

    def _initialize_costs(self, layout, centralized=False):
        from ._costs import CostLearner
        if layout['world'] and not centralized:
            raise ValueError('IPPO does not support world cost scope')
        for role, costs in layout['roles'].items():
            for cost_id, metadata in costs.items():
                self.cost_state[(role, cost_id)] = CostLearner(self, role, metadata, centralized)
        for cost_id, metadata in layout['world'].items():
            self.cost_state[(None, cost_id)] = CostLearner(self, None, metadata, centralized)
        self._cost_layout = layout

    def _observations(self, observations, role):
        from tensordict import TensorDict
        b, n = self.shapes[role]
        return TensorDict({key: observations.actors[role][key].detach().reshape(b*n, -1).clone()
                           for key in self.fields[role]}, [b*n], device=self.device)

    def to_motion(self, role, raw):
        lo, hi = self.bounds[role]
        return lo + (hi - lo) * raw

    def set_mode(self, mode):
        if mode not in ('TRAIN', 'EVALUATE'):
            raise ValueError('Unsupported IPPO mode')
        if any(c.storage.step or c.storage.pending is not None for c in self.children.values()):
            raise RuntimeError('Cannot change mode during a rollout')
        self.mode = mode
        for child in self.children.values():
            child.train_mode() if mode == 'TRAIN' else child.eval_mode()
        for cost in self.cost_state.values():
            cost.critic.train(mode == 'TRAIN')

    def act(self, observations, role_masks, rng, deterministic):
        import torch
        if not self.children:
            raise RuntimeError('Initialize IPPO first')
        if any(c.storage.pending is not None or c.storage.step == self.collection_steps for c in self.children.values()):
            raise RuntimeError('Consume pending decision or update full rollout before act')
        self._sequence += 1
        token = self._sequence
        actions = {}
        with self._learner_rng(), torch.no_grad():
            for role, child in self.children.items():
                obs = self._observations(observations, role)
                if self.mode == 'TRAIN':
                    raw = child.act(obs)
                    if deterministic:
                        raw = child.actor(obs)
                        child.transition.actions = raw.detach()
                        child.transition.actions_log_prob = child.actor.get_output_log_prob(raw).detach()
                    child.storage.pending = child.transition
                    child.storage.pending_aux = token
                    child.transition = type(child.transition)()
                else:
                    raw = child.actor(obs, stochastic_output=False)
                actions[role] = MotionAction(self.to_motion(role, raw).reshape(*self.shapes[role], -1))
            if self.mode == 'TRAIN':
                for cost in self.cost_state.values():
                    cost.act(observations, role_masks)
        return ActResult(actions, token)

    def discard_pending(self):
        for cost in self.cost_state.values():
            cost.discard_pending()
        for child in self.children.values():
            child.storage.pending = None
            child.storage.pending_aux = None
            child.transition.clear()

    def process_env_step(self, transition, act_aux):
        import torch
        if self.mode != 'TRAIN':
            return
        if any(c.storage.pending is None or c.storage.pending_aux != act_aux for c in self.children.values()):
            raise RuntimeError('Missing or consumed pending act_aux')
        with torch.no_grad():
            for cost in self.cost_state.values():
                cost.process_env_step(transition)
            for role, child in self.children.items():
                st = child.storage
                obs = self._observations(transition.pre_reset_next_observations, role)
                terminal = transition.world_terminated[:, None] | transition.learning_done[role]
                truncated = transition.world_truncated[:, None].expand(self.shapes[role])
                masks = transition.act_masks[role]
                st.final_values[st.step].copy_(child.critic(obs))
                st.terminated[st.step].copy_(terminal.reshape(-1, 1))
                st.truncated[st.step].copy_(truncated.reshape(-1, 1))
                st.actor_valid[st.step].copy_((masks.present & masks.alive & masks.controllable).reshape(-1, 1))
                st.critic_valid[st.step].copy_(masks.present.reshape(-1, 1))
                child.transition = st.pending
                child.process_env_step(obs, transition.rewards[role].reshape(-1),
                                       (terminal | truncated).reshape(-1), extras={})
                st.pending = None
                st.pending_aux = None

    def update(self):
        import torch
        from ._storage import masked_update
        if self.mode != 'TRAIN' or any(c.storage.step != self.collection_steps or c.storage.pending is not None
                                       for c in self.children.values()):
            raise RuntimeError('Update requires a full rollout without pending decision')
        metrics = {}
        with self._learner_rng():
            for cost in self.cost_state.values():
                cost.compute_returns()
            for role, child in self.children.items():
                compute_returns(child.storage, child.gamma, child.lam)
                for cost in self.cost_state.values():
                    cost.constrain_advantages(role, child.storage)
                metrics[role] = masked_update(child)
                if any(not torch.isfinite(p).all() for model in (child.actor, child.critic) for p in model.parameters()):
                    raise RuntimeError('Nonfinite IPPO parameters')
            for cost in self.cost_state.values():
                cost.update()
        self.update_count += 1
        return metrics

    def _parameter_names(self, child):
        return [f'{kind}.{name}' for kind in ('actor', 'critic')
                for name, _ in getattr(child, kind).named_parameters()]

    def learning_state_dict(self):
        """返回普通state_dict，可借用live tensor；稳定CPU快照归Runtime。"""
        state = {'revision': 1, 'updates': self.update_count, 'cpu_rng': self._cpu_rng,
                          'cuda_rng': self._cuda_rng, 'roles': {
                              role: {'actor': child.actor.state_dict(), 'critic': child.critic.state_dict(),
                                     'optimizer': child.optimizer.state_dict(),
                                     'parameter_names': self._parameter_names(child)}
                                for role, child in self.children.items()}}
        if self.cost_state:
            # JSON数组键保留role/world与任意名称边界，且满足Runtime的primitive key契约。
            state['costs'] = {json.dumps(key): cost.state_dict() for key, cost in self.cost_state.items()}
        return state

    def load_learning_state_dict(self, state):
        import torch
        costs = {json.dumps(key): cost for key, cost in self.cost_state.items()}
        if ('costs' in state) != bool(costs) or set(state.get('costs', {})) != set(costs):
            raise ValueError('Cost checkpoint layout mismatch')
        for key, cost in costs.items():
            cost.validate_state_dict(state['costs'][key])
        if state['revision'] != 1 or set(state['roles']) != set(self.children):
            raise ValueError('Incompatible learning state revision/role names')
        for key, current in [('cpu_rng', self._cpu_rng), ('cuda_rng', self._cuda_rng)]:
            saved_rng = state[key]
            if current is None:
                if saved_rng is not None:
                    raise ValueError('RNG device mismatch')
            elif (not isinstance(saved_rng, torch.Tensor) or saved_rng.shape != current.shape
                  or saved_rng.dtype != current.dtype):
                raise ValueError('RNG state shape/dtype mismatch')
        # 所有 role 先检查再写入，不以新 optimizer 空 lazy state 拒绝合法历史状态。
        for role, child in self.children.items():
            saved = state['roles'][role]
            if saved['parameter_names'] != self._parameter_names(child):
                raise ValueError('Optimizer parameter names do not match')
            for kind in ('actor', 'critic'):
                current = getattr(child, kind).state_dict()
                if set(saved[kind]) != set(current):
                    raise ValueError(f'{role}.{kind} state names do not match')
                for name, value in current.items():
                    other = saved[kind][name]
                    if other.shape != value.shape or other.dtype != value.dtype:
                        raise ValueError(f'{role}.{kind}.{name} shape/dtype mismatch')
            optimizer = saved['optimizer']
            groups = child.optimizer.state_dict()['param_groups']
            if [g['params'] for g in optimizer['param_groups']] != [g['params'] for g in groups]:
                raise ValueError('Optimizer parameter groups do not match')
            # 与cost一致，固定Adam配置在任何live学习状态写入前核对。
            for group, current_group in zip(optimizer['param_groups'], groups):
                if set(group) != set(current_group) or any(
                        type(group[key]) is not type(value) or group[key] != value
                        for key, value in current_group.items() if key != 'params'):
                    raise ValueError('Optimizer configuration mismatch')
            parameters = list(child.actor.parameters()) + list(child.critic.parameters())
            for index, values in optimizer['state'].items():
                if index not in range(len(parameters)):
                    raise ValueError('Unknown optimizer parameter')
                if set(values) != {'step', 'exp_avg', 'exp_avg_sq'}:
                    raise ValueError('Adam state names do not match')
                if not isinstance(values['step'], torch.Tensor) or values['step'].shape != torch.Size([]):
                    raise ValueError('Adam step shape mismatch')
                for name in ('exp_avg', 'exp_avg_sq'):
                    if values[name].shape != parameters[index].shape or values[name].dtype != parameters[index].dtype:
                        raise ValueError('Adam parameter shape/dtype mismatch')
        for role, child in self.children.items():
            saved = state['roles'][role]
            child.actor.load_state_dict(saved['actor'], strict=True)
            child.critic.load_state_dict(saved['critic'], strict=True)
            child.optimizer.load_state_dict(deepcopy(saved['optimizer']))
            child.learning_rate = child.optimizer.param_groups[0]['lr']
            child.storage.clear()
        self.discard_pending()
        for key, cost in costs.items():
            cost.load_state_dict(state['costs'][key])
        self.update_count = state['updates']
        self._cpu_rng = state['cpu_rng'].cpu().clone()
        self._cuda_rng = state['cuda_rng'].cpu().clone() if state['cuda_rng'] is not None else None
