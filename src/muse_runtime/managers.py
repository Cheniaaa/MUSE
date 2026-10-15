from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.observation_manager import ObservationManager, ObservationTermCfg


def check_output(value, shape, dtype, device, name):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError(f'{name}: output shape/dtype mismatch')
    if value.device != torch.device(device):
        raise ValueError(f'{name}: output device mismatch')
    if not torch.isfinite(value).all():
        raise RuntimeError(f'{name}: nonfinite output')
    return value


@dataclass(kw_only=True)
class CheckedObservationTermCfg(ObservationTermCfg):
    output_shape: tuple[int, ...]
    output_dtype: torch.dtype


class CheckedObservationManager(ObservationManager):
    def _resolve_common_term_cfg(self, term_name, term_cfg):
        super()._resolve_common_term_cfg(term_name, term_cfg)
        implementation = term_cfg.func
        def checked(env, **params):
            value = implementation(env, **params)
            return check_output(value, term_cfg.output_shape, term_cfg.output_dtype, env.device, term_name)
        # 原生发现并构造实例后，仅在返回边界加检查；reset仍调用同一实例。
        if callable(getattr(implementation, 'reset', None)):
            checked.reset = implementation.reset
        term_cfg.func = checked


class CheckedWorldRewardManager(RewardManager):
    def reset(self, env_ids):
        # 与 role 指标一致，记录累计值；不按某个 timeout 名称猜测归一化时长。
        metrics = {f'Episode_Reward/{name}': values[env_ids].mean().item()
                   for name, values in self._episode_sums.items()}
        for values in self._episode_sums.values():
            values[env_ids] = 0
        for cfg in self._class_term_cfgs:
            cfg.func.reset(env_ids=env_ids)
        return metrics

    def _resolve_common_term_cfg(self, term_name, term_cfg):
        super()._resolve_common_term_cfg(term_name, term_cfg)
        implementation = term_cfg.func
        def checked(env, **params):
            value = check_output(implementation(env, **params), (env.num_envs,), torch.float32, env.device, term_name)
            if not torch.isfinite(value * term_cfg.weight).all():
                raise RuntimeError(f'{term_name}: nonfinite weighted reward')
            return value
        if callable(getattr(implementation, 'reset', None)):
            checked.reset = implementation.reset
        term_cfg.func = checked

    def compute(self, dt):
        for cfg in self._term_cfgs:
            if cfg.weight == 0:
                cfg.func(self._env, **cfg.params)
        result = super().compute(dt)
        if any(not torch.isfinite(value).all() for value in self._episode_sums.values()):
            raise RuntimeError('nonfinite world episode reward')
        return check_output(result, (self.num_envs,), torch.float32, self.device, 'world reward aggregate')


@dataclass(kw_only=True)
class FixedTargetCfg(CommandTermCfg):
    def build(self, env):
        return FixedTarget(self, env)


class FixedTarget(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.target_world = {}
        for role, binding in env.experiment.bindings.items():
            targets = torch.tensor(env.experiment.scenario.command_targets_world[role], device=env.device)
            self.target_world[role] = targets.unsqueeze(0).repeat(env.num_envs, 1, 1)

    @property
    def command(self):
        return self.target_world

    def _resample(self, env_ids):
        self.time_left[env_ids] = float('inf')
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids):
        for role, binding in self._env.experiment.bindings.items():
            targets = torch.tensor(self._env.experiment.scenario.command_targets_world[role], device=self.device)
            self.target_world[role][env_ids] = targets

    def _update_metrics(self):
        pass

    def _update_command(self, env_ids):
        pass


class RoleRewardManager(ManagerBase):
    def __init__(self, cfg: dict[str, RewardTermCfg], roles, env):
        self.cfg = cfg
        self.roles = roles
        super().__init__(env)
        self.episode_sums = {name: torch.zeros_like(env.role_positions_world[roles[name]][..., 0]) for name in cfg}

    @property
    def active_terms(self):
        return list(self.cfg)

    def _prepare_terms(self):
        self._class_term_cfgs = []
        for name, cfg in self.cfg.items():
            self._resolve_common_term_cfg(name, cfg)
            if callable(getattr(cfg.func, 'reset', None)):
                self._class_term_cfgs.append(cfg)

    def compute(self):
        rewards = {role: torch.zeros_like(pos[..., 0]) for role, pos in self._env.role_positions_world.items()}
        for name, cfg in self.cfg.items():
            value = cfg.func(self._env, **cfg.params)
            role = self.roles[name]
            check_output(value, rewards[role].shape, torch.float32, self.device, name)
            contribution = value * cfg.weight
            rewards[role].add_(contribution)
            self.episode_sums[name].add_(contribution)
        if any(not torch.isfinite(value).all() for value in rewards.values()):
            raise RuntimeError('nonfinite reward aggregate')
        if any(not torch.isfinite(value).all() for value in self.episode_sums.values()):
            raise RuntimeError('nonfinite episode reward')
        return rewards

    def reset(self, env_ids):
        metrics = {f'Episode_Reward/{name}': value[env_ids].mean().item() for name, value in self.episode_sums.items()}
        for value in self.episode_sums.values():
            value[env_ids] = 0
        for cfg in self._class_term_cfgs:
            cfg.func.reset(env_ids=env_ids)
        return metrics


@dataclass(kw_only=True)
class CostTermCfg(ManagerTermBaseCfg):
    cost_id: str


class _CostManager(ManagerBase):
    @property
    def active_terms(self):
        return list(self.cfg)

    def _prepare_terms(self):
        self._class_term_cfgs = []
        for name, cfg in self.cfg.items():
            self._resolve_common_term_cfg(name, cfg)
            if callable(getattr(cfg.func, 'reset', None)):
                self._class_term_cfgs.append(cfg)

    def reset(self, env_ids):
        for cfg in self._class_term_cfgs:
            cfg.func.reset(env_ids=env_ids)
        return {}


class RoleCostManager(_CostManager):
    def __init__(self, cfg: dict[str, CostTermCfg], roles, env):
        self.cfg = cfg
        self.roles = roles
        super().__init__(env)

    def compute(self):
        result = {}
        for name, cfg in self.cfg.items():
            value = cfg.func(self._env, **cfg.params)
            role = self.roles[name]
            expected = self._env.role_positions_world[role][..., 0].shape
            check_output(value, expected, torch.float32, self.device, name)
            result.setdefault(role, {})[cfg.cost_id] = value
        return result


class WorldCostManager(_CostManager):
    def __init__(self, cfg: dict[str, CostTermCfg], env):
        self.cfg = cfg
        super().__init__(env)

    def compute(self):
        result = {}
        for name, cfg in self.cfg.items():
            value = cfg.func(self._env, **cfg.params)
            check_output(value, (self.num_envs,), torch.float32, self.device, name)
            result[cfg.cost_id] = value
        return result
