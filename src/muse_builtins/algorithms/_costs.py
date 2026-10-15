"""每个命名cost独占critic/Adam/lambda；actor优化仍由原PPO负责。"""
import math
from copy import deepcopy

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from ._storage import MaskedStorage
from .ippo import compute_cost_returns, update_lambda


class CostLearner:
    def __init__(self, algorithm, role, metadata, centralized):
        self.algorithm, self.role = algorithm, role
        self.metadata = metadata
        self.centralized = centralized
        self.shape = algorithm.shapes[role] if role is not None else (next(iter(algorithm.shapes.values()))[0],)
        count = math.prod(self.shape)
        width = (sum(map(math.prod, algorithm.critic_fields.values())) + (self.shape[1] if role else 0)
                 if centralized else sum(map(math.prod, algorithm.fields[role].values())))
        obs = TensorDict({'cost': torch.zeros(count, width, device=algorithm.device)}, [count])
        options = {'hidden_dims': algorithm.config['hidden_dims']}
        model = MLPModel
        recurrent = algorithm.config.get('recurrent')
        if recurrent:
            from ._recurrent import MaskedRNNModel
            model = MaskedRNNModel
            options.update(rnn_type=recurrent['type'], rnn_hidden_dim=recurrent['hidden_size'], rnn_num_layers=1)
        self.critic = model(obs, {'critic': ['cost']}, 'critic', 1, **options).to(algorithm.device)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=algorithm.config['learning_rate'])
        self.multiplier = torch.tensor(metadata['lambda_initial'], device=algorithm.device)
        self.storage = MaskedStorage('rl', count, algorithm.collection_steps, obs, [0], str(algorithm.device))
        # cost只消费critic序列；空动作分布无需分配actor状态。
        self.storage.distribution_params = ()
        self.storage.sample_kind = 'critic'
        self.clear_episode()

    def clear_episode(self):
        self.storage.clear()
        self.storage.pending = None
        self.episode_sum = torch.zeros(self.shape, device=self.algorithm.device)
        self.episode_discount = torch.ones_like(self.episode_sum)
        self.episode_valid = torch.zeros(self.shape, dtype=torch.bool, device=self.algorithm.device)
        self.completed_sum = torch.zeros((), device=self.algorithm.device)
        self.completed_count = 0
        if self.critic.is_recurrent:
            rnn = self.critic.rnn.rnn
            zero = next(self.critic.parameters()).new_zeros(rnn.num_layers, math.prod(self.shape), rnn.hidden_size)
            hidden = (zero, torch.zeros_like(zero)) if isinstance(rnn, torch.nn.LSTM) else zero
            self.critic.reset(hidden_state=hidden)

    def observations(self, observations, valid):
        if self.centralized:
            b = self.shape[0]
            central = torch.cat([observations.critic_inputs[group][key].reshape(b, -1)
                                 for group, key in self.algorithm.critic_fields], dim=-1)
            if self.role is None:
                data = central
            else:
                n = self.shape[1]
                slots = torch.eye(n, device=central.device).expand(b, n, n)
                data = torch.cat((central[:, None].expand(b, n, -1), slots), dim=-1).flatten(0, 1)
        else:
            data = torch.cat([observations.actors[self.role][key].reshape(math.prod(self.shape), -1)
                              for key in self.algorithm.fields[self.role]], dim=-1)
        data = torch.where(valid.reshape(-1, 1), data, 0.).detach().clone()
        return TensorDict({'cost': data}, [math.prod(self.shape)])

    def valid_mask(self, masks):
        if self.role is None:
            return torch.ones(self.shape, dtype=torch.bool, device=self.algorithm.device)
        return masks[self.role].present & masks[self.role].alive

    def act(self, observations, masks):
        valid = self.valid_mask(masks)
        self.critic.reset((~valid).flatten())
        pending = self.storage.Transition()
        pending.observations = self.observations(observations, valid)
        pending.hidden_states = (None, self.critic.get_hidden_state())
        pending.values = self.critic(pending.observations).detach()
        self.storage.pending = pending

    def discard_pending(self):
        if self.storage.pending is not None:
            self.critic.reset(hidden_state=self.storage.pending.hidden_states[1])
        self.storage.pending = None

    def transition_cost(self, costs):
        return (costs['world'] if self.role is None else costs['roles'][self.role])[self.metadata['cost_id']]

    def process_env_step(self, transition):
        st = self.storage
        valid = self.valid_mask(transition.act_masks)
        cost = torch.where(valid, self.transition_cost(transition.costs), 0.)
        terminal = transition.world_terminated
        truncated = transition.world_truncated
        if self.role is not None:
            terminal = terminal[:, None] | transition.learning_done[self.role]
            truncated = truncated[:, None].expand(self.shape)
        hidden = self.critic.get_hidden_state()
        try:
            final = self.critic(self.observations(transition.pre_reset_next_observations, valid)).reshape(self.shape)
        finally:
            self.critic.reset(hidden_state=hidden)
        boundary = terminal | truncated
        i = st.step
        st.observations[i].copy_(st.pending.observations)
        st.values[i].copy_(st.pending.values)
        st.rewards[i].copy_(cost.reshape(-1, 1))
        st.final_values[i].copy_(final.reshape(-1, 1))
        st.terminated[i].copy_(terminal.reshape(-1, 1))
        st.truncated[i].copy_(truncated.reshape(-1, 1))
        st.critic_valid[i].copy_(valid.reshape(-1, 1))
        st.dones[i].copy_((boundary | ~valid).reshape(-1, 1))
        st._save_hidden_states(st.pending.hidden_states)
        st.step += 1
        st.pending = None
        self.critic.reset((boundary | ~valid).flatten())
        self.episode_sum.add_(self.episode_discount * cost)
        self.episode_discount.mul_(self.metadata['discount'])
        self.episode_valid |= valid
        completed = boundary & self.episode_valid
        estimates = self.episode_sum + self.episode_discount * torch.where(truncated & ~terminal & valid, final, 0.)
        self.completed_sum.add_(estimates[completed].sum())
        self.completed_count += int(completed.sum().item())
        self.episode_sum[boundary] = 0.
        self.episode_discount[boundary] = 1.
        self.episode_valid[boundary] = False

    def compute_returns(self):
        st = self.storage
        returns, advantages = compute_cost_returns(st.rewards, st.values, st.final_values, st.terminated,
            st.truncated, self.metadata['discount'], self.algorithm.config['lam'], valid=st.critic_valid)
        st.returns.copy_(returns)
        st.advantages.copy_(advantages)

    def constrain_advantages(self, owner, storage):
        if self.centralized:
            role, slot = self.algorithm.owners[owner]
        else:
            role, slot = owner, None
        if self.role is not None and self.role != role:
            return
        h = self.algorithm.collection_steps
        b, n = self.algorithm.shapes[role]
        if self.role is None:
            advantage = self.storage.advantages.reshape(h, b, 1).expand(h, b, n)
        else:
            advantage = self.storage.advantages.reshape(h, b, n)
        if slot is not None:
            advantage = advantage[:, :, slot:slot + 1]
        storage.advantages.sub_(self.multiplier * advantage.reshape_as(storage.advantages))

    def update(self):
        st, cfg = self.storage, self.algorithm.config
        count = int(st.critic_valid.sum().item())
        if count:
            generator = st.recurrent_mini_batch_generator if self.critic.is_recurrent else st.mini_batch_generator
            for batch in generator(min(cfg['num_mini_batches'], count), cfg['num_learning_epochs']):
                predicted = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                loss = (predicted - batch.returns).square().mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg['max_grad_norm'])
                self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.critic.parameters()):
                raise RuntimeError('Nonfinite cost critic parameters')
        if self.completed_count:
            self.multiplier.copy_(update_lambda(self.multiplier, self.completed_sum / self.completed_count,
                                               self.metadata['budget'], self.metadata['lambda_learning_rate']))
        self.completed_sum.zero_()
        self.completed_count = 0
        st.clear()

    def signature(self):
        algorithm = self.algorithm
        return {'role': self.role, 'slots': self.shape[1:], 'centralized': self.centralized,
                'inputs': list(algorithm.critic_fields.items()) if self.centralized
                else list(algorithm.fields[self.role].items())}

    def state_dict(self):
        return {'revision': 1, 'metadata': deepcopy(self.metadata), 'signature': self.signature(),
                'critic': self.critic.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'parameter_names': [name for name, _ in self.critic.named_parameters()],
                'lambda': self.multiplier}

    def validate_state_dict(self, saved):
        if (saved.get('revision') != 1 or saved.get('metadata') != self.metadata
                or saved.get('signature') != self.signature()):
            raise ValueError('Cost checkpoint metadata/layout mismatch')
        current = self.critic.state_dict()
        if set(saved['critic']) != set(current):
            raise ValueError('Cost critic state names mismatch')
        for name, value in current.items():
            other = saved['critic'][name]
            if not isinstance(other, torch.Tensor) or other.shape != value.shape or other.dtype != value.dtype:
                raise ValueError('Cost critic shape/dtype mismatch')
        multiplier = saved['lambda']
        if (not isinstance(multiplier, torch.Tensor) or multiplier.shape != self.multiplier.shape
                or multiplier.dtype != self.multiplier.dtype or not torch.isfinite(multiplier) or multiplier < 0):
            raise ValueError('Cost lambda shape/dtype/value mismatch')
        parameters = list(self.critic.parameters())
        if saved['parameter_names'] != [name for name, _ in self.critic.named_parameters()]:
            raise ValueError('Cost optimizer parameter names mismatch')
        optimizer = saved['optimizer']
        current_groups = self.optimizer.state_dict()['param_groups']
        if ([g['params'] for g in optimizer['param_groups']] !=
                [g['params'] for g in current_groups]):
            raise ValueError('Cost optimizer parameter groups mismatch')
        # 固定Adam配置是恢复契约的一部分；库load不会验证缺少lr等情况。
        for group, current_group in zip(optimizer['param_groups'], current_groups):
            if set(group) != set(current_group) or any(
                    type(group[key]) is not type(value) or group[key] != value
                    for key, value in current_group.items() if key != 'params'):
                raise ValueError('Cost optimizer configuration mismatch')
        for index, values in optimizer['state'].items():
            if index not in range(len(parameters)) or set(values) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Cost Adam parameter/state names mismatch')
            if not isinstance(values['step'], torch.Tensor) or values['step'].shape != torch.Size([]):
                raise ValueError('Cost Adam step shape mismatch')
            for name in ('exp_avg', 'exp_avg_sq'):
                value = values[name]
                if value.shape != parameters[index].shape or value.dtype != parameters[index].dtype:
                    raise ValueError('Cost Adam parameter shape/dtype mismatch')

    def load_state_dict(self, saved):
        # 全部owner由入口先校验；这里只恢复学习状态，新环境清除采样状态。
        self.critic.load_state_dict(saved['critic'], strict=True)
        self.optimizer.load_state_dict(deepcopy(saved['optimizer']))
        self.multiplier.copy_(saved['lambda'])
        self.clear_episode()
