"""RSL 单一 storage 的有效样本投影；PPO loss 和 optimizer 仍归 RSL。"""
import torch
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class MaskedStorage(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_values = torch.zeros_like(self.values)
        self.terminated = torch.zeros_like(self.dones, dtype=torch.bool)
        self.truncated = torch.zeros_like(self.terminated)
        self.actor_valid = torch.zeros_like(self.terminated)
        self.critic_valid = torch.zeros_like(self.terminated)
        self.pending = None
        self.pending_aux = None
        self.sample_kind = 'actor'

    def _save_hidden_states(self, hidden_states):
        # None 表示零初态；已分配 buffer 时也必须覆盖上一 rollout 的值。
        for state, saved in zip(hidden_states, (self.saved_hidden_state_a, self.saved_hidden_state_c)):
            if state is None and saved is not None:
                for tensor in saved:
                    tensor[self.step].zero_()
        super()._save_hidden_states(hidden_states)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        valid = getattr(self, self.sample_kind + '_valid').flatten().nonzero().flatten()
        for _ in range(num_epochs):
            indices = valid[torch.randperm(valid.numel(), device=self.device)]
            for index in indices.tensor_split(num_mini_batches):
                yield self.Batch(
                    observations=self.observations.flatten(0, 1)[index],
                    actions=self.actions.flatten(0, 1)[index],
                    values=self.values.flatten(0, 1)[index],
                    advantages=self.advantages.flatten(0, 1)[index],
                    returns=self.returns.flatten(0, 1)[index],
                    old_actions_log_prob=self.actions_log_prob.flatten(0, 1)[index],
                    old_distribution_params=tuple(p.flatten(0, 1)[index] for p in self.distribution_params))

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        valid = getattr(self, self.sample_kind + '_valid').squeeze(-1)
        indices = valid.flatten().nonzero().flatten()
        batches = min(num_mini_batches, indices.numel())
        if not batches:
            return
        for _ in range(num_epochs):
            shuffled = indices[torch.randperm(indices.numel(), device=self.device)]
            for index in shuffled.tensor_split(batches):
                # 只筛选 loss，所涉及 env 的真实序列和 episode 前缀全部保留。
                envs = (index % self.num_envs).unique(sorted=True)
                selected = torch.zeros_like(valid)
                selected.flatten()[index] = True
                selected = selected[:, envs]
                obs = self.observations[:, envs].clone()
                obs['_loss_mask'] = selected.unsqueeze(-1)
                dones = self.dones[:, envs]
                padded, masks = split_and_pad_trajectories(obs, dones)
                starts = torch.ones_like(dones.squeeze(-1), dtype=torch.bool)
                starts[1:] = dones[:-1, :, 0].bool()
                hidden_states = []
                for saved in (self.saved_hidden_state_a, self.saved_hidden_state_c):
                    if saved is None:
                        hidden_states.append(None)
                    else:
                        states = tuple(state[:, :, envs].permute(2, 0, 1, 3)[starts.T]
                                       .transpose(0, 1).contiguous() for state in saved)
                        hidden_states.append(states[0] if len(states) == 1 else states)
                yield self.Batch(
                    observations=padded,
                    actions=self.actions[:, envs][selected].unsqueeze(0),
                    values=self.values[:, envs][selected].unsqueeze(0),
                    advantages=self.advantages[:, envs][selected].unsqueeze(0),
                    returns=self.returns[:, envs][selected].unsqueeze(0),
                    old_actions_log_prob=self.actions_log_prob[:, envs][selected].unsqueeze(0),
                    old_distribution_params=tuple(p[:, envs][selected].unsqueeze(0)
                                                  for p in self.distribution_params),
                    hidden_states=tuple(hidden_states), masks=masks)


def masked_update(child):
    """不共享参数的两个目标独立优化，避免 actor mask 删除有效 critic 样本。"""
    st = child.storage
    mini_batches, value_coef, entropy_coef = child.num_mini_batches, child.value_loss_coef, child.entropy_coef
    metrics = {}
    try:
        for kind in ('actor', 'critic'):
            count = int(getattr(st, kind + '_valid').sum().item())
            metrics[kind + '_samples'] = count
            if not count:
                continue
            st.sample_kind = kind
            child.actor.requires_grad_(kind == 'actor')
            child.critic.requires_grad_(kind == 'critic')
            child.num_mini_batches = min(mini_batches, count)
            child.value_loss_coef = value_coef if kind == 'critic' else 0.
            child.entropy_coef = entropy_coef if kind == 'actor' else 0.
            # PPO.clear 只移动写入游标，第二目标仍消费同一份不可变 rollout。
            metrics[kind] = child.update()
    finally:
        child.actor.requires_grad_(True)
        child.critic.requires_grad_(True)
        child.num_mini_batches, child.value_loss_coef, child.entropy_coef = mini_batches, value_coef, entropy_coef
    st.clear()
    return metrics
