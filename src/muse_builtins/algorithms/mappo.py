"""合作 MAPPO：局部 actor、集中 critic；IPPO 提供共同学习生命周期。"""
import hashlib
import math
from copy import deepcopy
from typing import Literal

from pydantic import Field

from muse_sdk.runtime_types import ActResult
from muse_sdk.config import DataModel
from .ippo import IPPO, IPPOConfig


class RecurrentConfig(DataModel):
    type: Literal['gru', 'lstm']
    hidden_size: int = Field(default=64, gt=0)


class MAPPOConfig(IPPOConfig):
    share_actor_parameters: bool = True
    recurrent: RecurrentConfig | None = None


class MAPPO(IPPO):
    def __init__(self, config):
        super().__init__(config)
        self.owners = {}
        self.critic_fields = {}

    @classmethod
    def describe(cls):
        return dict(super().describe(), algorithm_id='mappo', config_schema=MAPPOConfig,
                    critic_inputs='all_centralized', reward_inputs=['world'], cost_inputs=['role', 'world'])

    def initialize(self, experiment, context, rng):
        import torch
        from tensordict import TensorDict
        from rsl_rl.algorithms import PPO
        from rsl_rl.models import MLPModel
        from ._storage import MaskedStorage

        if self.children:
            raise RuntimeError('MAPPO is already initialized')
        if experiment.training_schedule.collection_steps != self.collection_steps:
            raise ValueError('TrainingSchedule collection_steps must match MAPPO storage H')
        self.device = torch.device(context.device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        seed = int.from_bytes(hashlib.sha256(f'{experiment.root_seed}:mappo:learner'.encode()).digest()[:8], 'little')
        self._cpu_rng = torch.Generator(device='cpu').manual_seed(seed).get_state()
        self._cuda_rng = (torch.Generator(device=self.device).manual_seed(seed).get_state()
                          if self.device.type == 'cuda' else None)
        self.action_specs = experiment.action_specs
        self.critic_fields = {(item.path[2], item.path[3]): tuple(item.tensor.shape[1:])
                              for items in experiment.algorithm_inputs.critics.values() for item in items}
        if not self.critic_fields:
            raise ValueError('MAPPO requires centralized critic inputs')
        model_type = MLPModel
        model_options = {'hidden_dims': self.config['hidden_dims']}
        if self.config['recurrent'] is not None:
            from ._recurrent import MaskedRNNModel
            model_type = MaskedRNNModel
            model_options.update(rnn_type=self.config['recurrent']['type'],
                                 rnn_hidden_dim=self.config['recurrent']['hidden_size'], rnn_num_layers=1)
        with self._learner_rng():
            for role, spec in experiment.action_specs.items():
                b, n, width = spec.shape
                self.shapes[role] = (b, n)
                self.fields[role] = {item.path[3]: tuple(item.tensor.shape[2:])
                                     for item in experiment.algorithm_inputs.actors[role]}
                if not self.fields[role]:
                    raise ValueError(f'MAPPO requires actor observations for {role}')
                self.bounds[role] = (torch.tensor(spec.minimum, device=self.device),
                                     torch.tensor(spec.maximum, device=self.device))
                slots = [None] if self.config['share_actor_parameters'] else range(n)
                for slot in slots:
                    owner = role if slot is None else f'{role}/slot/{slot}'
                    self.owners[owner] = (role, slot)
                    count = b * (n if slot is None else 1)
                    obs = TensorDict({'actor': torch.zeros(count, sum(map(math.prod, self.fields[role].values())),
                                                          device=self.device),
                                      'critic': torch.zeros(count, sum(map(math.prod, self.critic_fields.values())),
                                                           device=self.device)}, [count], device=self.device)
                    groups = {'actor': ['actor'], 'critic': ['critic']}
                    width, distribution = self._action_model_options(role)
                    actor = model_type(obs, groups, 'actor', width, **model_options,
                                      distribution_cfg=distribution)
                    critic = model_type(obs, groups, 'critic', 1, **model_options)
                    storage = MaskedStorage('rl', count, self.collection_steps, obs, [width], str(self.device))
                    ppo = {key: self.config[key] for key in IPPOConfig.model_fields
                           if key not in ('collection_steps', 'hidden_dims')}
                    self.children[owner] = PPO(actor, critic, storage, **ppo, schedule='fixed', device=str(self.device))
            self._initialize_costs(experiment.costs.model_dump(), centralized=True)
        self._clear_hidden_states()
        self.set_mode(self.mode)

    def _clear_hidden_states(self):
        """真实零初态供首 transition 保存；不通过一次虚假 forward 初始化。"""
        import torch
        for child in self.children.values():
            for model in (child.actor, child.critic):
                if model.is_recurrent:
                    rnn = model.rnn.rnn
                    zero = next(model.parameters()).new_zeros(rnn.num_layers, child.storage.num_envs, rnn.hidden_size)
                    hidden = (zero, torch.zeros_like(zero)) if self.config['recurrent']['type'] == 'lstm' else zero
                    model.reset(hidden_state=hidden)

    def reset_worlds(self, mask):
        for owner, child in self.children.items():
            role, _ = self.owners[owner]
            boundary = self._select_slots(mask[:, None].expand(self.shapes[role]), owner).reshape(-1)
            child.actor.reset(boundary)
            child.critic.reset(boundary)

    def _select_slots(self, value, owner):
        _, slot = self.owners[owner]
        return value if slot is None else value[:, slot:slot + 1]

    def _observations(self, observations, owner):
        import torch
        from tensordict import TensorDict
        role, slot = self.owners[owner]
        b, n = self.shapes[role]
        count = n if slot is None else 1
        actor = torch.cat([self._select_slots(observations.actors[role][key], owner).reshape(b * count, -1)
                           for key in self.fields[role]], dim=-1)
        central = torch.cat([observations.critic_inputs[group][key].reshape(b, -1)
                             for group, key in self.critic_fields], dim=-1)
        critic = central[:, None, :].expand(b, count, -1).reshape(b * count, -1)
        return TensorDict({'actor': actor.detach().clone(), 'critic': critic.detach().clone()},
                          [b * count], device=self.device)

    def act(self, observations, role_masks, rng, deterministic):
        import torch
        if not self.children:
            raise RuntimeError('Initialize MAPPO first')
        if any(c.storage.pending is not None or c.storage.step == self.collection_steps for c in self.children.values()):
            raise RuntimeError('Consume pending decision or update full rollout before act')
        self._sequence += 1
        token = self._sequence
        actions = {}
        with self._learner_rng(), torch.no_grad():
            for owner, child in self.children.items():
                role, slot = self.owners[owner]
                obs = self._observations(observations, owner)
                if self.mode == 'TRAIN':
                    raw = child.act(obs)
                    if deterministic:
                        raw = self._deterministic_raw(role, child.actor)
                        child.transition.actions = raw
                        child.transition.actions_log_prob = child.actor.get_output_log_prob(raw).detach()
                    child.storage.pending = child.transition
                    child.storage.pending_aux = token
                    child.transition = type(child.transition)()
                else:
                    raw = child.actor(obs, stochastic_output=False)
                b, n = self.shapes[role]
                if role not in actions:
                    actions[role] = torch.empty(b, n, raw.shape[-1], device=self.device)
                role_raw = raw.reshape(b, -1, raw.shape[-1])
                if slot is None:
                    actions[role].copy_(role_raw)
                else:
                    actions[role][:, slot:slot + 1].copy_(role_raw)
            if self.mode == 'TRAIN':
                for cost in self.cost_state.values():
                    cost.act(observations, role_masks)
        return ActResult({role: self._role_action(role, value) for role, value in actions.items()}, token)

    def process_env_step(self, transition, act_aux):
        import torch
        if self.mode != 'TRAIN':
            for owner, child in self.children.items():
                role, _ = self.owners[owner]
                boundary = ((transition.world_terminated | transition.world_truncated)[:, None]
                            | transition.learning_done[role])
                boundary = self._select_slots(boundary, owner).reshape(-1)
                child.actor.reset(boundary)
                child.critic.reset(boundary)
            return
        if any(c.storage.pending is None or c.storage.pending_aux != act_aux for c in self.children.values()):
            raise RuntimeError('Missing or consumed pending act_aux')
        if transition.world_reward is None:
            raise ValueError('MAPPO requires explicit world_reward')
        with torch.no_grad():
            for cost in self.cost_state.values():
                cost.process_env_step(transition)
            for owner, child in self.children.items():
                role, _ = self.owners[owner]
                st = child.storage
                obs = self._observations(transition.pre_reset_next_observations, owner)
                terminal = self._select_slots(transition.world_terminated[:, None] | transition.learning_done[role], owner)
                truncated = self._select_slots(transition.world_truncated[:, None].expand(self.shapes[role]), owner)
                masks = transition.act_masks[role]
                hidden = child.critic.get_hidden_state()
                try:
                    st.final_values[st.step].copy_(child.critic(obs))
                finally:
                    child.critic.reset(hidden_state=hidden)
                st.terminated[st.step].copy_(terminal.reshape(-1, 1))
                st.truncated[st.step].copy_(truncated.reshape(-1, 1))
                st.actor_valid[st.step].copy_(self._select_slots(
                    masks.present & masks.alive & masks.controllable, owner).reshape(-1, 1))
                st.critic_valid[st.step].copy_(self._select_slots(masks.present, owner).reshape(-1, 1))
                rewards = transition.world_reward[:, None].expand(self.shapes[role])
                child.transition = st.pending
                child.process_env_step(obs, self._select_slots(rewards, owner).reshape(-1),
                                       (terminal | truncated).reshape(-1), extras={})
                st.pending = None
                st.pending_aux = None

    def _state_signature(self):
        return dict(super()._state_signature(), algorithm='mappo', owners=deepcopy(self.owners),
                    critics=list(self.critic_fields.items()))

    def load_learning_state_dict(self, state):
        super().load_learning_state_dict(state)
        self._clear_hidden_states()
