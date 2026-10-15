"""真实 RSL PPO 的 recurrent 投影回归；全部在 CPU 执行。"""
import copy
import math

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from muse_builtins.algorithms._recurrent import MaskedRNNModel
from muse_builtins.algorithms._storage import MaskedStorage, masked_update


def clone_hidden(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(x.clone() for x in state)
    return state.clone()


def assert_hidden(actual, expected):
    if isinstance(expected, tuple):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def make_child(kind, horizon=5, envs=3):
    torch.manual_seed(37)
    obs = TensorDict({'actor': torch.zeros(envs, 2), 'critic': torch.zeros(envs, 3)}, [envs])
    groups = {'actor': ['actor'], 'critic': ['critic']}
    cls = MaskedRNNModel
    kw = dict(hidden_dims=[8], rnn_type=kind, rnn_hidden_dim=7)
    actor = cls(obs, groups, 'actor', 2, **kw,
                distribution_cfg={'class_name': 'rsl_rl.modules.distribution:BetaDistribution',
                                  'action_range': (0., 1.)})
    critic = cls(obs, groups, 'critic', 1, **kw)
    st = MaskedStorage('rl', envs, horizon, obs, [2], 'cpu')
    return PPO(actor, critic, st, num_learning_epochs=2, num_mini_batches=2,
               schedule='fixed', device='cpu', entropy_coef=.01)


@torch.no_grad()
def collect(child, inactive=False):
    st = child.storage
    for t in range(st.num_transitions_per_env):
        tr = st.Transition()
        n = st.num_envs
        tr.observations = TensorDict({'actor': torch.randn(n, 2), 'critic': torch.randn(n, 3)}, [n])
        tr.hidden_states = (clone_hidden(child.actor.get_hidden_state()),
                            clone_hidden(child.critic.get_hidden_state()))
        tr.actions = child.actor(tr.observations, stochastic_output=True)
        tr.values = child.critic(tr.observations)
        tr.actions_log_prob = child.actor.get_output_log_prob(tr.actions)
        tr.distribution_params = child.actor.output_distribution_params
        tr.rewards = torch.arange(n).float() + t + 1
        tr.dones = (torch.arange(n) + t) % 3 == 1
        st.add_transition(tr)
        st.actor_valid[t, :, 0] = False if inactive else (torch.arange(n) + t) % 3 != 0
        st.critic_valid[t, :, 0] = (torch.arange(n) + t) % 4 != 0
        child.actor.reset(tr.dones)
        child.critic.reset(tr.dones)
    st.returns.copy_(st.values + torch.arange(st.values.numel()).reshape_as(st.values)/7 + 1)
    st.advantages.copy_(st.returns - st.values)


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
@pytest.mark.parametrize('target', ['actor', 'critic'])
@pytest.mark.parametrize('warm_start', [False, True])
def test_recurrent_batches_project_each_sample_once_and_replay_stored_ratio(kind, target, warm_start):
    child = make_child(kind)
    if warm_start:
        collect(child)
        child.storage.clear()
    collect(child)
    st = child.storage
    st.sample_kind = target
    valid = getattr(st, target + '_valid').squeeze(-1)
    # returns 作为唯一样本 ID，验证 mask 不连续与非整除尾部。
    st.returns.copy_(torch.arange(15).reshape(5, 3, 1))
    before = [clone_hidden(m.get_hidden_state()) for m in (child.actor, child.critic)]
    batches = list(st.recurrent_mini_batch_generator(3, 2))
    assert len(batches) == 6
    expected = st.returns[valid].flatten().tolist()
    for epoch in range(2):
        seen = []
        for batch in batches[epoch*3:(epoch+1)*3]:
            assert batch.actions.shape[:1] == (1,)
            assert batch.actions.shape[1] > 0
            seen.extend(batch.returns.flatten().tolist())
            child.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                        stochastic_output=True)
            ratio = (child.actor.get_output_log_prob(batch.actions) - batch.old_actions_log_prob.flatten()).exp()
            torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=2e-6, rtol=2e-6)
            values = child.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            torch.testing.assert_close(values, batch.values, atol=1e-7, rtol=1e-6)
        assert sorted(seen) == sorted(expected)
    for m, h in zip((child.actor, child.critic), before):
        assert_hidden(m.get_hidden_state(), h)


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_update_trains_rnn_without_mutating_hidden_and_freezes_inactive_adam(kind):
    child = make_child(kind)
    collect(child)
    before = [copy.deepcopy(m.state_dict()) for m in (child.actor, child.critic)]
    hidden = [clone_hidden(m.get_hidden_state()) for m in (child.actor, child.critic)]
    metrics = masked_update(child)
    assert metrics['actor_samples'] == 10
    assert metrics['critic_samples'] == 11
    for model, state, h in zip((child.actor, child.critic), before, hidden):
        assert any(not torch.equal(p, model.state_dict()[name]) for name, p in state.items() if 'rnn' in name)
        assert_hidden(model.get_hidden_state(), h)
    for result in (metrics['actor'], metrics['critic']):
        assert all(math.isfinite(x) for x in result.values())
    before_actor = copy.deepcopy(child.actor.state_dict())
    before_critic = copy.deepcopy(child.critic.state_dict())
    collect(child, inactive=True)
    child.storage.critic_valid.fill_(True)
    metrics = masked_update(child)
    assert metrics['actor_samples'] == 0
    assert metrics['critic_samples'] == 15
    assert 'actor' not in metrics
    for name, p in before_actor.items():
        torch.testing.assert_close(p, child.actor.state_dict()[name], atol=0, rtol=0)
    assert any(not torch.equal(p, child.critic.state_dict()[name]) for name, p in before_critic.items())
    collect(child, inactive=True)
    child.storage.critic_valid.zero_()
    assert list(child.storage.recurrent_mini_batch_generator(2, 1)) == []
    before = copy.deepcopy(child.optimizer.state_dict())
    assert masked_update(child) == {'actor_samples': 0, 'critic_samples': 0}
    for key, state in before['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(child.optimizer.state_dict()['state'][key][name], value, atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
@pytest.mark.parametrize('episode_boundary', [False, True])
def test_loss_at_last_step_backpropagates_through_inactive_prefix(kind, episode_boundary):
    child = make_child(kind, envs=1)
    collect(child)
    st = child.storage
    st.dones.zero_()
    if episode_boundary:
        st.dones[1] = True
    st.actor_valid.zero_()
    st.actor_valid[-1] = True
    batch = next(st.recurrent_mini_batch_generator(1, 1))
    batch.observations['actor'].requires_grad_()
    output = child.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0])
    assert output.shape == (1, 1, 2)
    output.sum().backward()
    gradient = batch.observations['actor'].grad
    assert gradient[0, -1].abs().sum() > 0
    if episode_boundary:
        assert torch.count_nonzero(gradient[:, 0]) == 0


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_single_step_with_no_saved_hidden_and_many_minibatches(kind):
    child = make_child(kind, horizon=1, envs=3)
    collect(child)
    assert child.storage.saved_hidden_state_a is None
    child.num_mini_batches = 9
    metrics = masked_update(child)
    assert metrics['actor_samples'] == 2
    assert metrics['critic_samples'] == 2
    assert child.num_mini_batches == 9
    assert all(math.isfinite(v) for v in metrics['critic'].values())


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_missing_initial_hidden_after_reset_does_not_reuse_previous_rollout(kind):
    child = make_child(kind)
    collect(child)
    child.storage.clear()
    # 第二个 rollout 保存非零起点，以便第三次能识别 stale buffer。
    collect(child)
    child.storage.clear()
    child.actor.reset()
    child.critic.reset()
    collect(child)
    for batch in child.storage.recurrent_mini_batch_generator(3, 1):
        child.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                    stochastic_output=True)
        log_prob = child.actor.get_output_log_prob(batch.actions).flatten()
        torch.testing.assert_close(log_prob, batch.old_actions_log_prob.flatten(), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_masked_targets_cannot_change_corresponding_parameters(kind):
    left = make_child(kind)
    collect(left)
    right = copy.deepcopy(left)
    # critic pass 中的 actor 冻结；actor pass 的 value loss 权重为零。
    right.storage.advantages[~right.storage.actor_valid] = 10000
    right.storage.returns[~right.storage.critic_valid] = -10000
    rng = torch.random.get_rng_state()
    masked_update(left)
    torch.random.set_rng_state(rng)
    masked_update(right)
    for a, b in zip((left.actor, left.critic), (right.actor, right.critic)):
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['gru', 'lstm'])
def test_batch_metrics_include_every_entropy_sample_and_correct_denominator(kind):
    child = make_child(kind, envs=7)
    collect(child)
    child.num_learning_epochs = 1
    child.learning_rate = 0.
    for group in child.optimizer.param_groups:
        group['lr'] = 0.
    st = child.storage
    rng = torch.random.get_rng_state()
    expected = {}
    counts = {}
    for target in ('actor', 'critic'):
        st.sample_kind = target
        entropies = []
        for batch in st.recurrent_mini_batch_generator(child.num_mini_batches, 1):
            child.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0],
                        stochastic_output=True)
            entropies.append(child.actor.output_entropy.mean().item())
        expected[target] = sum(entropies)/len(entropies)
        counts[target] = len(entropies)
    torch.random.set_rng_state(rng)
    metrics = masked_update(child)
    assert counts['actor'] == child.num_mini_batches
    for target in ('actor', 'critic'):
        assert metrics[target]['entropy'] == pytest.approx(expected[target], abs=1e-7)


def test_recurrent_update_keeps_requested_batch_count_when_samples_exceed_horizon():
    child = make_child('gru', envs=7)
    collect(child)
    batches = list(child.storage.recurrent_mini_batch_generator(2, 1))
    assert len(batches) == 2
    assert sum(batch.actions.numel()//2 for batch in batches) == int(child.storage.actor_valid.sum())
