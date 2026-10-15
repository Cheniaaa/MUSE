from types import SimpleNamespace as NS

import torch
import copy
import pytest

from test_returns import make_algorithm as make_ippo, transition as ippo_transition
from test_mappo import make_algorithm as make_mappo, transition as mappo_transition, assert_tree_equal


def metadata(cost_id='energy', scope='role', **overrides):
    return dict(cost_id=cost_id, scope=scope, units='J', statistic='episode_discounted_sum',
                boundary='world' if scope == 'world' else 'world_or_learning_done',
                **dict(dict(discount=.5, budget=1., lambda_initial=2., lambda_learning_rate=.25), **overrides))


def layout(world=False, **overrides):
    return {'roles': {'scout': {'energy': metadata(**overrides)}},
            'world': {'collision': metadata('collision', 'world', **overrides)} if world else {}}


def test_role_named_world_and_world_cost_have_distinct_owners():
    from muse_builtins.algorithms.mappo import MAPPO, MAPPOConfig
    learner = MAPPO(MAPPOConfig(hidden_dims=[8]).model_dump())
    learner.device = torch.device('cpu')
    learner.shapes = {'world': (2, 2)}
    learner.fields = {'world': {'target': (3,)}}
    learner.critic_fields = {('team', 'target'): (3,)}
    learner._initialize_costs({'roles': {'world': {'energy': metadata()}},
                              'world': {'energy': metadata('energy', 'world')}}, centralized=True)
    assert len(learner.cost_state) == 2
    assert {c.role for c in learner.cost_state.values()} == {'world', None}


def cost_transition(algorithm, obs, masks, step=0, value=2., terminal=False, timeout=False):
    fn = mappo_transition if hasattr(algorithm, 'owners') else ippo_transition
    tr = fn(obs, masks, step)
    tr.world_terminated.fill_(terminal)
    tr.world_truncated.fill_(timeout)
    costs = {'roles': {role: {key: torch.full(mask.present.shape, value)
                             for key in algorithm._cost_layout['roles'].get(role, {})}
                       for role, mask in masks.items() if role in algorithm._cost_layout['roles']},
             'world': {key: torch.full(tr.world_terminated.shape, value)
                       for key in algorithm._cost_layout['world']}}
    return tr._replace(costs=costs)


def collect_costs(algorithm, obs, masks, terminal=True, value=2.):
    for step in range(algorithm.collection_steps):
        result = algorithm.act(obs, masks, None, False)
        tr = cost_transition(algorithm, obs, masks, step, value=value,
                             terminal=terminal and step == algorithm.collection_steps - 1)
        algorithm.process_env_step(tr, result.act_aux)


@pytest.mark.parametrize('factory', [make_ippo, make_mappo])
def test_same_rollout_cost_changes_actual_actor_gradient(factory):
    left, obs, masks = factory()
    right, _, _ = factory(costs=layout(lambda_initial=3.))
    # cost模型初始化消费了额外随机数；对齐采样RNG以比较完全相同的实际rollout。
    right._cpu_rng = left._cpu_rng.clone()
    for step in range(left.collection_steps):
        for algorithm in (left, right):
            act = algorithm.act(obs, masks, None, False)
            algorithm.process_env_step(cost_transition(algorithm, obs, masks, step, value=step + 1., terminal=True), act.act_aux)
    for owner in left.children:
        assert_tree_equal(left.children[owner].storage.actions, right.children[owner].storage.actions)
    left.update()
    right.update()
    assert any(not torch.equal(p, q) for owner in left.children if owner.startswith('scout')
               for p, q in zip(left.children[owner].actor.parameters(), right.children[owner].actor.parameters()))


@pytest.mark.parametrize('recurrent', ['gru', 'lstm'])
@pytest.mark.parametrize('sharing', [True, False])
def test_world_cost_changes_actor_without_role_cost(recurrent, sharing):
    options = dict(recurrent={'type': recurrent, 'hidden_size': 7}, share_actor_parameters=sharing)
    left, obs, masks = make_mappo(costs=layout(world=True, lambda_initial=0.), **options)
    right, _, _ = make_mappo(costs=layout(world=True, lambda_initial=3.), **options)
    for learner in (left, right):
        collect_costs(learner, obs, masks)
        learner.update()
    assert any(not torch.equal(p, q) for key in left.children if key.startswith('guard')
               for p, q in zip(left.children[key].actor.parameters(), right.children[key].actor.parameters()))


def test_two_role_costs_keep_units_and_old_lambdas_in_actor_objective():
    costs = layout()
    costs['roles']['scout']['risk'] = metadata('risk', discount=.9, lambda_initial=.3)
    algorithm, obs, masks = make_ippo(costs=costs)
    for step in range(4):
        act = algorithm.act(obs, masks, None, False)
        tr = cost_transition(algorithm, obs, masks, step, terminal=True)
        tr.costs['roles']['scout']['risk'].fill_(10. * (step + 1.))
        algorithm.process_env_step(tr, act.act_aux)
    from muse_builtins.algorithms.ippo import compute_returns
    st = algorithm.children['scout'].storage
    compute_returns(st, algorithm.config['gamma'], algorithm.config['lam'])
    expected = st.advantages.clone()
    for cost in algorithm.cost_state.values():
        cost.compute_returns()
        expected -= cost.multiplier * cost.storage.advantages
        cost.constrain_advantages('scout', st)
    torch.testing.assert_close(st.advantages, expected)
    energy, risk = algorithm.cost_state.values()
    assert not torch.equal(energy.storage.returns, risk.storage.returns)
    assert risk.storage.advantages.abs().max() > 10.


def test_lambda_crosses_H_and_uses_completed_episode_mean_once():
    algorithm, obs, masks = make_ippo(costs=layout())
    owner = algorithm.cost_state[('scout', 'energy')]
    collect_costs(algorithm, obs, masks, terminal=False, value=2.)
    algorithm.update()
    assert owner.multiplier.item() == 2.
    # 两个world，各自第6步terminal与第8步timeout；V固定4，手算真实起点折扣和。
    with torch.no_grad():
        for p in owner.critic.parameters():
            p.zero_()
        list(owner.critic.parameters())[-1].fill_(4.)
    for step in range(4):
        act = algorithm.act(obs, masks, None, False)
        tr = cost_transition(algorithm, obs, masks, value=2.)
        if step == 1:
            tr.learning_done['scout'][0, 0] = True
        if step == 3:
            tr.world_truncated[1] = True
        algorithm.process_env_step(tr, act.act_aux)
    expected = 2. + .25 * (((sum(2. * .5**t for t in range(6))) +
                            (sum(2. * .5**t for t in range(8)) + .5**8 * 4.)) / 2. - 1.)
    algorithm.update()
    assert owner.multiplier.item() == pytest.approx(expected)
    assert owner.episode_sum[0, 0].item() == 3.
    assert owner.completed_count == 0


@pytest.mark.parametrize('sharing', [True, False])
@pytest.mark.parametrize('recurrent', [None, 'gru', 'lstm'])
def test_cost_owner_scope_rnn_real_training_and_eval_freeze(sharing, recurrent):
    algorithm, obs, masks = make_mappo(costs=layout(world=True), share_actor_parameters=sharing,
        recurrent=None if recurrent is None else {'type': recurrent, 'hidden_size': 7})
    assert set(algorithm.cost_state) == {('scout', 'energy'), (None, 'collision')}
    owned = [id(p) for child in algorithm.children.values() for model in (child.actor, child.critic)
             for p in model.parameters()]
    for cost in algorithm.cost_state.values():
        owned.extend(id(p) for p in cost.critic.parameters())
        assert {id(p) for group in cost.optimizer.param_groups for p in group['params']} == {
            id(p) for p in cost.critic.parameters()}
    assert len(owned) == len(set(owned))
    before = {k: copy.deepcopy(c.critic.state_dict()) for k, c in algorithm.cost_state.items()}
    masks['scout'].controllable.zero_()
    for step in range(4):
        act = algorithm.act(obs, masks, None, False)
        live = {k: copy.deepcopy(c.critic.get_hidden_state()) for k, c in algorithm.cost_state.items()}
        tr = cost_transition(algorithm, obs, masks, value=2., terminal=step == 3)
        if step == 1:
            tr.learning_done['scout'][0, 1] = True
        final = copy.deepcopy(obs)
        final.critic_inputs['team']['target'].add_(10. + step)
        tr = tr._replace(pre_reset_next_observations=final)
        with torch.no_grad():
            final_values = {k: copy.deepcopy(c.critic)(c.observations(final, c.valid_mask(masks)))
                            for k, c in algorithm.cost_state.items()}
        algorithm.process_env_step(tr, act.act_aux)
        for k, c in algorithm.cost_state.items():
            torch.testing.assert_close(c.storage.final_values[step], final_values[k])
            oracle = copy.deepcopy(c.critic)
            oracle.reset(hidden_state=live[k])
            boundary = tr.world_terminated | tr.world_truncated
            if c.role:
                boundary = boundary[:, None] | tr.learning_done[c.role]
            oracle.reset(boundary.flatten())
            assert_tree_equal(oracle.get_hidden_state(), c.critic.get_hidden_state())
    role = algorithm.cost_state[('scout', 'energy')]
    world = algorithm.cost_state[(None, 'collision')]
    assert role.storage.values.shape == (4, 4, 1)
    assert world.storage.values.shape == (4, 2, 1)
    assert not torch.equal(role.storage.values[:, 0], role.storage.values[:, 1])
    assert not world.storage.terminated[1].any()
    assert role.storage.terminated[1, 1]
    algorithm.update()
    for k, c in algorithm.cost_state.items():
        assert c.optimizer.state
        assert any(not torch.equal(v, c.critic.state_dict()[n]) for n, v in before[k].items())
    assert world.multiplier.item() == pytest.approx(2. + .25 * (3.75 - 1.))
    algorithm.set_mode('EVALUATE')
    before = {k: (copy.deepcopy(c.critic.state_dict()), copy.deepcopy(c.optimizer.state_dict()),
                  c.multiplier.clone(), copy.deepcopy(c.critic.get_hidden_state()), c.episode_sum.clone())
              for k, c in algorithm.cost_state.items()}
    collect_costs(algorithm, obs, masks)
    for k, c in algorithm.cost_state.items():
        assert_tree_equal(before[k], (c.critic.state_dict(), c.optimizer.state_dict(), c.multiplier,
                                     c.critic.get_hidden_state(), c.episode_sum))


@pytest.mark.parametrize('mask', ['present', 'alive'])
@pytest.mark.parametrize('factory', [make_ippo, make_mappo])
def test_invalid_cost_cannot_change_critic_actor_or_lambda(factory, mask):
    left, obs, masks = factory(costs=layout())
    right, _, _ = factory(costs=layout())
    getattr(masks['scout'], mask)[0, 0] = False
    for step in range(4):
        for algorithm, invalid in ((left, 0.), (right, float('nan'))):
            act = algorithm.act(obs, masks, None, False)
            tr = cost_transition(algorithm, obs, masks, step, terminal=True)
            tr.costs['roles']['scout']['energy'][0, 0] = invalid
            algorithm.process_env_step(tr, act.act_aux)
    left.update()
    right.update()
    for owner in left.children:
        assert_tree_equal(left.children[owner].actor.state_dict(), right.children[owner].actor.state_dict())
    a, b = left.cost_state[('scout', 'energy')], right.cost_state[('scout', 'energy')]
    assert_tree_equal(a.critic.state_dict(), b.critic.state_dict())
    assert_tree_equal(a.multiplier, b.multiplier)
    assert a.multiplier.item() == pytest.approx(2.25)


def test_completely_invalid_episode_does_not_count_or_train():
    algorithm, obs, masks = make_ippo(costs=layout())
    cost = algorithm.cost_state[('scout', 'energy')]
    before = copy.deepcopy(cost.critic.state_dict())
    masks['scout'].alive.zero_()
    collect_costs(algorithm, obs, masks)
    algorithm.update()
    assert cost.multiplier.item() == 2.
    assert not cost.optimizer.state
    assert_tree_equal(before, cost.critic.state_dict())


def test_named_cost_returns_use_terminal_and_timeout_boundaries():
    from muse_builtins.algorithms.ippo import compute_cost_returns

    costs = torch.tensor([1., 2., 3., 4.]).reshape(4, 1, 1)
    values = torch.tensor([.2, .4, .6, .8]).reshape(4, 1, 1)
    final_values = torch.tensor([2., 7., 11., 13.]).reshape(4, 1, 1)
    terminated = torch.tensor([False, True, False, False]).reshape(4, 1, 1)
    truncated = torch.tensor([False, False, False, True]).reshape(4, 1, 1)
    returns, advantages = compute_cost_returns(costs, values, final_values, terminated, truncated, .9, .8)

    expected = []
    for start in range(4):
        total, discount = values[start].item(), 1.
        for end in range(start, 4):
            total += discount * (costs[end].item() + (.9 * final_values[end].item()
                                                        if not terminated[end] else 0.) - values[end].item())
            if terminated[end] or truncated[end]:
                break
            discount *= .9 * .8
        expected.append(total)
    torch.testing.assert_close(returns.flatten(), torch.tensor(expected))
    torch.testing.assert_close(advantages.flatten(), returns.flatten() - values.flatten())


def test_lambda_update_is_projected_and_uses_budget_error():
    from muse_builtins.algorithms.ippo import update_lambda

    assert update_lambda(torch.tensor(1.), torch.tensor(3.), 2., .5).item() == 1.5
    assert update_lambda(torch.tensor(.2), torch.tensor(1.), 2., .5).item() == 0.
