from types import SimpleNamespace as NS

import torch

from muse_sdk.runtime_types import Observations, RoleMask, TransitionView
from muse_runtime.checkpoint import snapshot_learning_state
from test_returns import make_algorithm


def _cost_transition(obs, masks, step):
    batch = next(iter(masks.values())).present.shape[0]
    terminated = torch.tensor([False, step == 1])
    truncated = torch.tensor([step == 3, False])
    costs = {'roles': {role: {'energy': torch.full(mask.present.shape, .5 + step)}
                       for role, mask in masks.items()}, 'world': {}}
    return TransitionView(masks, obs,
                          {role: torch.zeros_like(mask.present) for role, mask in masks.items()}, None,
                          terminated, truncated, torch.zeros(batch, dtype=torch.long),
                          {role: torch.zeros_like(mask.present) for role, mask in masks.items()},
                          terminated | truncated, costs, wireless=None)


def test_ippo_named_cost_updates_critic_actor_and_lambda():
    costs = {'roles': {
        'scout': {'energy': {'cost_id': 'energy', 'scope': 'role', 'discount': .9,
                             'units': 'J', 'statistic': 'episode_discounted_sum', 'boundary': 'world_or_learning_done',
                             'budget': .1, 'lambda_initial': .2, 'lambda_learning_rate': .5}},
        'guard': {'energy': {'cost_id': 'energy', 'scope': 'role', 'discount': .9,
                             'units': 'J', 'statistic': 'episode_discounted_sum', 'boundary': 'world_or_learning_done',
                             'budget': .1, 'lambda_initial': .2, 'lambda_learning_rate': .5}},
    }, 'world': {}}
    algorithm, obs, masks = make_algorithm(costs=costs)
    assert set(algorithm.cost_state) == {('scout', 'energy'), ('guard', 'energy')}
    before_actor = {key: snapshot_learning_state(child.actor.state_dict())
                    for key, child in algorithm.children.items()}
    before_cost = {key: snapshot_learning_state(cost.critic.state_dict())
                   for key, cost in algorithm.cost_state.items()}
    for step in range(algorithm.collection_steps):
        result = algorithm.act(obs, masks, None, False)
        algorithm.process_env_step(_cost_transition(obs, masks, step), result.act_aux)
    metrics = algorithm.update()
    assert metrics.keys() == algorithm.children.keys()
    assert set(algorithm.cost_state) == {('scout', 'energy'), ('guard', 'energy')}
    assert all(value.multiplier.item() > .2 for value in algorithm.cost_state.values())
    for key, child in algorithm.children.items():
        assert any(not torch.equal(value, child.actor.state_dict()[name])
                   for name, value in before_actor[key].items())
        cost = algorithm.cost_state[(key, 'energy')]
        assert any(not torch.equal(value, cost.critic.state_dict()[name])
                   for name, value in before_cost[(key, 'energy')].items())


def test_cost_free_ippo_has_no_cost_state():
    algorithm, _, _ = make_algorithm()
    assert algorithm.cost_state == {}
    assert all(not hasattr(child, 'cost_critics') for child in algorithm.children.values())
