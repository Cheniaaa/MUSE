"""独立手算：mask切断GAE，lambda从episode起点累计而非H起点。"""
import pytest
import torch
from types import SimpleNamespace as NS

from muse_builtins.algorithms.ippo import compute_cost_returns


@pytest.mark.parametrize('gamma', [0., .5, 1.])
@pytest.mark.parametrize('lam', [0., .8, 1.])
def test_cost_returns_scalar_oracle_with_invalid_and_boundaries(gamma, lam):
    c, v, final = [1., 99., 3., 4., 5.], [.2, 888., .6, .8, 1.], [2., 777., 11., 13., 17.]
    valid = [True, False, True, True, True]
    term, trunc = [False, False, True, False, False], [False, False, False, False, True]
    def tensor(data, dtype=None):
        return torch.tensor(data, dtype=dtype).reshape(5, 1, 1)
    returns, advantages = compute_cost_returns(tensor(c), tensor(v), tensor(final),
        tensor(term), tensor(trunc), gamma, lam, valid=tensor(valid))
    expected = []
    for start in range(5):
        total, weight = 0., 1.
        for end in range(start, 5):
            if not valid[end]:
                break
            total += weight * (c[end] + (0. if term[end] else gamma * final[end]) - v[end])
            if term[end] or trunc[end]:
                break
            weight *= gamma * lam
        expected.append(total)
    torch.testing.assert_close(advantages.flatten(), torch.tensor(expected))
    torch.testing.assert_close(returns.flatten()[tensor(valid).flatten()],
                               (torch.tensor(expected) + torch.tensor(v))[tensor(valid).flatten()])


def test_invalid_nan_cost_does_not_leak_into_valid_gae():
    c = torch.tensor([1., float('nan'), 3.]).reshape(3, 1, 1)
    zero = torch.zeros_like(c)
    mask = torch.tensor([True, False, True]).reshape(3, 1, 1)
    r, a = compute_cost_returns(c, zero, zero, ~mask, ~mask, .9, .95, valid=mask)
    torch.testing.assert_close(a.flatten(), torch.tensor([1., 0., 3.]))
    assert torch.isfinite(r).all()


@pytest.mark.parametrize('scope', ['role', 'world'])
@pytest.mark.parametrize('recurrent', [None, 'gru', 'lstm'])
def test_episode_oracle_and_cost_regression_without_environment(scope, recurrent):
    """直接测试cost owner的数值消费者；不构造或替代SDK Experiment。"""
    from muse_builtins.algorithms._costs import CostLearner
    from muse_builtins.algorithms.mappo import MAPPOConfig
    from muse_sdk.runtime_types import Observations, RoleMask, TransitionView
    cfg = MAPPOConfig(collection_steps=2, hidden_dims=[8], num_learning_epochs=1, num_mini_batches=2,
        recurrent=None if recurrent is None else {'type': recurrent, 'hidden_size': 5}).model_dump()
    algorithm = NS(shapes={'scout': (2, 2)}, fields={'scout': {'x': (3,)}}, critic_fields={('team', 'x'): (3,)},
                   device=torch.device('cpu'), config=cfg, collection_steps=2,
                   owners={'scout': ('scout', None), 'scout/slot/1': ('scout', 1)})
    metadata = dict(cost_id='energy', scope=scope, units='J', discount=.5, budget=1., lambda_initial=2.,
                    lambda_learning_rate=.25, statistic='episode_discounted_sum',
                    boundary='world' if scope == 'world' else 'world_or_learning_done')
    owner = CostLearner(algorithm, 'scout' if scope == 'role' else None, metadata, centralized=True)
    with torch.no_grad():
        for p in owner.critic.parameters():
            p.zero_()
        owner.critic.mlp[-1].bias.fill_(4.)
    obs = Observations({'scout': {'x': torch.ones(2, 2, 3)}}, {'team': {'x': torch.ones(2, 3)}})
    masks = {'scout': RoleMask(*(torch.ones(2, 2, dtype=torch.bool) for _ in range(3)))}
    expected_parameters = [p.clone() for p in owner.critic.parameters()]
    for step in range(4):
        with torch.no_grad():
            owner.act(obs, masks)
            costs = {'roles': {'scout': {'energy': torch.full((2, 2), 2.)}} if scope == 'role' else {},
                     'world': {'energy': torch.full((2,), 2.)} if scope == 'world' else {}}
            terminal = torch.tensor([step == 3, False])
            timeout = torch.tensor([False, step == 3])
            tr = TransitionView(masks, obs, {}, None, terminal, timeout, torch.zeros(2, dtype=torch.long),
                                {'scout': torch.zeros(2, 2, dtype=torch.bool)}, terminal | timeout, costs, wireless=None)
            owner.process_env_step(tr)
        if step == 1:
            owner.compute_returns()
            owner.update()
            assert owner.multiplier.item() == 2.
            torch.testing.assert_close(owner.episode_sum, torch.full(owner.shape, 3.))
            # 保持后半段timeout V=4以便独立手算。
            with torch.no_grad():
                for p in owner.critic.parameters():
                    p.zero_()
                owner.critic.mlp[-1].bias.fill_(4.)
    assert owner.completed_count == (4 if scope == 'role' else 2)
    # terminal=2+1+.5+.25=3.75；timeout再补.5^4*4=.25。
    assert owner.completed_sum.item() / owner.completed_count == pytest.approx(3.875)
    owner.compute_returns()
    # actor消费原单位advantage和本block更新前lambda；共享/独立slot都只投影样本轴。
    raw = owner.storage.advantages.reshape(2, *owner.shape)
    expected = raw if scope == 'role' else raw[:, :, None].expand(2, 2, 2)
    for actor, n in [('scout', 2), ('scout/slot/1', 1)]:
        actor_storage = NS(advantages=torch.ones(2, 2 * n, 1))
        owner.constrain_advantages(actor, actor_storage)
        selected = expected if n == 2 else expected[:, :, 1:2]
        torch.testing.assert_close(actor_storage.advantages, 1. - 2. * selected.reshape_as(actor_storage.advantages))
    owner.update()
    assert owner.multiplier.item() == pytest.approx(2.71875)
    assert any(not torch.equal(old, new) for old, new in zip(expected_parameters, owner.critic.parameters()))
    assert owner.optimizer.state
