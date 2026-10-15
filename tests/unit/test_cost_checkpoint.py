import copy
import io

import pytest
import torch

from muse_runtime.checkpoint import snapshot_learning_state
from test_cost_learning import layout, make_ippo, make_mappo, collect_costs, cost_transition
from test_mappo import assert_tree_equal


def test_checkpoint_hook_rejects_unexpected_cost_state():
    # 直接覆盖hook的布局边界，不依赖并行Task1的Experiment构造。
    from muse_builtins.algorithms.ippo import IPPO, IPPOConfig
    learner = IPPO(IPPOConfig().model_dump())
    learner._cpu_rng = torch.random.get_rng_state().clone()
    learner._cuda_rng = None
    saved = snapshot_learning_state(learner.learning_state_dict())
    saved['costs'] = {'world/unknown': {}}
    with pytest.raises(ValueError, match='[Cc]ost'):
        learner.load_learning_state_dict(saved)


@pytest.mark.parametrize('recurrent', [None, 'gru', 'lstm'])
@pytest.mark.parametrize('sharing', [True, False])
def test_cost_checkpoint_roundtrip_new_episode_and_training(recurrent, sharing):
    options = dict(costs=layout(world=True), share_actor_parameters=sharing,
                   recurrent=None if recurrent is None else {'type': recurrent, 'hidden_size': 7})
    algorithm, obs, masks = make_mappo(**options)
    collect_costs(algorithm, obs, masks)
    algorithm.update()
    collect_costs(algorithm, obs, masks, terminal=False)
    algorithm.update()
    state = snapshot_learning_state(algorithm.learning_state_dict())
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    fresh, fresh_obs, fresh_masks = make_mappo(batch=3, **options)
    # 新环境也可能已采过半段：load必须清除pending/episode/block/hidden。
    act = fresh.act(fresh_obs, fresh_masks, None, False)
    fresh.process_env_step(cost_transition(fresh, fresh_obs, fresh_masks, terminal=True), act.act_aux)
    fresh.act(fresh_obs, fresh_masks, None, False)
    fresh.load_learning_state_dict(state)
    assert_tree_equal(state, fresh.learning_state_dict())
    for key, cost in fresh.cost_state.items():
        assert_tree_equal(algorithm.cost_state[key].critic.state_dict(), cost.critic.state_dict())
        assert_tree_equal(algorithm.cost_state[key].optimizer.state_dict(), cost.optimizer.state_dict())
        assert_tree_equal(algorithm.cost_state[key].multiplier, cost.multiplier)
        assert cost.storage.step == 0 and cost.storage.pending is None
        assert not cost.episode_sum.any() and not cost.episode_valid.any()
        assert torch.all(cost.episode_discount == 1.)
        assert cost.completed_count == 0 and cost.completed_sum == 0
        if recurrent:
            hidden = cost.critic.get_hidden_state()
            for tensor in hidden if isinstance(hidden, tuple) else (hidden,):
                assert not tensor.any()
    restored, _, _ = make_mappo(batch=3, **options)
    restored.load_learning_state_dict(state)
    for learner in (fresh, restored):
        collect_costs(learner, fresh_obs, fresh_masks)
        learner.update()
    assert_tree_equal(fresh.learning_state_dict(), restored.learning_state_dict())
    # Runtime snapshot必须与后续live参数更新隔离。
    assert any(not torch.equal(value, fresh.learning_state_dict()['costs'][key]['critic'][name])
               for key, saved in state['costs'].items() for name, value in saved['critic'].items())


@pytest.mark.parametrize('corrupt', ['missing_cost', 'extra_cost', 'missing_all', 'budget', 'scope', 'discount',
                                   'units', 'boundary', 'shape', 'optimizer_shape', 'lambda_shape', 'lambda_negative',
                                   'missing_lr', 'invalid_betas', 'lr_type'])
def test_cost_checkpoint_mismatch_is_rejected_before_mutation(corrupt):
    algorithm, obs, masks = make_ippo(costs=layout())
    collect_costs(algorithm, obs, masks)
    algorithm.update()
    state = snapshot_learning_state(algorithm.learning_state_dict())
    assert 'costs' in state
    saved = next(iter(state['costs'].values()))
    if corrupt == 'missing_cost':
        state['costs'].clear()
    elif corrupt == 'extra_cost':
        state['costs']['other'] = copy.deepcopy(saved)
    elif corrupt == 'missing_all':
        state.pop('costs')
    elif corrupt in ('budget', 'scope', 'discount', 'units', 'boundary'):
        saved['metadata'][corrupt] = {'budget': 99., 'scope': 'world', 'discount': .9,
                                      'units': 'm', 'boundary': 'world'}[corrupt]
    elif corrupt == 'shape':
        saved['critic'][next(iter(saved['critic']))] = torch.zeros(1)
    elif corrupt == 'optimizer_shape':
        next(iter(saved['optimizer']['state'].values()))['exp_avg'] = torch.zeros(1)
    elif corrupt == 'lambda_shape':
        saved['lambda'] = torch.ones(2)
    elif corrupt == 'missing_lr':
        saved['optimizer']['param_groups'][0].pop('lr')
    elif corrupt == 'invalid_betas':
        saved['optimizer']['param_groups'][0]['betas'] = (1.5, .999)
    elif corrupt == 'lr_type':
        saved['optimizer']['param_groups'][0]['lr'] = '0.0003'
    else:
        saved['lambda'] = torch.tensor(-1.)
    fresh, _, _ = make_ippo(costs=layout())
    before = snapshot_learning_state(fresh.learning_state_dict())
    with pytest.raises(ValueError, match='[Cc]ost|[Ll]ambda'):
        fresh.load_learning_state_dict(state)
    assert_tree_equal(before, fresh.learning_state_dict())


@pytest.mark.parametrize('factory', [make_ippo, make_mappo], ids=['ippo', 'mappo'])
@pytest.mark.parametrize('corrupt', ['missing_lr', 'lr_type', 'invalid_betas'])
def test_ppo_optimizer_configuration_rejected_before_learning_tree_mutation(factory, corrupt):
    algorithm, obs, masks = factory(costs=layout())
    collect_costs(algorithm, obs, masks)
    algorithm.update()
    state = snapshot_learning_state(algorithm.learning_state_dict())
    # 最后一个owner损坏，前面的模型/Adam也不得提前写入。
    group = list(state['roles'].values())[-1]['optimizer']['param_groups'][0]
    if corrupt == 'missing_lr':
        group.pop('lr')
    elif corrupt == 'lr_type':
        group['lr'] = '0.0003'
    else:
        group['betas'] = (1.5, .999)
    fresh, _, _ = factory(costs=layout())
    before = snapshot_learning_state(fresh.learning_state_dict())
    try:
        with pytest.raises(ValueError, match='[Oo]ptimizer configuration'):
            fresh.load_learning_state_dict(state)
    finally:
        assert_tree_equal(before, fresh.learning_state_dict())


@pytest.mark.parametrize('factory', [make_ippo, make_mappo])
def test_cost_free_checkpoint_has_no_cost_fields_and_rejects_added_state(factory):
    free, _, _ = factory()
    assert not any('cost' in key for key in free.learning_state_dict())
    enabled, _, _ = factory(costs=layout())
    with pytest.raises(ValueError, match='[Cc]ost'):
        free.load_learning_state_dict(enabled.learning_state_dict())
    with pytest.raises(ValueError, match='[Cc]ost'):
        enabled.load_learning_state_dict(free.learning_state_dict())
