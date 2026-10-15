"""真实builtin混合learner：物理动作、单次RNN推进和不可混淆的学习快照。"""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from muse_builtins.algorithms.ippo import IPPO, IPPOConfig
from muse_builtins.algorithms.mappo import MAPPO, MAPPOConfig
from muse_runtime.checkpoint import snapshot_learning_state
from muse_sdk.experiment import ActionSpec, CostLayout
from muse_sdk.runtime_types import MotionAction, MotionRadioAction, Observations, RoleMask
from test_mappo import assert_tree_equal, transition


CASES = [('ippo', True, None)] + [('mappo', sharing, recurrent)
    for sharing in (True, False) for recurrent in (None, 'gru', 'lstm')]


def output_head(actor):
    return [module for module in actor.mlp if isinstance(module, torch.nn.Linear)][-1]


def make_hybrid(kind='ippo', sharing=True, recurrent=None, batch=2, channels=(7, -3, 19), change=None):
    specs = {role: ActionSpec(interface='body_velocity' if width == 4 else 'diff_drive', dtype='float32',
        shape=[batch, n, width], fields=['motion']*width, units=['unit']*width, frames=['body_flu']*width,
        minimum=[-2.]*width, maximum=[3.]*width,
        radio=dict(power_minimum_w=.0001, power_maximum_w=.001, channel_ids=list(channels))
        if role == 'scout' else None) for role, n, width in [('scout', 2, 4), ('guard', 1, 2)]}
    actors = {role: [NS(path=['observations', 'actors', role, name],
                        tensor=NS(shape=[batch, spec.shape[1], 3])) for name in ('target', 'feedback')]
              for role, spec in specs.items()}
    if change == 'channel_order':
        specs['scout'].radio.channel_ids.reverse()
    elif change == 'channel_value':
        specs['scout'].radio.channel_ids[-1] = 23
    elif change == 'power':
        specs['scout'].radio.power_maximum_w = .002
    elif change == 'motion':
        specs['scout'].maximum[0] = 4.
    elif change == 'slot':
        specs['scout'].shape[1] = 3
        for item in actors['scout']:
            item.tensor.shape[1] = 3
    elif change == 'feature_shape':
        actors['scout'][1].tensor.shape = [batch, 2, 1, 3]
    elif change == 'feature_order':
        actors['scout'].reverse()
    config_type = IPPOConfig if kind == 'ippo' else MAPPOConfig
    config = config_type(collection_steps=4, hidden_dims=[12], num_learning_epochs=2,
                         num_mini_batches=2, entropy_coef=0.).model_dump()
    if kind == 'mappo':
        config.update(share_actor_parameters=sharing,
                      recurrent=None if recurrent is None else dict(type=recurrent, hidden_size=7))
    experiment = NS(root_seed=17, action_specs=specs,
        algorithm_inputs=NS(actors=actors, critics={'team': [NS(
            path=['observations', 'critic_inputs', 'team', 'target'], tensor=NS(shape=[batch, 3, 2]))]}),
        training_schedule=NS(collection_steps=4), costs=CostLayout(roles={}, world={}))
    learner = (IPPO if kind == 'ippo' else MAPPO)(config)
    learner.initialize(experiment, NS(device='cpu'), None)
    obs = Observations({role: {item.path[-1]: torch.arange(torch.tensor(item.tensor.shape).prod()).float().reshape(
        item.tensor.shape)/10 for item in fields} for role, fields in actors.items()},
        {'team': {'target': torch.arange(batch*6).float().reshape(batch, 3, 2)/7}})
    masks = {role: RoleMask(*(torch.ones(spec.shape[:2], dtype=torch.bool) for _ in range(3)))
             for role, spec in specs.items()}
    return learner, obs, masks


@pytest.mark.parametrize('kind,sharing,recurrent', CASES)
def test_builtin_hybrid_raw_typed_ratio_and_task_loss_update(kind, sharing, recurrent):
    learner, obs, masks = make_hybrid(kind, sharing, recurrent)
    before = snapshot_learning_state(learner.learning_state_dict())
    for step in range(learner.collection_steps):
        result = learner.act(obs, masks, None, False)
        assert isinstance(result.role_actions['scout'], MotionRadioAction)
        assert isinstance(result.role_actions['guard'], MotionAction)
        assert result.role_actions['scout'].radio.channel.dtype == torch.int32
        for owner, child in learner.children.items():
            role, slot = (owner, None) if kind == 'ippo' else learner.owners[owner]
            pending = child.storage.pending
            raw = pending.actions
            width = 4 if role == 'scout' else 2
            assert raw.shape[-1] == width + (2 if role == 'scout' else 0)
            assert len(pending.distribution_params) == (3 if role == 'scout' else 2)
            params = pending.distribution_params
            logp = torch.distributions.Beta(*params[:2]).log_prob(raw[..., :width+(role == 'scout')]).sum(-1)
            if role == 'scout':
                ids = torch.tensor([7, -3, 19])
                index = (raw[..., -1, None] == ids).long().argmax(-1)
                logp += torch.distributions.Categorical(logits=params[2]).log_prob(index)
            torch.testing.assert_close((logp - pending.actions_log_prob).exp(), torch.ones(raw.shape[0]))
            action = result.role_actions[role]
            selected = slice(None) if slot is None else slice(slot, slot+1)
            torch.testing.assert_close(action.motion[:, selected].flatten(0, 1), -2 + 5*raw[..., :width])
            if role == 'scout':
                torch.testing.assert_close(action.radio.power_w[:, selected].flatten(), .0001 + .0009*raw[..., 4])
                torch.testing.assert_close(action.radio.channel[:, selected].flatten(), raw[..., 5].int())
        tr = transition(obs, masks, step)
        rewards = {role: result.role_actions[role].motion[..., 0].square() + step
                   for role in masks}
        learner.process_env_step(tr._replace(rewards=rewards), result.act_aux)
    learner.update()
    for owner, child in learner.children.items():
        role = owner if kind == 'ippo' else learner.owners[owner][0]
        head = output_head(child.actor)
        name = next(name for name, parameter in child.actor.named_parameters() if parameter is head.weight)
        delta = head.weight - before['roles'][owner]['actor'][name]
        assert delta.abs().sum() > 0
        if role == 'scout':
            assert delta[:4].abs().sum() + delta[5:9].abs().sum() > 0
            assert delta[4].abs().sum() + delta[9].abs().sum() > 0
            assert delta[10:].abs().sum() > 0
        assert child.optimizer.state
    saved = snapshot_learning_state(learner.learning_state_dict())
    fresh, fresh_obs, fresh_masks = make_hybrid(kind, sharing, recurrent, batch=3)
    fresh.load_learning_state_dict(saved)
    assert_tree_equal(saved, fresh.learning_state_dict())
    fresh.set_mode('EVALUATE')
    before_eval = snapshot_learning_state(fresh.learning_state_dict())
    for _ in range(2):
        action = fresh.act(fresh_obs, fresh_masks, None, False)
        assert torch.isin(action.role_actions['scout'].radio.channel, torch.tensor([7, -3, 19])).all()
        fresh.process_env_step(transition(fresh_obs, fresh_masks), action.act_aux)
    assert_tree_equal(before_eval, fresh.learning_state_dict())


@pytest.mark.parametrize('kind,sharing,recurrent', CASES)
def test_deterministic_channel_mode_single_recurrent_step(kind, sharing, recurrent):
    learner, obs, masks = make_hybrid(kind, sharing, recurrent)
    expected = {}
    with torch.no_grad():
        for owner, child in learner.children.items():
            role = owner if kind == 'ippo' else learner.owners[owner][0]
            if role == 'scout':
                # 明确非均值频道：probability约[.09,.665,.245]，众数为-3。
                output_head(child.actor).weight[-3:].zero_()
                output_head(child.actor).bias[-3:].copy_(torch.tensor([0., 2., 1.]))
            oracle = deepcopy(child.actor)
            output = oracle(learner._observations(obs, owner), stochastic_output=False)
            expected[owner] = (output, oracle.get_hidden_state())
    result = learner.act(obs, masks, None, True)
    assert isinstance(result.role_actions['scout'], MotionRadioAction)
    assert (result.role_actions['scout'].radio.channel == -3).all()
    for owner, child in learner.children.items():
        raw, hidden = expected[owner]
        torch.testing.assert_close(child.storage.pending.actions, raw)
        assert_tree_equal(child.actor.get_hidden_state(), hidden)
        torch.testing.assert_close(child.storage.pending.actions_log_prob, child.actor.get_output_log_prob(raw))


@pytest.mark.parametrize('kind', ['ippo', 'mappo'])
@pytest.mark.parametrize('change', ['channel_order', 'channel_value', 'power', 'motion', 'slot',
                                   'feature_shape', 'feature_order', 'legacy'])
def test_hybrid_state_semantics_checked_before_write(kind, change):
    source, _, _ = make_hybrid(kind)
    state = snapshot_learning_state(source.learning_state_dict())
    if change == 'legacy':
        state['revision'] = 1
        state.pop('signature', None)
    target, _, _ = make_hybrid(kind, change=change)
    before = snapshot_learning_state(target.learning_state_dict())
    with pytest.raises(ValueError, match='signature|revision|layout'):
        target.load_learning_state_dict(state)
    assert_tree_equal(before, target.learning_state_dict())


@pytest.mark.parametrize('kind', ['ippo', 'mappo'])
def test_single_channel_is_legal_and_metadata_declares_radio(kind):
    learner, obs, masks = make_hybrid(kind, channels=(19,))
    assert 'power_channel' in learner.describe()['action_requirements']
    action = learner.act(obs, masks, None, True).role_actions['scout']
    assert isinstance(action, MotionRadioAction)
    assert (action.radio.channel == 19).all()


@pytest.mark.parametrize('sharing', [True, False])
@pytest.mark.parametrize('recurrent', [None, 'gru', 'lstm'])
def test_hybrid_actor_input_isolation(sharing, recurrent):
    source, obs, masks = make_hybrid('mappo', sharing, recurrent)
    baseline = source.act(obs, masks, None, True)
    for scope in ('critic', 'other_role'):
        learner, changed, _ = make_hybrid('mappo', sharing, recurrent)
        if scope == 'critic':
            changed.critic_inputs['team']['target'].add_(100)
        else:
            changed.actors['guard']['target'].add_(100)
        result = learner.act(changed, masks, None, True)
        assert_tree_equal(baseline.role_actions['scout'], result.role_actions['scout'])
