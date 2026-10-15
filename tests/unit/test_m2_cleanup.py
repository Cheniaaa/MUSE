"""M2收口：配置有效性、评估模式以及借用state与稳定snapshot的边界。"""
import pytest
import torch

from muse_runtime.checkpoint import snapshot_learning_state


@pytest.mark.parametrize('values,expected', [({}, None), ({'recurrent': None}, None),
    ({'recurrent': {'type': 'gru'}}, {'type': 'gru', 'hidden_size': 64}),
    ({'recurrent': {'type': 'lstm', 'hidden_size': 7}}, {'type': 'lstm', 'hidden_size': 7})])
def test_recurrent_configuration_has_only_effective_parameters(values, expected):
    from muse_builtins.algorithms.mappo import MAPPOConfig
    from muse_sdk.config import export_config_schema, resolve_config
    config = resolve_config(export_config_schema(MAPPOConfig), values, {}, require_complete=True)
    assert config['recurrent'] == expected
    assert 'recurrent_hidden_size' not in config
    assert MAPPOConfig.model_validate(config).model_dump() == config


@pytest.mark.parametrize('values', [
    {'recurrent': None, 'recurrent_hidden_size': 128},
    {'recurrent': 'gru'}, {'recurrent': {}},
    {'recurrent': {'type': 'gru', 'hidden_size': 0}},
    {'recurrent': {'type': 'gru', 'hidden_size': True}},
])
def test_invalid_or_unused_recurrent_parameters_rejected(values):
    from muse_builtins.algorithms.mappo import MAPPOConfig
    from muse_sdk.config import ConfigError, export_config_schema, resolve_config
    with pytest.raises(ConfigError):
        resolve_config(export_config_schema(MAPPOConfig), values, {}, require_complete=True)


@pytest.mark.parametrize('kind', ['ippo', 'mappo'])
def test_runtime_owns_stable_snapshot_of_borrowed_model_and_adam(kind):
    if kind == 'ippo':
        from test_returns import make_algorithm, collect
    else:
        from test_mappo import make_algorithm, collect
    learner, obs, masks = make_algorithm()
    collect(learner, obs, masks)
    learner.update()
    borrowed = learner.learning_state_dict()
    stable = snapshot_learning_state(borrowed)
    owner, child = next(iter(learner.children.items()))
    name, parameter = next(iter(child.actor.named_parameters()))
    expected = stable['roles'][owner]['actor'][name].clone()
    with torch.no_grad():
        parameter.add_(1)
    # Hook采用普通state_dict借用语义；Runtime副本可安全跨越后续更新。
    torch.testing.assert_close(borrowed['roles'][owner]['actor'][name], parameter)
    torch.testing.assert_close(stable['roles'][owner]['actor'][name], expected)
    index = next(iter(borrowed['roles'][owner]['optimizer']['state']))
    saved_adam = stable['roles'][owner]['optimizer']['state'][index]['exp_avg'].clone()
    next(iter(child.optimizer.state.values()))['exp_avg'].add_(2)
    torch.testing.assert_close(borrowed['roles'][owner]['optimizer']['state'][index]['exp_avg'], saved_adam + 2)
    torch.testing.assert_close(stable['roles'][owner]['optimizer']['state'][index]['exp_avg'], saved_adam)


def test_ippo_evaluate_is_deterministic_even_when_caller_requests_sampling():
    from test_returns import make_algorithm, transition
    from test_mappo import assert_tree_equal
    learner, obs, masks = make_algorithm()
    learner.set_mode('EVALUATE')
    before = snapshot_learning_state(learner.learning_state_dict())
    first = learner.act(obs, masks, None, False)
    learner.process_env_step(transition(obs, masks), first.act_aux)
    second = learner.act(obs, masks, None, False)
    assert_tree_equal(first.role_actions, second.role_actions)
    assert_tree_equal(before, learner.learning_state_dict())
