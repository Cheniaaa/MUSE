"""混合动作的精确编码、逐 agent 概率与真实 CPU 学习契约。"""
import math
import importlib.util

import pytest
import torch
from pydantic import ValidationError

from muse_sdk import experiment as contracts
from muse_sdk import runtime_types as actions


def make_spec(radio=True, batch=2, width=2):
    return contracts.ActionSpec(
        interface='diff_drive' if width == 2 else 'body_velocity', dtype='float32',
        shape=[batch, 3, width], fields=['motion'] * width, units=['unit'] * width,
        frames=['body_flu'] * width, minimum=[-1.] * width, maximum=[1.] * width,
        **({'radio': dict(power_minimum_w=0., power_maximum_w=2., channel_ids=[7, -3, 19])}
           if radio else {}))


def make_action(spec):
    motion = torch.linspace(-1., 1., math.prod(spec.shape)).reshape(spec.shape)
    if spec.radio is None:
        return actions.MotionAction(motion)
    return actions.MotionRadioAction(motion, actions.RadioAction(
        torch.ones(spec.shape[:2]), torch.tensor([[7, -3, 19]] * spec.shape[0], dtype=torch.int32)))


@pytest.mark.parametrize('radio', [False, True])
@pytest.mark.parametrize('width', [2, 4])
def test_codec_roundtrip_preserves_typed_leaves_and_actual_ids(radio, width):
    assert hasattr(contracts, 'RadioActionSpec'), '缺少 radio schema'
    from muse_runtime.action_codec import encode_role_action, decode_role_action
    spec = make_spec(radio, width=width)
    action = make_action(spec)
    packed = encode_role_action(spec, action, 'cpu')
    assert packed.shape == (2, 3, width + 2 * radio)
    assert packed.dtype == torch.float32
    decoded = decode_role_action(spec, packed)
    assert type(decoded) is type(action)
    torch.testing.assert_close(decoded.motion, action.motion, rtol=0, atol=0)
    if radio:
        torch.testing.assert_close(packed[..., width], action.radio.power_w)
        torch.testing.assert_close(packed[..., -1], action.radio.channel.float())
        torch.testing.assert_close(decoded.radio.channel, action.radio.channel, rtol=0, atol=0)
        torch.testing.assert_close(decoded.radio.power_w, action.radio.power_w, rtol=0, atol=0)
    else:
        assert not hasattr(decoded, 'radio')


@pytest.mark.parametrize('overrides', [
    {'channel_ids': []}, {'channel_ids': [1, 1]}, {'channel_ids': [True]},
    {'channel_ids': [1.]}, {'channel_ids': [2**24+1]}, {'channel_ids': [-2**24-1]},
    {'channel_ids': [2**31]}, {'power_minimum_w': -1.}, {'power_minimum_w': True},
    {'power_maximum_w': False}, {'power_maximum_w': float('inf')},
    {'power_minimum_w': float('nan')}, {'power_minimum_w': 2.},
    {'power_minimum_w': 3.}, {'power_maximum_w': 1e39},
])
def test_radio_schema_rejects_invalid_ids_and_bounds(overrides):
    assert hasattr(contracts, 'RadioActionSpec'), '缺少 radio schema'
    data = dict(power_minimum_w=0., power_maximum_w=2., channel_ids=[7, -3])
    with pytest.raises(ValidationError):
        contracts.RadioActionSpec(**(data | overrides))


@pytest.mark.parametrize('ids', [[-2**24, 2**24], [0]])
def test_channel_id_limits_are_exactly_roundtrippable(ids):
    assert hasattr(contracts, 'RadioActionSpec'), '缺少 radio schema'
    from muse_runtime.action_codec import encode_role_action, decode_role_action
    spec = make_spec()
    spec.radio = contracts.RadioActionSpec(power_minimum_w=0., power_maximum_w=2., channel_ids=ids)
    for channel in ids:
        action = make_action(spec)
        action.radio.channel.fill_(channel)
        torch.testing.assert_close(decode_role_action(spec, encode_role_action(spec, action, 'cpu')).radio.channel,
                                   action.radio.channel, rtol=0, atol=0)


@pytest.mark.parametrize('leaf', ['motion', 'power_w', 'channel'])
@pytest.mark.parametrize('bad', ['shape', 'dtype', 'device', 'bounds'])
def test_codec_rejects_invalid_typed_leaves(leaf, bad):
    assert hasattr(contracts, 'RadioActionSpec'), '缺少 radio schema'
    from muse_runtime.action_codec import encode_role_action
    spec = make_spec()
    action = make_action(spec)
    value = action.motion if leaf == 'motion' else getattr(action.radio, leaf)
    if bad == 'shape':
        value = value.flatten()
    elif bad == 'dtype':
        value = value.double() if leaf != 'channel' else value.long()
    elif bad == 'device':
        value = torch.empty_like(value, device='meta')
    else:
        value = value.clone().fill_(100)
    action = (action._replace(motion=value) if leaf == 'motion'
              else action._replace(radio=action.radio._replace(**{leaf: value})))
    with pytest.raises(ValueError, match='shape|dtype|device|bounds|channel'):
        encode_role_action(spec, action, 'cpu')


@pytest.mark.parametrize('column,value', [(0, float('nan')), (1, float('inf')), (2, float('-inf')),
                                        (0, 1.01), (2, 2.01), (3, 7.5), (3, 0.), (3, float('nan'))])
def test_decode_rejects_invalid_packed_without_rounding_or_clipping(column, value):
    assert hasattr(contracts, 'RadioActionSpec'), '缺少 radio schema'
    from muse_runtime.action_codec import encode_role_action, decode_role_action
    spec = make_spec()
    packed = encode_role_action(spec, make_action(spec), 'cpu')
    packed[0, 0, column] = value
    with pytest.raises(ValueError):
        decode_role_action(spec, packed)


def test_codec_rejects_packed_shape_dtype_and_wrong_variant():
    assert hasattr(contracts, 'RadioActionSpec'), '缺少 radio schema'
    from muse_runtime.action_codec import encode_role_action, decode_role_action
    spec = make_spec()
    action = make_action(spec)
    packed = encode_role_action(spec, action, 'cpu')
    for value in (packed.double(), packed.flatten(), packed[..., :-1]):
        with pytest.raises(ValueError):
            decode_role_action(spec, value)
    with pytest.raises(ValueError):
        encode_role_action(spec, actions.MotionAction(action.motion), 'cpu')
    with pytest.raises(ValueError):
        encode_role_action(make_spec(False), action, 'cpu')


def test_learning_contract_keeps_radio_order_and_ignores_batch():
    assert hasattr(contracts, 'RadioActionSpec'), '缺少 radio schema'
    # 使用真实 DTO 序列化，其他无关字段不需要解析场景资源。
    def experiment(spec):
        return contracts.ExperimentDefinition.model_construct(
            platform_revision='test', training_package=contracts.ResourceRef(
                resource_id='pkg', revision=1, sha256='a'*64),
            composition_id='default', algorithm_id='test', algorithm_config={}, training_schedule=None,
            task_id=None, task_config={}, costs=contracts.CostLayout(roles={}, world={}),
            model_bindings={}, activation={}, bindings={}, sensors={},
            action_specs={'role': spec}, action_layout=[], role_inputs={}, selections=[],
            algorithm_inputs=contracts.AlgorithmInputs(observations=[], actors={}, critics={}))
    first = make_spec()
    left = contracts.learning_contract(experiment(first))
    assert left['action_specs'][0][1]['radio'] == first.radio.model_dump(mode='json')
    assert left == contracts.learning_contract(experiment(make_spec(batch=5)))
    for ids in ([19, -3, 7], [7, -3, 20]):
        changed = make_spec()
        changed.radio.channel_ids = ids
        assert left != contracts.learning_contract(experiment(changed))


def hybrid(**overrides):
    assert importlib.util.find_spec('muse_builtins.algorithms.distributions') is not None, '缺少 hybrid distribution'
    from muse_builtins.algorithms.distributions import HybridDistribution
    return HybridDistribution(**(dict(output_dim=4, motion_dim=2, channel_ids=[7, -3, 19]) | overrides))


def test_distribution_matches_torch_oracles_per_agent():
    from torch.distributions import Beta, Categorical, kl_divergence
    dist = hybrid().double()
    assert dist.input_dim == 9
    raw = torch.linspace(-2., 2., 54, dtype=torch.float64).reshape(2, 3, 9)
    raw[..., -3:] = torch.tensor([[[0., 2., 1.], [2., 0., 1.], [1., 0., 2.]],
                                 [[1., 0., 2.], [0., 2., 1.], [2., 0., 1.]]])
    alpha = torch.nn.functional.softplus(raw[..., :3]) + 1
    beta = torch.nn.functional.softplus(raw[..., 3:6]) + 1
    continuous = Beta(alpha, beta)
    categorical = Categorical(logits=raw[..., 6:])
    dist.update(raw)
    assert len(dist.params) == 3
    torch.testing.assert_close(dist.params[0], alpha)
    torch.testing.assert_close(dist.params[1], beta)
    torch.testing.assert_close(dist.params[2], categorical.logits)
    deterministic = dist.deterministic_output(raw)
    torch.testing.assert_close(deterministic[..., :3], continuous.mean)
    torch.testing.assert_close(deterministic[..., -1], torch.tensor([[-3., 7., 19.], [19., -3., 7.]], dtype=raw.dtype))
    torch.testing.assert_close(dist.as_deterministic_output_module()(raw), deterministic)
    scripted = torch.jit.script(dist.as_deterministic_output_module())
    torch.testing.assert_close(scripted(raw), deterministic)
    ids = torch.tensor([7., -3., 19.], dtype=raw.dtype)
    expected_mean = (categorical.probs * ids).sum(-1)
    expected_variance = (categorical.probs * (ids - expected_mean.unsqueeze(-1)).square()).sum(-1)
    torch.testing.assert_close(dist.mean, torch.cat((continuous.mean, expected_mean.unsqueeze(-1)), -1))
    torch.testing.assert_close(dist.std, torch.cat((continuous.stddev, expected_variance.sqrt().unsqueeze(-1)), -1))
    entropy = continuous.entropy().sum(-1) + categorical.entropy()
    assert dist.entropy.shape == (2, 3)
    torch.testing.assert_close(dist.entropy, entropy)
    sample = dist.sample()
    assert sample.shape == (2, 3, 4)
    assert ((sample[..., :3] >= 0) & (sample[..., :3] <= 1)).all()
    matches = sample[..., -1, None] == ids
    assert matches.any(-1).all()
    index = matches.long().argmax(-1)
    expected = continuous.log_prob(sample[..., :3]).sum(-1) + categorical.log_prob(index)
    torch.testing.assert_close(dist.log_prob(sample), expected)
    # 显式标量求和防止把 B/N 轴合并；边缘连续值不静默 clipping。
    sample[0, 0, 0] = 1e-8
    sample[1, 2, 2] = 1. - 1e-8
    actual = dist.log_prob(sample)
    for b in range(2):
        for n in range(3):
            oracle = sum(Beta(alpha[b, n, d], beta[b, n, d]).log_prob(sample[b, n, d]) for d in range(3))
            oracle += categorical.log_prob(index)[b, n]
            torch.testing.assert_close(actual[b, n], oracle)
    old = tuple(p.clone() for p in dist.params)
    shifted = raw + torch.linspace(-.6, .7, 9, dtype=raw.dtype)
    dist.update(shifted)
    newer = dist.params
    expected_kl = (kl_divergence(continuous, Beta(newer[0], newer[1])).sum(-1)
                   + kl_divergence(categorical, Categorical(logits=newer[2])))
    assert dist.kl_divergence(old, newer).shape == (2, 3)
    torch.testing.assert_close(dist.kl_divergence(old, newer), expected_kl)
    torch.testing.assert_close(dist.kl_divergence(old, old), torch.zeros(2, 3, dtype=raw.dtype))


@pytest.mark.parametrize('overrides', [dict(output_dim=3), dict(motion_dim=0), dict(motion_dim=True),
                                     dict(channel_ids=[]), dict(channel_ids=[1, 1]),
                                     dict(channel_ids=[True]), dict(channel_ids=[1.]),
                                     dict(channel_ids=[2**24+1])])
def test_distribution_constructor_rejects_ambiguous_layout(overrides):
    with pytest.raises(ValueError):
        hybrid(**overrides)


@pytest.mark.parametrize('column,value', [(0, -.01), (2, 1.01), (0, float('nan')),
                                        (3, 7.1), (3, 0.), (3, float('inf'))])
def test_distribution_logprob_rejects_invalid_actions(column, value):
    dist = hybrid()
    dist.update(torch.zeros(2, 3, 9))
    sample = dist.sample()
    sample[0, 0, column] = value
    with pytest.raises(ValueError):
        dist.log_prob(sample)


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'finite', 'device'])
@pytest.mark.parametrize('method', ['update', 'deterministic_output'])
def test_distribution_rejects_invalid_model_output_in_both_modes(bad, method):
    dist = hybrid(channel_ids=[2**24-1, -2**24, 19])
    raw = torch.zeros(2, 3, 9)
    if bad == 'dtype':
        raw = raw.half()
    elif bad == 'shape':
        raw = torch.zeros(2, 3, 10)
    elif bad == 'finite':
        raw[0, 0, 0] = float('nan')
    else:
        raw = torch.empty_like(raw, device='meta')
    with pytest.raises(ValueError, match='shape|dtype|finite|device'):
        getattr(dist, method)(raw)


def test_real_cpu_ppo_storage_ratio_and_all_hybrid_heads_update():
    hybrid()
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from muse_builtins.algorithms._storage import MaskedStorage, masked_update
    from muse_builtins.algorithms.ippo import compute_returns
    from muse_runtime.action_codec import encode_role_action, decode_role_action

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(45)
        obs = TensorDict({'local': torch.randn(6, 5)}, batch_size=[6])
        groups = {'actor': ['local'], 'critic': ['local']}
        actor = MLPModel(obs, groups, 'actor', 4, hidden_dims=[16], distribution_cfg={
            'class_name': 'muse_builtins.algorithms.distributions:HybridDistribution',
            'motion_dim': 2, 'channel_ids': [7, -3, 19]})
        critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[16])
        storage = MaskedStorage('rl', 6, 4, obs, [4], 'cpu')
        ppo = PPO(actor, critic, storage, device='cpu', num_learning_epochs=2,
                  num_mini_batches=2, learning_rate=.003, schedule='adaptive', desired_kl=.02)
        before = {key: value.clone() for key, value in actor.state_dict().items()}
        # 观测真实 PPO backward 的各 head 梯度，不替换 optimizer 或 update。
        final_layer = [layer for layer in actor.mlp if isinstance(layer, torch.nn.Linear)][-1]
        gradients = []
        hook = final_layer.weight.register_hook(lambda grad: gradients.append(grad.detach().clone()))
        with torch.no_grad():
            for step in range(4):
                raw = ppo.act(obs)
                assert raw.shape == (6, 4)
                # 调用者作物理仿射，codec 解码的 ID 必须与存储 ID 完全一致。
                physical = raw.reshape(2, 3, 4).clone()
                physical[..., :2] = -1 + 2 * physical[..., :2]
                physical[..., 2] *= 2
                spec = make_spec()
                typed = decode_role_action(spec, physical)
                torch.testing.assert_close(encode_role_action(spec, typed, 'cpu'), physical, rtol=0, atol=0)
                reward = raw[:, 0] - .5 * raw[:, 1] + raw[:, 2] + (raw[:, 3] == -3).float() + step * .1
                storage.final_values[step].copy_(critic(obs))
                storage.actor_valid[step].fill_(True)
                storage.critic_valid[step].fill_(True)
                storage.terminated[step].fill_(step == 3)
                ppo.process_env_step(obs, reward, torch.full((6,), step == 3), extras={})
            assert len(storage.distribution_params) == 3
            for step in range(4):
                actor(storage.observations[step], stochastic_output=True)
                logprob = actor.get_output_log_prob(storage.actions[step])
                ratio = (logprob - storage.actions_log_prob[step].squeeze(-1)).exp()
                torch.testing.assert_close(ratio, torch.ones(6), rtol=0, atol=0)
            compute_returns(storage, ppo.gamma, ppo.lam)
        metrics = masked_update(ppo)
        hook.remove()
        assert metrics['actor_samples'] == metrics['critic_samples'] == 24
        assert storage.step == 0
        assert all(math.isfinite(v) for group in ('actor', 'critic') for v in metrics[group].values())
        weight_name = next(name for name, value in actor.named_parameters() if value is final_layer.weight)
        delta = final_layer.weight.detach() - before[weight_name]
        # alpha motion/power、beta motion/power、categorical 各自有非零梯度和真实更新。
        for rows in ([0, 1], [2], [3, 4], [5], [6, 7, 8]):
            assert any(g[rows].abs().sum() > 0 for g in gradients)
            assert delta[rows].abs().sum() > 0
        assert all(torch.isfinite(p).all() for p in actor.parameters())
