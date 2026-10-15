"""局部测量与逐目标位置界的独立行为断言。"""
from types import SimpleNamespace

import pytest
import torch

from muse_sdk.geometry import PropagationPathLayout
from muse_sdk.network import SensingFieldView


def term_env():
    # S顺序与role slot均反向；target point不是endpoint索引。
    points = [dict(group_id='uavs', slot=i, device_id='radio', body='root',
                   position_m=(0., 0., 0.), assumption='test') for i in range(2)]
    points += [dict(group_id='targets', slot=i, device_id=None, body='root',
                    position_m=(0., 0., 0.), assumption='test') for i in range(2)]
    layout = PropagationPathLayout(points=tuple(points),
        paths=tuple(dict(path_id=f'p{i}', tx=tx, rx=rx) for i, (tx, rx) in enumerate(
            [(1, 2), (2, 1), (0, 2), (2, 0), (1, 3), (3, 1), (0, 3), (3, 0)])),
        desired=(), sensing=tuple(dict(sensing_id=name, outbound=2*i, inbound=2*i+1, direct=None)
            for i, name in enumerate(['local', 'local', 'other', 'other'])), interference=())
    experiment = SimpleNamespace(geometry=SimpleNamespace(layout=layout),
        bindings={'observer': SimpleNamespace(group_id='uavs', role_slot_to_group_slot=[0, 1])},
        scenario=SimpleNamespace(scene=SimpleNamespace(control_substeps=20)))
    data = torch.tensor([[[20., 0.], [0., 12.], [0., 0.], [0., 0.]]], dtype=torch.float64)
    measurement = SensingFieldView(data, data != 0, torch.tensor([14]), torch.tensor([20]))
    return SimpleNamespace(experiment=experiment, sensing_outputs={'measurement': measurement}, device='cpu')


@pytest.mark.parametrize('slots,order', [([0, 1], [0, 1]), ([1, 0], [1, 0])])
def test_local_measurement_slot_resource_window_and_truth_isolation(slots, order):
    from examples.sensing_reference.training_package.terms import LocalPathLengthObservation
    env = term_env()
    env.experiment.bindings['observer'].role_slot_to_group_slot = slots
    cfg = SimpleNamespace(params=dict(role_id='observer', sensing_id='local', range_scale_m=100.))
    term = LocalPathLengthObservation(cfg, env)
    expected = torch.tensor([[[.12, 1., .3], [.20, 1., .3]]])[:, order]
    torch.testing.assert_close(term(env, **cfg.params), expected)
    env.role_positions_world = {'observer': torch.full((1, 2, 3), 999.)}
    env.sensing_outputs['fim'] = 'must not read'
    torch.testing.assert_close(term(env, **cfg.params), expected)
    env.sensing_outputs['measurement'].valid.zero_()
    torch.testing.assert_close(term(env, **cfg.params), torch.tensor([[[0., 0., .3], [0., 0., .3]]]))
    env.sensing_outputs['measurement'].interval_start_tick.zero_()
    env.sensing_outputs['measurement'].interval_end_tick.zero_()
    assert torch.equal(term(env, **cfg.params), torch.zeros(1, 2, 3))


@pytest.mark.parametrize('sensing_id', ['missing', 'other'])
def test_local_term_rejects_missing_or_ambiguous_slot_relation(sensing_id):
    from examples.sensing_reference.training_package.terms import LocalPathLengthObservation
    env = term_env()
    if sensing_id == 'other':
        layout = env.experiment.geometry.layout
        env.experiment.geometry.layout = layout.model_copy(update={'sensing': tuple(
            item.model_copy(update={'sensing_id': 'other'}) for item in layout.sensing)})
    cfg = SimpleNamespace(params=dict(role_id='observer', sensing_id=sensing_id, range_scale_m=100.))
    with pytest.raises(ValueError, match='exactly one monostatic'):
        LocalPathLengthObservation(cfg, env)


def test_position_bound_per_target_valid_rank_and_no_cross_target_information():
    from examples.sensing_reference.training_package.terms import PositionBoundViolation
    env = term_env()
    matrices = torch.zeros(4, 4, 2, 2, 2, dtype=torch.float64)
    # 每个target的两条关系各贡献一条正交rank1；不能跨target补rank。
    for target in (0, 1):
        matrices[:, 2*target, 0, 0, 0] = 100.
        matrices[:, 2*target+1, 1, 1, 1] = 100.
    matrices[1, 2:, :, :, :] *= .01
    matrices[2, 3, 1, 1, 1] = 0.
    valid = torch.ones(4, 4, 2, dtype=torch.bool)
    valid[3, 0, 1] = False
    env.sensing_outputs = {'fim': SensingFieldView(matrices, valid, torch.zeros(4), torch.ones(4))}
    term = PositionBoundViolation(SimpleNamespace(params={}), env)
    result = term(env, position_bound_limit_m=.25)
    assert result.dtype == torch.float32
    assert result.tolist() == [0., 1., 1., 1.]


@pytest.mark.parametrize('scale', [1e-100, 1., 1e100])
def test_position_bound_relative_rank_threshold(scale):
    from examples.sensing_reference.training_package.terms import PositionBoundViolation
    env = term_env()
    data = torch.zeros(2, 4, 1, 2, 2, dtype=torch.float64)
    eps = torch.finfo(torch.float64).eps
    for target in (0, 1):
        data[:, 2*target, 0, 0, 0] = scale
        data[0, 2*target+1, 0, 1, 1] = scale*eps
        data[1, 2*target+1, 0, 1, 1] = scale*4*eps
    env.sensing_outputs = {'fim': SensingFieldView(data, torch.ones(2, 4, 1, dtype=torch.bool), None, None)}
    term = PositionBoundViolation(SimpleNamespace(params={}), env)
    assert term(env, position_bound_limit_m=1e100).tolist() == [1., 0.]


def test_rule_role_controls_local_input_receipt_and_reset(capsys):
    import json
    from examples.sensing_reference.training_package.algorithm import Config, SensingReferenceRule
    from muse_sdk.network import WirelessSnapshot
    from muse_sdk.runtime_types import Observations
    config = Config(phases=[dict(duration_decisions=1, roles={
        'observer': dict(motion=[0., .1, 0., 0.], power_w=.001, channel_id=7),
        'third': dict(motion=[0., 0., 0., 0.], power_w=.0001, channel_id=-3)}),
        dict(duration_decisions=2, roles={
            'observer': dict(motion=[0., .1, 0., 0.], power_w=0., channel_id=-3),
            'third': dict(motion=[0., 0., 0., 0.], power_w=.001, channel_id=19)})]).model_dump()
    experiment = SimpleNamespace(topology=SimpleNamespace(world_count=2), action_specs={
        'observer': SimpleNamespace(shape=(2, 2, 4), radio=True),
        'third': SimpleNamespace(shape=(2, 1, 4), radio=True)})
    rule = SensingReferenceRule(config)
    rule.initialize(experiment, SimpleNamespace(device='cpu'), None)
    rule.set_mode('EVALUATE')
    with pytest.raises(ValueError, match='EVALUATE only'):
        rule.set_mode('TRAIN')
    observation = Observations({'observer': {'measurement': torch.full((2, 2, 3), .125)}, 'third': {}}, {})
    actions = rule.act(observation, {}, None, True).role_actions
    assert actions['observer'].radio.power_w.shape == (2, 2)
    assert actions['third'].radio.channel.tolist() == [[-3], [-3]]
    field = SensingFieldView(torch.ones(2, 1, 1), torch.ones(2, 1, 1, dtype=torch.bool),
                             torch.tensor([14, 14]), torch.tensor([20, 20]))
    # 无radio是正式可选块；终态测量来自pre-reset快照。
    transition = SimpleNamespace(world_reset=torch.tensor([True, False]), world_reward=None,
        costs={'world': {'position_bound_violation': torch.tensor([1., 0.])}},
        wireless=WirelessSnapshot(None, None, None, {}, {'measurement': field}))
    rule.process_env_step(transition, ())
    receipt = json.loads(capsys.readouterr().out.removeprefix('SENSING_RESULT '))
    assert receipt['policy_observations']['observer']['measurement'][0][0] == [.125]*3
    assert receipt['sensing']['measurement']['interval_start_tick'] == [14, 14]
    assert receipt['world_cost']['position_bound_violation'] == [1., 0.]
    assert receipt['radio'] == {} and receipt['sample_tick'] is None
    assert receipt['world_reward'] is None
    actions = rule.act(observation, {}, None, True).role_actions
    assert actions['third'].radio.channel.tolist() == [[-3], [19]]
    rule.reset_worlds(torch.tensor([False, True]))
    assert rule.step.tolist() == [0, 0]
    invalid = SensingReferenceRule({'phases': [dict(duration_decisions=1, roles={})]})
    with pytest.raises(ValueError, match='roles must match'):
        invalid.initialize(experiment, SimpleNamespace(device='cpu'), None)


def test_radio_reward_masks_actual_rate():
    from examples.sensing_reference.training_package.terms import radio_rate
    env = SimpleNamespace(network=SimpleNamespace(radio=SimpleNamespace(rate=SimpleNamespace(
        rate_bps=torch.tensor([[[3., 20.], [7., 0.]]]), valid=torch.tensor([[[True, False], [True, True]]])))))
    assert radio_rate(env).tolist() == [10.]


def test_native_managers_construct_call_and_reset_stateless_terms():
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from muse_runtime.managers import (CheckedObservationManager, CheckedObservationTermCfg,
                                       CostTermCfg, WorldCostManager)
    from examples.sensing_reference.training_package.terms import LocalPathLengthObservation, PositionBoundViolation
    env = term_env()
    env.num_envs = 1
    observation = CheckedObservationManager({'local': ObservationGroupCfg(terms={
        'measurement': CheckedObservationTermCfg(func=LocalPathLengthObservation,
            params=dict(role_id='observer', sensing_id='local', range_scale_m=100.),
            output_shape=(1, 2, 3), output_dtype=torch.float32)}, concatenate_terms=False)}, env)
    torch.testing.assert_close(observation.compute()['local']['measurement'],
                               torch.tensor([[[.12, 1., .3], [.20, 1., .3]]]))
    observation.reset(torch.tensor([0]))
    data = torch.zeros(1, 4, 2, 2, 2, dtype=torch.float64)
    data[:, :, :, 0, 0] = 100.
    data[:, :, :, 1, 1] = 100.
    env.sensing_outputs['fim'] = SensingFieldView(data, torch.ones(1, 4, 2, dtype=torch.bool), None, None)
    cost = WorldCostManager({'bound': CostTermCfg(func=PositionBoundViolation,
        params={'position_bound_limit_m': .25}, cost_id='position_bound_violation')}, env)
    assert cost.compute()['position_bound_violation'].tolist() == [0.]
    assert cost.reset(torch.tensor([0])) == {}
