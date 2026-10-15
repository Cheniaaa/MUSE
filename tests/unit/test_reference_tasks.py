"""小整数ledger独立核对任务评分，不运行物理或GPU。"""
from importlib import import_module
from types import SimpleNamespace as NS

import pytest
import torch

from muse_sdk.network import SensingFieldView, ServiceView


@pytest.mark.parametrize('scenario,name', [('isac_cost', 'IsacTask'), ('hybrid_history_markov', 'HistoryTask')])
def test_task_has_a_real_registration(scenario, name):
    module = import_module(f'examples.{scenario}.training_package')
    definition = module.build_training_package()
    task_id = 'isac_task' if scenario == 'isac_cost' else 'history_task'
    assert definition.tasks[task_id].__name__ == name


@pytest.fixture(params=['isac_cost', 'hybrid_history_markov'])
def ledger(request):
    scenario = request.param
    module = import_module(f'examples.{scenario}.training_package.task')
    roles = ('alpha', 'beta') if scenario == 'isac_cost' else ('pilot',)
    services = ('alpha_upload', 'beta_upload') if len(roles) == 2 else ('uplink',)
    controls = tuple(NS(role_id=role, endpoint_indices=(index,)) for index, role in enumerate(roles))
    layout = NS(sensing=(NS(outbound=0), NS(outbound=1)), paths=(NS(rx=2), NS(rx=2)),
                desired=tuple(NS(link_id=link) for link in (('alpha_up', 'beta_up') if len(roles) == 2 else ('up',))))
    network = NS(role_controls=controls, endpoint_limits=tuple(NS(max_power_w=.001) for _ in roles),
                 services=tuple(NS(service_id=service, link_indices=(i,), resource_indices=(0, 1),
                                   initial_bits=4e9) for i, service in enumerate(services)),
                 communication_allocations=tuple(NS(link_index=i, tx=i) for i in range(len(roles))))
    # 额外endpoint为jammer；它的能量不能进入学习端点ledger。
    power = torch.zeros(2, len(roles) + 1, 3)
    power[:, :len(roles), 1] = .001
    power[:, -1, :] = .1
    views = {service: ServiceView(torch.full((2, 1), 100., dtype=torch.float64),
                                  torch.tensor([[2.], [3.]], dtype=torch.float64),
                                  torch.tensor([[True], [False]]), torch.zeros(2), torch.ones(2))
             for service in services}
    env = NS(num_envs=2, device='cpu', experiment=NS(
        bindings={role: NS(role_slot_count=1) for role in roles}, network=network,
        geometry=NS(layout=layout), algorithm_config={'gamma': .5}, costs=NS(world={}),
        scenario=NS(scene=NS(physics_dt=.002, control_substeps=20))),
        role_positions_world={role: torch.tensor([[[0., 0., 1.]], [[0., 0., 1.]]]) for role in roles},
        physics_outputs=NS(episode_decision=torch.tensor([1, 2]),
                           crashed={role: torch.zeros(2, 1, dtype=torch.bool) for role in roles}),
        network=NS(services=views, simultaneous_transmissions=NS(power_w=power)),
        sensing_outputs={'fim': SensingFieldView(
            torch.eye(2, dtype=torch.float64).expand(2, 2, 3, 2, 2).clone() * 1e5,
            torch.ones(2, 2, 3, dtype=torch.bool), torch.zeros(2), torch.ones(2))})
    config = dict(horizon_decisions=3, workspace_by_role={role: dict(
        minimum_m=[-2, -2, .5], maximum_m=[2.5, 3, 1.5]) for role in roles},
        service_bits_scale=5, rf_energy_scale_j=.00004 * len(roles), rf_penalty_weight=1)
    if scenario == 'isac_cost':
        config['position_bound_limit_m'] = .01
    cls = getattr(module, 'IsacTask' if scenario == 'isac_cost' else 'HistoryTask')
    return scenario, cls, config, env


def test_decision_ledger_is_read_only_and_terms_do_not_rescore(ledger):
    scenario, cls, config, env = ledger
    task = env.task = cls(config, env)
    assert not task.done.any() and torch.equal(task.remaining_fraction, torch.ones(2, 1))
    before = {key: view.served_bits.clone() for key, view in env.network.services.items()}
    result = task.update(env)
    count = len(config['workspace_by_role'])
    torch.testing.assert_close(task.reward, torch.tensor([2 * count / 5 - 1, 3 * count / 5 - 1]))
    torch.testing.assert_close(result.diagnostics['rf_energy_j'], torch.full((2,), 4e-5 * count))
    assert result.diagnostics['service_valid'].tolist() == [True, False]
    assert result.role_status == result.learning_done == {}
    terms = import_module(f'examples.{scenario}.training_package.terms')
    for _ in range(3):
        assert terms.mission_reward(env) is task.reward
        assert terms.mission_done(env) is task.done
        for role in config['workspace_by_role']:
            torch.testing.assert_close(terms.team_reward(env, role_id=role), task.reward[:, None])
            torch.testing.assert_close(terms.remaining_time(env, role_id=role), task.remaining_fraction[:, None])
    for key, view in env.network.services.items():
        assert torch.equal(view.served_bits, before[key])
    if scenario == 'isac_cost':
        assert terms.mission_cost(env) is task.mission_cost
        assert task.mission_cost.tolist() == [0, 0]
    else:
        assert not hasattr(task, 'mission_cost')


@pytest.mark.parametrize('gamma,step,tail', [(.5, 1, 1.75), (.5, 2, 1.5), (.5, 3, 1),
                                          (1., 1, 3), (1., 2, 2), (1., 3, 1)])
def test_failure_tail_and_asynchronous_reset(ledger, gamma, step, tail):
    scenario, cls, config, env = ledger
    env.experiment.algorithm_config['gamma'] = gamma
    env.physics_outputs.episode_decision[:] = step
    role = next(iter(config['workspace_by_role']))
    env.physics_outputs.crashed[role][0] = True
    env.role_positions_world[role][0, :, 0] = 10
    task = env.task = cls(config, env)
    result = task.update(env)
    assert task.reward[0].item() == pytest.approx(-tail)
    assert task.done.tolist() == [True, step == 3]
    assert result.diagnostics['failed'].tolist() == [True, False]
    assert result.diagnostics['served_bits'][0] > 0
    if scenario == 'isac_cost':
        assert task.mission_cost.tolist() == [tail, 0]
    other = task.reward[1].clone()
    task.reset(torch.tensor([0]))
    assert task.remaining_fraction[0].item() == 1
    assert task.remaining_fraction[1].item() == pytest.approx((3 - step) / 3)
    assert task.reward[0] == 0 and task.reward[1] == other and not task.done[0]
    if scenario == 'isac_cost':
        assert task.mission_cost[0] == 0


@pytest.mark.parametrize('bad', ['keys', 'bounds'])
def test_task_constructor_owns_cross_field_validation(ledger, bad):
    scenario, cls, config, env = ledger
    if bad == 'keys':
        config['workspace_by_role']['unknown'] = config['workspace_by_role'].pop(next(iter(config['workspace_by_role'])))
    elif bad == 'bounds':
        config['workspace_by_role'][next(iter(config['workspace_by_role']))]['maximum_m'][0] = -3
    with pytest.raises(ValueError, match='workspace|discount'):
        cls(config, env)


def test_isac_cost_discount_is_checked_once_at_construction(ledger):
    scenario, cls, config, env = ledger
    if scenario == 'isac_cost':
        env.experiment.costs.world['mission_position_violation'] = NS(discount=.99)
        with pytest.raises(ValueError, match='discount'):
            cls(config, env)
        env.experiment.costs.world['mission_position_violation'].discount = .5
        assert cls(config, env).mission_cost.shape == (2,)
    else:
        assert 'mission_cost' not in cls.describe()['published_fields']


@pytest.mark.parametrize('bad', ['missing', 'wrong_link'])
def test_task_rejects_wrong_service_relationship(ledger, bad):
    _, cls, config, env = ledger
    if bad == 'missing':
        env.experiment.network.services = ()
    else:
        env.experiment.network.services[0].link_indices = (42,)
    with pytest.raises(ValueError, match='service'):
        cls(config, env)


def test_workspace_failure_without_crash_and_actual_power_failure_bound(ledger):
    _, cls, config, env = ledger
    role = next(iter(config['workspace_by_role']))
    env.role_positions_world[role][0, :, 1] = 4
    for limit in env.experiment.network.endpoint_limits:
        limit.max_power_w = .002
    config['rf_penalty_weight'] = 3
    task = cls(config, env)
    result = task.update(env)
    assert result.diagnostics['failed'].tolist() == [True, False]
    assert task.reward[0] == pytest.approx(-6 * 1.75)


def test_multislot_service_and_endpoint_ledger_has_one_world_failure_tail(ledger):
    scenario, cls, config, env = ledger
    network = env.experiment.network
    roles = tuple(config['workspace_by_role'])
    count = len(roles)
    for index, role in enumerate(roles):
        env.experiment.bindings[role].role_slot_count = 2
        env.role_positions_world[role] = env.role_positions_world[role].expand(-1, 2, -1).clone()
        env.physics_outputs.crashed[role] = torch.zeros(2, 2, dtype=torch.bool)
        network.role_controls[index].endpoint_indices = (index, index + count)
        network.services[index].link_indices = (index, index + count)
        view = env.network.services[network.services[index].service_id]
        env.network.services[network.services[index].service_id] = view._replace(
            served_bits=view.served_bits.expand(-1, 2), valid=view.valid.expand(-1, 2))
    network.endpoint_limits += tuple(NS(max_power_w=.001) for _ in roles)
    network.communication_allocations += tuple(NS(link_index=index + count, tx=index + count) for index in range(count))
    env.experiment.geometry.layout.desired *= 2
    # resource轴重新排序并增宽，不改变总发射功率。
    env.network.simultaneous_transmissions.power_w = torch.zeros(2, 2 * count + 1, 4)
    env.network.simultaneous_transmissions.power_w[:, :2 * count, 3] = .001
    env.network.simultaneous_transmissions.power_w[:, -1] = 1
    task = env.task = cls(config, env)
    result = task.update(env)
    assert result.diagnostics['served_bits'].tolist() == [4 * count, 6 * count]
    torch.testing.assert_close(result.diagnostics['rf_energy_j'], torch.full((2,), 8e-5 * count))
    terms = import_module(f'examples.{scenario}.training_package.terms')
    assert terms.team_reward(env, role_id=roles[0]).shape == (2, 2)
    assert terms.remaining_time(env, role_id=roles[0]).shape == (2, 2, 1)
    env.physics_outputs.crashed[roles[0]][0] = True
    task.update(env)
    assert task.reward[0] == pytest.approx(-2 * 1.75)
    if scenario == 'isac_cost':
        assert task.mission_cost[0] == 1.75


def test_isac_diagnostic_keeps_raw_violation_without_cost_selection(ledger):
    scenario, cls, config, env = ledger
    task = cls(config, env)
    if scenario == 'isac_cost':
        env.sensing_outputs['fim'].valid[0, 0, 0] = False
        env.sensing_outputs['fim'].data[1] = 0
        result = task.update(env)
        assert result.diagnostics['position_violation'].tolist() == [1, 1]
        assert result.diagnostics['mission_cost'].tolist() == [1, 1]
    else:
        result = task.update(env)
        assert set(result.diagnostics) == {'episode_decision', 'served_bits', 'service_valid',
                                           'rf_energy_j', 'failed', 'task_done', 'task_reward'}
