"""经真实注册、resolver、物理 step 的命名 cost 边界验收。"""
import copy
from pathlib import Path
import shutil

import pytest
import torch
import yaml

from flight_helpers import archive
from mappo_helpers import EXAMPLE, cooperative_pipeline


# 仅扩展示例包的 Term；保留真实 Task、场景、控制器及 step/reset。
COST_PACKAGE = '''

class RoleCost:
    constructed = 0

    def __init__(self, cfg, env):
        import torch
        type(self).constructed += 1
        self.value = torch.zeros_like(env.role_positions_world[cfg.params['role_id']][..., 0])
        self.calls = 0
        self.fault = None

    def __call__(self, env, *, role_id):
        import torch
        self.calls += 1
        self.value.add_(torch.arange(1, self.value.shape[1] + 1, device=env.device))
        if self.fault == 'shape':
            return self.value[..., None]
        if self.fault == 'dtype':
            return self.value.double()
        if self.fault == 'device':
            return self.value.cpu()
        if self.fault == 'nonfinite':
            return self.value * float('nan')
        return self.value

    def reset(self, env_ids):
        self.value[env_ids] = 0


class WorldCost:
    constructed = 0

    def __init__(self, cfg, env):
        import torch
        type(self).constructed += 1
        self.value = torch.zeros(env.num_envs, device=env.device)
        self.calls = 0
        self.fault = None

    def __call__(self, env):
        self.calls += 1
        self.value.add_(10)
        if self.fault == 'shape':
            return self.value[:, None]
        if self.fault == 'dtype':
            return self.value.double()
        if self.fault == 'device':
            return self.value.cpu()
        if self.fault == 'nonfinite':
            return self.value * float('inf')
        return self.value

    def reset(self, env_ids):
        self.value[env_ids] = 0


_base_package = build_training_package


def build_training_package():
    definition = _base_package()
    for name in ('energy', 'effort'):
        definition.terms[name] = TermRegistration(RoleCost, 'cost', 'role', EmptyConfig,
            (), tensor(['B', 'N_role'], units='J'), cost_id=name)
    definition.terms['risk'] = TermRegistration(WorldCost, 'cost', 'world', EmptyConfig,
        (), tensor(['B'], units='1'), cost_id='risk')
    return definition
'''


@pytest.fixture(scope='module')
def cost_plans(pipeline, tmp_path_factory):
    from muse_server.resolver import resolve_experiment

    root = tmp_path_factory.mktemp('cost-env')
    catalog, request = cooperative_pipeline(pipeline, root)
    package = root / 'cost_package'
    shutil.copytree(EXAMPLE / 'training_package', package)
    entry = package / '__init__.py'
    entry.write_text(entry.read_text() + COST_PACKAGE)
    composition_path = package / 'training_package.yaml'
    package_config = yaml.safe_load(composition_path.read_text())
    composition = package_config['compositions']['default']
    for role, name in [('scout', 'energy'), ('rover', 'effort')]:
        composition['agent_roles'][role]['cost_terms'] = [{'selection_id': name, 'term': name}]
    composition['world_cost_terms'] = [{'selection_id': 'risk', 'term': 'risk'}]
    composition_path.write_text(yaml.safe_dump(package_config, sort_keys=False))
    resource = catalog.register_archive('training_package', archive(package, root / 'cost.zip'))
    request.update(training_package_id=resource.resource_id, training_package_revision=resource.revision)
    constraint = dict(discount=.9, budget=2., lambda_initial=.1, lambda_learning_rate=.01)
    request['cost_config'] = dict(roles={'scout': {'energy': constraint}, 'rover': {'effort': constraint}},
                                  world={'risk': constraint})
    plans = {}
    for enabled in (True, False):
        candidate = copy.deepcopy(request)
        if not enabled:
            candidate['cost_config'] = {'roles': {}, 'world': {}}
            candidate['term_config'] = {
                'roles': {role: {'cost': {name: {'enabled': False}}}
                          for role, name in [('scout', 'energy'), ('rover', 'effort')]},
                'world': {'cost': {'risk': {'enabled': False}}},
            }
        report = resolve_experiment(candidate, catalog.snapshot())
        assert report.valid, report.errors
        identity = catalog.save_experiment(report.resolved_experiment, source=candidate)
        plans[enabled] = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0', max_decision_steps=8)
    return plans


@pytest.fixture
def cost_env_factory(cost_plans, gpu_stream):
    from contextlib import ExitStack
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng

    with ExitStack() as stack:
        def build(enabled=True):
            plan = cost_plans[enabled]
            package = stack.enter_context(load_package(Path(plan.resource_paths['training_package'])))
            env = MuseEnv(plan.experiment, RuntimeContext(plan.device, plan.resource_paths,
                package.definition.terms, package.definition.tasks['team_progress']), WorldRng(7, 2, plan.device))
            stack.callback(env.close)
            return env
        yield build


def zero_actions(env):
    from muse_sdk.runtime_types import MotionAction
    return {role: MotionAction(torch.zeros(spec.shape, device=env.device))
            for role, spec in env.experiment.action_specs.items()}


def cost_instances(env):
    return {cfg.cost_id: cfg.func for manager in (env.cost_manager, env.world_cost_manager)
            for cfg in manager.cfg.values()}


def test_disabled_costs_have_empty_layout_without_term_construction(cost_env_factory):
    env = cost_env_factory(False)
    assert env.experiment.costs.roles == {} and env.experiment.costs.world == {}
    assert env.cost_manager is None and env.world_cost_manager is None
    assert env.step(zero_actions(env)).transition.costs == {'roles': {}, 'world': {}}
    assert all(env.term_registry[name].implementation.constructed == 0 for name in ('energy', 'effort', 'risk'))


def test_named_costs_different_role_slots_and_single_class_evaluation(cost_env_factory):
    env = cost_env_factory()
    assert set(env.experiment.costs.roles) == {'scout', 'rover'}
    assert set(env.experiment.costs.world) == {'risk'}
    terms = cost_instances(env)
    assert len({id(term) for term in terms.values()}) == 3
    assert all(term.calls == 0 for term in terms.values())
    out = env.step(zero_actions(env))
    costs = out.transition.costs
    assert set(costs) == {'roles', 'world'}
    assert set(costs['roles']) == {'scout', 'rover'}
    assert set(costs['roles']['scout']) == {'energy'}
    assert set(costs['roles']['rover']) == {'effort'}
    assert set(costs['world']) == {'risk'}
    torch.testing.assert_close(costs['roles']['scout']['energy'], torch.ones(2, 1, device=env.device))
    torch.testing.assert_close(costs['roles']['rover']['effort'],
                               torch.tensor([[1., 2.], [1., 2.]], device=env.device))
    torch.testing.assert_close(costs['world']['risk'], torch.full((2,), 10., device=env.device))
    assert all(term.calls == 1 for term in terms.values())
    env.step(zero_actions(env))
    assert all(term.calls == 2 for term in terms.values())
    assert costs['world']['risk'].tolist() == [10., 10.]


@pytest.mark.parametrize('terminal', [False, True], ids=['timeout', 'terminal_wins_timeout'])
def test_pre_reset_observations_and_costs_own_snapshots(cost_env_factory, terminal):
    env = cost_env_factory()
    env.step(zero_actions(env))
    env.episode_length_buf[0] = 7
    if terminal:
        motion = env.action_manager.get_term(env.experiment.bindings['scout'].group_id)
        env.sim.data.qpos[0, motion.qpos_ids[0, 3:7]] = torch.tensor([0., 1., 0., 0.], device=env.device)
        env.sim.forward()
    out = env.step(zero_actions(env))
    transition = out.transition
    assert transition.world_terminated.tolist() == [terminal, False]
    assert transition.world_truncated.tolist() == [not terminal, False]
    assert transition.world_reset.tolist() == [True, False]
    phase = transition.pre_reset_next_observations.critic_inputs['team']['phase']
    assert phase.tolist() == [[8.], [2.]]
    assert out.policy_observations.critic_inputs['team']['phase'].tolist() == [[0.], [2.]]
    terms = cost_instances(env)
    saved = [transition.costs['roles']['scout']['energy'], transition.costs['roles']['rover']['effort'],
             transition.costs['world']['risk']]
    expected = [[[2.], [2.]], [[2., 4.], [2., 4.]], [20., 20.]]
    for value, name, oracle in zip(saved, ('energy', 'effort', 'risk'), expected):
        assert value.tolist() == oracle
        assert value.data_ptr() != terms[name].value.data_ptr()
        assert not terms[name].value[0].any()
        torch.testing.assert_close(terms[name].value[1], value[1])
        assert terms[name].calls == 2
    qpos = env.sim.data.qpos[1].clone()
    env.reset(torch.tensor([True, False], device=env.device))
    assert torch.equal(env.sim.data.qpos[1], qpos)
    for value, name in zip(saved, ('energy', 'effort', 'risk')):
        torch.testing.assert_close(terms[name].value[1], value[1])
    env.step(zero_actions(env))
    assert phase.tolist() == [[8.], [2.]]
    for value, oracle in zip(saved, expected):
        assert value.tolist() == oracle
    assert terms['risk'].value.tolist() == [10., 30.]


@pytest.mark.parametrize('scope', ['role', 'world'])
@pytest.mark.parametrize('fault', ['shape', 'dtype', 'device', 'nonfinite'])
def test_invalid_cost_step_is_not_committed_by_driver(cost_env_factory, scope, fault):
    from muse_runtime.driver import run
    from muse_sdk.runtime_types import ActResult

    env = cost_env_factory()
    terms = cost_instances(env)
    failing = terms['energy' if scope == 'role' else 'risk']
    events, checkpoints = [], []

    class Collector:
        def __init__(self):
            self.transitions = []
            self.discarded = 0
            self.updates = 0

        def set_mode(self, mode):
            self.mode = mode

        def act(self, observations, masks, rng, deterministic):
            if self.transitions:
                failing.fault = fault
            return ActResult(zero_actions(env), None)

        def process_env_step(self, transition, aux):
            self.transitions.append(transition)

        def discard_pending(self):
            self.discarded += 1

        def update(self):
            self.updates += 1
            return {}

    collector = Collector()
    with pytest.raises((ValueError, RuntimeError), match='output'):
        run(env, collector, lambda: None, lambda *args: events.append(args), mode='TRAIN',
            collection_steps=2, max_decision_steps=2,
            checkpoint_sink=lambda **kwargs: checkpoints.append(kwargs))
    assert len(collector.transitions) == 1
    assert collector.transitions[0].costs['world']['risk'].tolist() == [10., 10.]
    assert collector.discarded == 1 and collector.updates == 0
    assert len(events) == 1 and events[0][0] == 'step' and events[0][1]['steps'] == 1
    assert checkpoints == []
    assert failing.calls == 2
