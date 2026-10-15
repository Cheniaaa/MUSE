import json
from pathlib import Path
import zipfile
import copy
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/basic_flight'


@pytest.fixture(scope='module')
def flight_plan(tmp_path_factory):
    return build_plan(tmp_path_factory.mktemp('flight-env'))


def build_plan(root, progress=False, shared_group=False, worlds=2, history=0):
    from muse_builtins.metadata import export_metadata
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment
    metadata = root / 'builtin.json'
    metadata.write_text(export_metadata(EXAMPLE / 'controller.json').model_dump_json())
    catalog = Catalog(root / 'catalog', metadata)
    resources = {}
    for kind, directory in [('environment', 'environment'), ('training_package', 'training_package')]:
        archive = root / f'{kind}.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            for path in (EXAMPLE / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    if progress and path.name == 'training_package.yaml':
                        import yaml
                        composition = yaml.safe_load(path.read_text())
                        for role in composition['agent_roles'].values():
                            role['reward_terms'][0]['term'] = 'builtin:progress'
                        extra = copy.deepcopy(composition['agent_roles']['scout']['reward_terms'][0])
                        extra.update(selection_id='progress_second', weight=.25, config={'scale': 4.0})
                        composition['agent_roles']['scout']['reward_terms'].append(extra)
                        for name in ('velocity_world', 'orientation_wxyz', 'angular_velocity_body', 'previous_action'):
                            composition['agent_roles']['scout']['observation_terms'].append(
                                {'selection_id': name, 'term': 'builtin:' + name})
                        composition['critic_inputs'] = {'terminal': {'observation_terms': [
                            {'selection_id': 'guard_target', 'term': 'builtin:relative_target', 'role_id': 'guard'}]}}
                        stream.writestr('training_package.yaml', yaml.safe_dump(composition, sort_keys=False))
                    else:
                        stream.write(path, path.relative_to(EXAMPLE / directory))
        resources[kind] = catalog.register_archive(kind, archive)
    request = json.loads((EXAMPLE / 'request.json').read_text())
    request['topology']['world_count'] = worlds
    if history:
        request['term_config'] = {'roles': {role: {'observation': {'target': {'history_length': history}}}
                                          for role in request['role_config']}}
    if shared_group:
        scout, guard = request['scenario']['entity_groups']
        scout['count'] = 3
        request['scenario']['placements'][scout['placement_id']].extend(request['scenario']['placements'][guard['placement_id']])
        request['scenario']['entity_groups'] = [scout]
        request['role_config']['guard'] = {'group_id': scout['group_id'], 'slots': [1, 2]}
    for kind, resource in resources.items():
        request[f'{kind}_id'] = resource.resource_id
        request[f'{kind}_revision'] = resource.revision
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    return catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0')


@pytest.fixture(scope='module')
def flight_env(flight_plan, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    context = RuntimeContext(device='cuda:0', resource_paths=flight_plan.resource_paths, term_registry={})
    env = MuseEnv(flight_plan.experiment, context, WorldRng(42, 2, 'cuda:0'))
    yield env
    env.close()


def actions(env, value=.05):
    from muse_sdk.runtime_types import MotionAction
    return {role: MotionAction(torch.full(tuple(spec.shape), value, device=env.device))
            for role, spec in env.experiment.action_specs.items()}


def test_native_environment_and_one_decision(flight_env, monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    env = flight_env
    assert isinstance(env, ManagerBasedRlEnv)
    assert env.policy_input.observations.actors['guard']['target'].shape == (2, 2, 3)
    counts = {}
    def count(obj, method, name):
        original = getattr(obj, method)
        def wrapped(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return original(*args, **kwargs)
        monkeypatch.setattr(obj, method, wrapped)
    count(env.action_manager, 'process_action', 'process')
    for group in env.action_manager.active_terms:
        count(env.action_manager.get_term(group).controller, 'compute', group)
    for name in ['command_manager', 'observation_manager', 'reward_manager', 'termination_manager']:
        count(getattr(env, name), 'compute', name)
    out = env.step(actions(env))
    assert counts['process'] == 1
    for group in env.action_manager.active_terms:
        assert counts[group] == 20
    assert all(counts[name] == 1 for name in ['command_manager', 'observation_manager', 'reward_manager', 'termination_manager'])
    assert not out.transition.world_reset.any()
    assert out.transition.rewards['guard'].shape == (2, 2)
    assert out.transition.world_reward is None
    assert torch.isfinite(env.sim.data.qpos).all()


def test_subset_reset_buffers_and_snapshot(flight_env):
    env = flight_env
    out = env.step(actions(env))
    snapshot = out.transition.pre_reset_next_observations.actors['scout']['target'].clone()
    qpos = env.sim.data.qpos.clone()
    ptr = env.sim.data.qpos.data_ptr()
    history = env.action_manager.action.clone()
    env.reset(torch.tensor([True, False], device=env.device))
    assert env.sim.data.qpos.data_ptr() == ptr
    assert torch.equal(env.sim.data.qpos[1], qpos[1])
    assert torch.equal(env.action_manager.action[1], history[1])
    assert env.action_manager.action[0].count_nonzero() == 0
    assert torch.equal(snapshot, out.transition.pre_reset_next_observations.actors['scout']['target'])


def test_nonfinite_physics_fails_before_reset(flight_env):
    env = flight_env
    env.sim.data.qvel[0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        env.step(actions(env))
    assert torch.isnan(env.sim.data.qvel[0, 0])
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))


def test_progress_is_registered():
    from muse_builtins.terms import registrations
    assert 'progress' in registrations()


def test_flight_observations_are_registered():
    from muse_builtins.terms import registrations
    assert {'velocity_world', 'orientation_wxyz', 'angular_velocity_body', 'previous_action'} <= registrations().keys()


def test_progress_real_selections_and_task_order(tmp_path, gpu_stream):
    from dataclasses import replace
    from muse_builtins.terms import ProgressReward, registrations
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    plan = build_plan(tmp_path, progress=True)
    events = []
    class OrderedProgress(ProgressReward):
        def __init__(self, cfg, env):
            assert events == []
            super().__init__(cfg, env)
        def reset(self, env_ids):
            assert events[-1] == 'task_reset'
            super().reset(env_ids)
        def __call__(self, env, **params):
            assert events[-1] == 'task_update'
            return super().__call__(env, **params)
    class Task:
        def __init__(self, cfg, env):
            self.env = env
            self.samples = torch.zeros((2, 1), device=env.device)
        def reset(self, ids):
            events.append('task_reset')
            self.samples[ids] = self.env.rng.uniform(ids, 'task', 'reset', (1,))
        def update(self, env):
            events.append('task_update')
            from muse_sdk.runtime_types import TaskUpdateResult
            return TaskUpdateResult(role_status={}, learning_done={}, diagnostics={})
    registry = {'builtin:progress': replace(registrations()['progress'], implementation=OrderedProgress)}
    context = RuntimeContext('cuda:0', plan.resource_paths, registry, Task)
    from muse_sdk.experiment import BoundTask
    plan.experiment.task_spec = BoundTask(input_fields=[], published_fields={}, learning_done_roles=[])
    env = MuseEnv(plan.experiment, context, WorldRng(42, 2, 'cuda:0'))
    terms = list(env.reward_manager.cfg.values())
    assert [cfg.func.previous_distance.shape for cfg in terms] == [torch.Size([2, 1]), torch.Size([2, 1]), torch.Size([2, 2])]
    assert len({cfg.func.previous_distance.data_ptr() for cfg in terms}) == 3
    previous = [cfg.func.previous_distance.clone() for cfg in terms]
    out = env.step(actions(env))
    torch.testing.assert_close(out.policy_observations.actors['scout']['previous_action'], actions(env)['scout'].motion)
    current = [cfg.func.previous_distance.clone() for cfg in terms]
    expected = 2 * (previous[0]-current[0]) + (previous[1]-current[1])
    torch.testing.assert_close(out.transition.rewards['scout'], expected)
    assert events == ['task_reset', 'task_update']
    samples = env.task.samples.clone()
    generator = env.rng.generator(1, 'task', 'reset')
    state = generator.get_state().clone()
    env.reset(torch.tensor([True, False], device=env.device))
    for cfg, before in zip(terms, current):
        assert torch.equal(cfg.func.previous_distance[1], before[1])
        torch.testing.assert_close(cfg.func.previous_distance[0], torch.linalg.vector_norm(
            env.role_positions_world[cfg.params['role_id']][0] - env.commands.target_world[cfg.params['role_id']][0], dim=-1))
    assert torch.equal(env.task.samples[1], samples[1])
    assert torch.equal(generator.get_state(), state)
    env.episode_length_buf[0] = 499
    out = env.step(actions(env))
    assert out.transition.world_truncated.tolist() == [True, False]
    critic = out.transition.pre_reset_next_observations.critic_inputs['terminal']['guard_target']
    saved = critic.clone()
    assert critic.data_ptr() != out.policy_observations.critic_inputs['terminal']['guard_target'].data_ptr()
    env.step(actions(env))
    assert torch.equal(critic, saved)
    env.close()


def test_raw_observation_inf_is_rejected_before_clip(flight_plan, gpu_stream, monkeypatch):
    from dataclasses import replace
    from mjlab.managers.observation_manager import ObservationManager
    from muse_builtins.terms import registrations
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    original = ObservationManager.__init__
    def configured(manager, cfg, env):
        for group in cfg.values():
            for term in group.terms.values():
                term.clip = (-1., 1.)
        original(manager, cfg, env)
    monkeypatch.setattr(ObservationManager, '__init__', configured)
    def nonfinite(env, *, role_id):
        return torch.full_like(env.role_positions_world[role_id], float('inf'))
    registry = {'builtin:relative_target': replace(registrations()['relative_target'], implementation=nonfinite)}
    with pytest.raises((RuntimeError, ValueError), match='nonfinite|NaN/Inf'):
        MuseEnv(flight_plan.experiment, RuntimeContext('cuda:0', flight_plan.resource_paths, registry), WorldRng(1, 2, 'cuda:0'))


@pytest.mark.parametrize('field', ['nacon', 'nefc', 'overflow'])
def test_overflow_fails_before_reset(flight_env, field):
    env = flight_env
    tensor = getattr(env.sim.data, field)
    before = tensor.clone()
    tensor.fill_(env.sim.data.naconmax + env.sim.data.njmax + 1)
    try:
        with pytest.raises(RuntimeError, match='overflow'):
            env.step(actions(env))
        assert torch.equal(tensor, torch.full_like(tensor, env.sim.data.naconmax + env.sim.data.njmax + 1))
    finally:
        tensor.copy_(before)


@pytest.mark.parametrize('crash', ['flip', 'ground', 'entity'])
def test_crash_resets_whole_world_and_wins_timeout(flight_env, crash, monkeypatch):
    env = flight_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    term = env.action_manager.get_term('scout_group')
    if crash == 'flip':
        env.sim.data.qpos[0, term.qpos_ids[0, 3:7]] = torch.tensor([0., 1., 0., 0.], device=env.device)
    elif crash == 'ground':
        env.sim.data.qpos[0, term.qpos_ids[0, 2]] = 0.
        env.sim.forward()
        assert env.sim.data.nacon[0] > 0, (env.sim.data.nacon, env.sim.data.geom_xpos[0, :, 2].min())
    else:
        env.sim.data.qpos[0, term.qpos_ids[0, :3]] = torch.tensor([0., .95, 1.], device=env.device)
        env.sim.forward()
        assert 0 < env.sim.data.nacon[0] <= env.sim.data.naconmax
    before_reset = {}
    original_reset = env.reset
    def capture(mask):
        before_reset['qpos'] = env.sim.data.qpos[1].clone()
        before_reset['qvel'] = env.sim.data.qvel[1].clone()
        before_reset['action'] = env.action_manager.action[1].clone()
        before_reset['controllers'] = {group: env.action_manager.get_term(group).controller.velocity_integral[1].clone()
                                      for group in env.action_manager.active_terms}
        return original_reset(mask)
    monkeypatch.setattr(env, 'reset', capture)
    env.episode_length_buf[0] = 499
    out = env.step(actions(env, 0.))
    assert out.transition.world_terminated.tolist() == [True, False]
    assert not out.transition.world_truncated.any()
    assert out.diagnostics['termination'] == {'terminated': 1, 'truncated': 0}
    assert set(out.diagnostics['control_saturation']) == {'scout_group', 'guard_group'}
    assert all(0 <= value <= 1 for value in out.diagnostics['control_saturation'].values())
    assert out.transition.world_end_reason.tolist() == [1, 0]
    assert env.episode_length_buf.tolist() == [0, 1]
    assert env.action_manager.action[0].count_nonzero() == 0
    assert torch.equal(env.sim.data.qpos[1], before_reset['qpos'])
    assert torch.equal(env.sim.data.qvel[1], before_reset['qvel'])
    assert torch.equal(env.action_manager.action[1], before_reset['action'])
    for group in env.action_manager.active_terms:
        motion = env.action_manager.get_term(group)
        pose, velocity = motion.state()
        assert torch.equal(velocity[0], torch.zeros_like(velocity[0]))
        assert torch.equal(motion.controller.velocity_integral[0], torch.zeros_like(motion.controller.velocity_integral[0]))
        assert torch.equal(motion.controller.velocity_integral[1], before_reset['controllers'][group])
    terminal = out.transition.pre_reset_next_observations.actors['scout']['target']
    assert terminal.data_ptr() != out.policy_observations.actors['scout']['target'].data_ptr()


def test_shared_group_has_one_controller(tmp_path, gpu_stream, monkeypatch):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    plan = build_plan(tmp_path, shared_group=True)
    env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {}), WorldRng(1, 2, 'cuda:0'))
    assert len(env.action_manager.active_terms) == 1
    term = env.action_manager.get_term('scout_group')
    calls = []
    original = term.controller.compute
    def compute(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(term.controller, 'compute', compute)
    role_actions = actions(env)
    role_actions['guard'].motion.neg_()
    env.step(role_actions)
    assert len(calls) == 20
    torch.testing.assert_close(term.setpoint[:, :1], role_actions['scout'].motion)
    torch.testing.assert_close(term.setpoint[:, 1:], role_actions['guard'].motion)
    env.close()


def test_single_world_preserves_compiled_asset_and_placement(tmp_path, gpu_stream):
    import mujoco
    import numpy as np
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    plan = build_plan(tmp_path, worlds=1)
    env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {}), WorldRng(1, 1, 'cuda:0'))
    model = env.sim.mj_model
    assert model.ngeom == 3 * 39 + 1
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert model.opt.density == 1.225 and model.opt.viscosity == 1.8e-5
    assert model.opt.timestep == .002
    assert model.nkey == 1
    for group in env.action_manager.active_terms:
        term = env.action_manager.get_term(group)
        np.testing.assert_allclose(model.body_mass[term.body_ids], .027)
    torch.testing.assert_close(env.sim.data.qpos[0], torch.tensor(model.qpos0, dtype=torch.float32, device=env.device))
    assert not env.step(actions(env)).transition.world_reset.any()
    env.close()


def test_substep_nonfinite_is_not_hidden_by_timeout_reset(flight_env, monkeypatch):
    env = flight_env
    original = env.sim.step
    def broken():
        original()
        env.sim.data.qvel[0, 0] = float('inf')
    monkeypatch.setattr(env.sim, 'step', broken)
    env.episode_length_buf[0] = 499
    with pytest.raises(RuntimeError, match='nonfinite'):
        env.step(actions(env))
    assert torch.isinf(env.sim.data.qvel[0, 0])
    assert env.episode_length_buf[0] == 499
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))


def test_native_history_subset_and_all_physics_addresses(tmp_path, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    flight_plan = build_plan(tmp_path, history=3)
    cpu_rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone()
    env = MuseEnv(flight_plan.experiment, RuntimeContext('cuda:0', flight_plan.resource_paths, {}), WorldRng(1, 2, 'cuda:0'))
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    names = ['qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qacc_warmstart', 'mocap_pos', 'mocap_quat']
    pointers = {name: getattr(env.sim.data, name).data_ptr() for name in names}
    env.step(actions(env))
    env.step(actions(env))
    buffers = env.observation_manager._group_obs_term_history_buffer
    before = {group: {name: buf.buffer.clone() for name, buf in terms.items()} for group, terms in buffers.items()}
    physics = {name: getattr(env.sim.data, name).clone() for name in ['qpos', 'qvel', 'ctrl']}
    controller = {g: env.action_manager.get_term(g).controller.velocity_integral.clone() for g in env.action_manager.active_terms}
    for name in ['ctrl', 'qfrc_applied', 'xfrc_applied', 'qacc_warmstart']:
        getattr(env.sim.data, name)[0] = 1
    env.reset(torch.tensor([True, False], device=env.device))
    for group, terms in buffers.items():
        for name, buf in terms.items():
            assert torch.equal(buf.buffer[1], before[group][name][1])
    for name, value in physics.items():
        assert torch.equal(getattr(env.sim.data, name)[1], value[1])
    for name in ['ctrl', 'qfrc_applied', 'xfrc_applied']:
        assert getattr(env.sim.data, name)[0].count_nonzero() == 0
    for group, value in controller.items():
        assert torch.equal(env.action_manager.get_term(group).controller.velocity_integral[1], value[1])
    assert all(getattr(env.sim.data, name).data_ptr() == pointer for name, pointer in pointers.items())
    env.close()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 3.4e38])
def test_raw_reward_nonfinite_and_aggregation_overflow(flight_env, monkeypatch, value):
    env = flight_env
    cfg = next(iter(env.reward_manager.cfg.values()))
    def invalid(env, **params):
        return torch.full_like(env.role_positions_world[params['role_id']][..., 0], value)
    monkeypatch.setattr(cfg, 'func', invalid)
    monkeypatch.setattr(cfg, 'weight', 2.)
    with pytest.raises(RuntimeError, match='nonfinite'):
        env.step(actions(env))
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))


@pytest.mark.parametrize('case', ['bounds', 'dtype', 'shape', 'nan'])
def test_invalid_action_does_not_advance_physics(flight_env, case):
    from muse_sdk.runtime_types import MotionAction
    env = flight_env
    inputs = actions(env)
    value = inputs['scout'].motion
    if case == 'bounds':
        value.fill_(1.)
    elif case == 'dtype':
        value = value.double()
    elif case == 'shape':
        value = value[..., :3]
    else:
        value.fill_(float('nan'))
    inputs['scout'] = MotionAction(value)
    before = env.sim.data.qpos.clone()
    with pytest.raises(ValueError):
        env.step(inputs)
    assert torch.equal(before, env.sim.data.qpos)


@pytest.mark.parametrize('case', ['shape', 'dtype'])
def test_observation_frozen_output_boundary(flight_plan, gpu_stream, case):
    from dataclasses import replace
    from muse_builtins.terms import registrations
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    def invalid(env, *, role_id):
        value = env.role_positions_world[role_id]
        return value[..., :1] if case == 'shape' else value.double()
    registry = {'builtin:relative_target': replace(registrations()['relative_target'], implementation=invalid)}
    with pytest.raises(ValueError, match='shape|dtype'):
        MuseEnv(flight_plan.experiment, RuntimeContext('cuda:0', flight_plan.resource_paths, registry), WorldRng(1, 2, 'cuda:0'))


@pytest.mark.parametrize('local_timeout', [False, True])
def test_world_reward_uses_native_unscaled_manager(flight_plan, gpu_stream, local_timeout):
    from dataclasses import replace
    from mjlab.managers.reward_manager import RewardManager
    from muse_builtins.terms import registrations
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    exp = flight_plan.experiment.model_copy(deep=True)
    source = next(s for s in exp.selections if s.term_kind == 'reward')
    world = source.model_copy(update={'owner_path': ('world',), 'binding_role_id': None,
        'selection_id': 'team', 'term': 'world_test', 'config': {}, 'weight': 2.0,
        'output_spec': source.output_spec.model_copy(update={'shape': [2]})})
    exp.selections = (*exp.selections, world)
    output = {'value': 1., 'calls': 0}
    def world_reward(env):
        output['calls'] += 1
        return torch.full((env.num_envs,), output['value'], device=env.device)
    registry = {'world_test': replace(registrations()['distance'], scope='world', implementation=world_reward)}
    if local_timeout:
        for selection in exp.selections:
            if selection.term == 'builtin:timeout':
                selection.term = 'local_timeout'
        registry['local_timeout'] = registrations()['timeout']
    env = MuseEnv(exp, RuntimeContext('cuda:0', flight_plan.resource_paths, registry), WorldRng(1, 2, 'cuda:0'))
    assert isinstance(env.world_reward_manager, RewardManager)
    torch.testing.assert_close(env.step(actions(env)).transition.world_reward, torch.full((2,), 2., device=env.device))
    assert output['calls'] == 1
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    assert float(env.extras['log']['Episode_Reward/world.reward.team']) == 2.
    output['value'] = 3.4e38
    with pytest.raises(RuntimeError, match='nonfinite'):
        env.step(actions(env))
    env.close()


def test_step_owns_torch_warp_stream_alignment(flight_env, monkeypatch):
    import warp as wp
    env = flight_env
    original = env.action_manager.apply_action
    def checked():
        assert wp.get_stream(env.device).cuda_stream == torch.cuda.current_stream(env.device).cuda_stream
        original()
    monkeypatch.setattr(env.action_manager, 'apply_action', checked)
    with wp.ScopedStream(wp.Stream(env.device)):
        env.step(actions(env))


def test_reset_forward_checks_capacity(flight_env, monkeypatch):
    env = flight_env
    original = env.sim.forward
    def overflowing():
        original()
        env.sim.data.overflow.fill_(1)
    with monkeypatch.context() as patch:
        patch.setattr(env.sim, 'forward', overflowing)
        with pytest.raises(RuntimeError, match='overflow'):
            env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))


def test_episode_reward_accumulation_cannot_overflow(flight_env, monkeypatch):
    env = flight_env
    cfg = next(iter(env.reward_manager.cfg.values()))
    def large(env, **params):
        return torch.full_like(env.role_positions_world[params['role_id']][..., 0], 2e38)
    monkeypatch.setattr(cfg, 'func', large)
    monkeypatch.setattr(cfg, 'weight', 1.)
    env.step(actions(env))
    with pytest.raises(RuntimeError, match='nonfinite'):
        env.step(actions(env))
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
