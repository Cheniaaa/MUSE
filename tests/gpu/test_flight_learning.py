import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/basic_flight'


@pytest.fixture(scope='module')
def training_case(tmp_path_factory):
    from muse_builtins.metadata import export_metadata
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment
    root = tmp_path_factory.mktemp('ippo-flight')
    metadata = root / 'builtin.json'
    metadata.write_text(export_metadata(EXAMPLE / 'controller.json').model_dump_json())
    catalog = Catalog(root / 'catalog', metadata)
    request = json.loads((EXAMPLE / 'request_train.json').read_text())
    for kind, directory in [('environment', 'environment'), ('training_package', 'training_package_ippo')]:
        archive = root / (kind + '.zip')
        with zipfile.ZipFile(archive, 'w') as stream:
            for path in (EXAMPLE / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    stream.write(path, path.relative_to(EXAMPLE / directory))
        resource = catalog.register_archive(kind, archive)
        request[kind + '_id'] = resource.resource_id
        request[kind + '_revision'] = resource.revision
    request['algorithm_config'] = {'collection_steps': 4, 'num_learning_epochs': 2, 'num_mini_batches': 1}
    request['term_config'] = {'world': {'termination': {'episode_limit': {'config': {'episode_max_steps': 4}}}}}
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment)
    return catalog, identity, request


def test_training_schedule_and_mode_gates(training_case):
    from muse_sdk.config import ConfigError
    catalog, identity, request = training_case
    plan = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0')
    assert plan.experiment.training_schedule.collection_steps == plan.experiment.algorithm_config['collection_steps'] == 4
    assert {s.term for s in plan.experiment.selections if s.term_kind == 'reward'} == {'builtin:progress', 'builtin:distance'}
    for mode, checkpoint in [('EVALUATE', None), ('TRAIN', 'checkpoint'), ('INTERACTIVE', None)]:
        with pytest.raises(ConfigError):
            catalog.create_run_plan(identity, mode=mode, device='cuda:0', checkpoint_artifact_id=checkpoint)


def test_controller_action_spec_beta_endpoints(training_case):
    from muse_builtins.algorithms.ippo import IPPO
    catalog, identity, _ = training_case
    experiment = catalog.get_experiment(identity)
    algorithm = IPPO(experiment.algorithm_config)
    algorithm.initialize(experiment, SimpleNamespace(device='cpu'), None)
    motion = json.loads((EXAMPLE / 'controller.json').read_text())['motion']
    for role, spec in experiment.action_specs.items():
        for name in ('fields', 'units', 'frames', 'minimum', 'maximum'):
            assert getattr(spec, name) == motion[name]
        for raw, expected in [(0., motion['minimum']), (.5, [0.]*4), (1., motion['maximum'])]:
            torch.testing.assert_close(algorithm.to_motion(role, torch.full((4,), raw)), torch.tensor(expected))


@pytest.fixture(scope='module')
def learning_env(training_case, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    catalog, identity, _ = training_case
    plan = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0')
    env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, {}), WorldRng(7, 2, 'cuda:0'))
    yield env
    env.close()


def test_two_blocks_real_training_final_values_and_rng(learning_env, monkeypatch):
    from muse_builtins.algorithms.ippo import IPPO
    from muse_runtime.driver import run
    from muse_runtime.rng import WorldRng
    env = learning_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    algorithm = IPPO(env.experiment.algorithm_config)
    algorithm.initialize(env.experiment, SimpleNamespace(device=env.device), env.rng)
    deltas, returns = [], []
    original_update, original_process = algorithm.update, algorithm.process_env_step
    final_values = {}
    def process(tr, aux):
        for role, child in algorithm.children.items():
            fields = tr.pre_reset_next_observations.actors[role]
            from tensordict import TensorDict
            local = TensorDict({key: value.flatten(0, 1).reshape(value.shape[0]*value.shape[1], -1)
                                for key, value in fields.items()}, [fields['target'].shape[0]*fields['target'].shape[1]])
            with torch.no_grad():
                final_values[role] = child.critic(local).clone()
        original_process(tr, aux)
        if tr.world_truncated.all():
            for role, child in algorithm.children.items():
                torch.testing.assert_close(child.storage.final_values[3], final_values[role])
                assert not torch.equal(tr.pre_reset_next_observations.actors[role]['previous_action'],
                                       env.policy_input.observations.actors[role]['previous_action'])
    def update():
        before = {r: torch.cat([p.detach().flatten() for p in c.actor.parameters()]).clone()
                  for r, c in algorithm.children.items()}
        expected, full_expected = {}, {}
        for role, child in algorithm.children.items():
            st = child.storage
            assert st.truncated[-1].all() and not st.terminated[-1].any()
            expected[role] = st.rewards[-1] + child.gamma * final_values[role]
            rewards, values, finals = (x.detach().cpu().squeeze(-1).tolist()
                                       for x in (st.rewards, st.values, st.final_values))
            terminal, truncated = (x.cpu().squeeze(-1).tolist() for x in (st.terminated, st.truncated))
            oracle = [[0.] * st.num_envs for _ in range(4)]
            for sample in range(st.num_envs):
                for start in range(4):
                    total, discount = values[start][sample], 1.
                    for end in range(start, 4):
                        total += discount * (rewards[end][sample] - values[end][sample]
                            + (child.gamma * finals[end][sample] if not terminal[end][sample] else 0.))
                        if terminal[end][sample] or truncated[end][sample]:
                            break
                        discount *= child.gamma * child.lam
                    oracle[start][sample] = total
            full_expected[role] = torch.tensor(oracle, device=env.device).unsqueeze(-1)
        result = original_update()
        for role, child in algorithm.children.items():
            torch.testing.assert_close(child.storage.returns[-1], expected[role], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(child.storage.returns, full_expected[role], rtol=1e-5, atol=1e-6)
            delta = (torch.cat([p.detach().flatten() for p in child.actor.parameters()]) - before[role]).abs().sum().item()
            assert delta > 0
            deltas.append((role, delta))
            assert torch.isfinite(child.storage.returns).all()
            returns.append(child.storage.returns[-1].cpu().tolist())
        return result
    monkeypatch.setattr(algorithm, 'process_env_step', process)
    monkeypatch.setattr(algorithm, 'update', update)
    events = []
    result = run(env, algorithm, lambda: None, lambda name, data: events.append(name),
                 mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert result == dict(state='COMPLETED', steps=8, updates=2)
    assert events.count('step') == 8 and events.count('update') == 2
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    reference = WorldRng(7, 2, 'cuda:0')
    ids = torch.tensor([0, 1], device=env.device)
    torch.testing.assert_close(env.rng.uniform(ids, 'reset', 'position', (3,)),
                               reference.uniform(ids, 'reset', 'position', (3,)), rtol=0, atol=0)
    print({'actor_parameter_l1_per_update': deltas, 'tail_returns': returns})


def test_rule_flight_uses_same_driver(learning_env):
    from muse_builtins.rules import ConstantVelocity
    from muse_runtime.driver import run
    env = learning_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    algorithm = ConstantVelocity({'motion': [.05, 0., 0., 0.]})
    algorithm.initialize(env.experiment, SimpleNamespace(device=env.device), env.rng)
    before = env.role_positions_world['scout'].clone()
    result = run(env, algorithm, lambda: None, lambda *args: None,
                 mode='EVALUATE', collection_steps=None, max_decision_steps=3)
    assert result == dict(state='COMPLETED', steps=3, updates=0)
    assert torch.all(env.role_positions_world['scout'][..., 0] > before[..., 0])


def test_real_failed_step_and_partial_cancel(learning_env):
    from muse_builtins.algorithms.ippo import IPPO
    from muse_runtime.driver import run
    env = learning_env
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    algorithm = IPPO(env.experiment.algorithm_config)
    algorithm.initialize(env.experiment, SimpleNamespace(device=env.device), env.rng)
    env.sim.data.qvel[0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        run(env, algorithm, lambda: None, lambda *args: None,
            mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert all(c.storage.pending is None and c.storage.step == 0 for c in algorithm.children.values())
    env.reset(torch.ones(2, dtype=torch.bool, device=env.device))
    initial_steps = env.common_step_counter
    result = run(env, algorithm, lambda: 'cancel' if env.common_step_counter - initial_steps == 2 else None,
                 lambda *args: None, mode='TRAIN', collection_steps=4, max_decision_steps=8)
    assert result == dict(state='CANCELLED', steps=2, updates=0)
    assert all(c.storage.step == 2 for c in algorithm.children.values())
    assert all(not c.optimizer.state for c in algorithm.children.values())


def test_production_example_defaults_run_two_blocks(training_case, gpu_stream):
    from muse_builtins.algorithms.ippo import IPPO
    from muse_runtime.driver import run
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.rng import WorldRng
    from muse_server.resolver import resolve_experiment
    catalog, _, request = training_case
    production_request = {**request, 'algorithm_config': {}, 'term_config': {}}
    report = resolve_experiment(production_request, catalog.snapshot())
    assert report.valid, report.errors
    experiment = report.resolved_experiment
    assert experiment.training_schedule.collection_steps == 24
    assert experiment.algorithm_config['num_learning_epochs'] == 4
    assert experiment.algorithm_config['num_mini_batches'] == 4
    identity = catalog.save_experiment(experiment)
    plan = catalog.create_run_plan(identity, mode='TRAIN', device='cuda:0')
    rng = WorldRng(experiment.root_seed, experiment.topology.world_count, plan.device)
    env = MuseEnv(experiment, RuntimeContext(plan.device, plan.resource_paths, {}), rng)
    try:
        algorithm = IPPO(experiment.algorithm_config)
        algorithm.initialize(experiment, SimpleNamespace(device=plan.device), rng)
        before = {r: torch.cat([p.detach().flatten() for p in c.actor.parameters()]).clone()
                  for r, c in algorithm.children.items()}
        result = run(env, algorithm, lambda: None, lambda *args: None,
                     mode='TRAIN', collection_steps=24, max_decision_steps=48)
        assert result == dict(state='COMPLETED', steps=48, updates=2)
        for role, child in algorithm.children.items():
            after = torch.cat([p.detach().flatten() for p in child.actor.parameters()])
            assert torch.isfinite(after).all() and not torch.equal(after, before[role])
            assert child.storage.step == 0
    finally:
        env.close()
