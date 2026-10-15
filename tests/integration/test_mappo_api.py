"""合作包真实 CPU 注册；HTTP GPU 验收必须显式开启。"""
import os
from types import SimpleNamespace

import pytest

from mappo_helpers import EXAMPLE, cooperative_pipeline, example_definition


def test_cooperative_registration_and_resolution(pipeline, tmp_path):
    from muse_server.resolver import resolve_experiment
    catalog, request = cooperative_pipeline(pipeline, tmp_path)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    exp = report.resolved_experiment
    assert {r: sum(f.tensor.shape[-1] for f in fields) for r, fields in exp.algorithm_inputs.actors.items()} == {'scout': 17, 'rover': 9}
    assert exp.action_specs['scout'].shape == [2, 1, 4]
    assert exp.action_specs['rover'].shape == [2, 2, 2]
    assert [f.tensor.shape for f in exp.algorithm_inputs.critics['team']] == [[2, 1, 3], [2, 1, 3], [2, 2, 2], [2, 2, 2], [2, 1]]
    rewards = [s for s in exp.selections if s.term_kind == 'reward']
    assert len(rewards) == 1 and rewards[0].owner_path == ('world',)
    assert exp.task_spec.learning_done_roles == ['scout']
    assert exp.task_spec.published_fields['phase'].shape == [2, 1]
    assert exp.task_config == {'learning_horizon': 3}
    assert exp.training_schedule.collection_steps == 4
    identity = catalog.save_experiment(exp)
    assert catalog.create_run_plan(identity, mode='TRAIN', device='cpu', max_decision_steps=8).experiment == exp
    request['algorithm_config']['share_actor_parameters'] = False
    assert resolve_experiment(request, catalog.snapshot()).valid
    request['task_config'] = {'learning_horizon': 0}
    assert not resolve_experiment(request, catalog.snapshot()).valid


def test_team_task_phase_pulse_and_subset_reset():
    import torch
    from muse_sdk.runtime_types import RoleMask
    definition = example_definition()
    masks = {'scout': RoleMask(*(torch.ones(2, 1, dtype=torch.bool) for _ in range(3)))}
    env = SimpleNamespace(num_envs=2, device='cpu', episode_length_buf=torch.tensor([2, 3]), role_masks=masks)
    task = definition.tasks['team_progress']({}, env)
    assert task.phase.shape == (2, 1) and task.phase.dtype == torch.float32
    result = task.update(env)
    assert result.role_status == {} and result.learning_done['scout'].tolist() == [[False], [True]]
    assert task.phase.tolist() == [[2.], [3.]]
    env.episode_length_buf.add_(1)
    assert task.phase.tolist() == [[2.], [3.]]
    assert task.update(env).learning_done['scout'].tolist() == [[True], [False]]
    task.reset(torch.tensor([1]))
    assert task.phase.tolist() == [[3.], [0.]]
    assert env.episode_length_buf.tolist() == [3, 4]
    assert all(t.all() for t in masks['scout'])
    assert definition.tasks['team_progress']({'learning_horizon': 4}, env).update(env).learning_done['scout'].tolist() == [[False], [True]]


def test_team_reward_weights_slots_equally_and_uses_action_targets():
    import torch
    definition = example_definition()
    env = SimpleNamespace(role_positions_world={'scout': torch.tensor([[[3., 0., 0.]], [[0., 0., 0.]]]),
        'rover': torch.tensor([[[0., 4., 0.], [0., 0., 5.]], [[0., 0., 0.], [0., 0., 0.]]])})
    targets = {r: torch.zeros_like(p) for r, p in env.role_positions_world.items()}
    env.act_commands = SimpleNamespace(target_world=targets)
    env.commands = SimpleNamespace(target_world={r: t + 99 for r, t in targets.items()})
    torch.testing.assert_close(definition.terms['team_distance'].implementation(env), torch.tensor([-4., 0.]))
    env.task = SimpleNamespace(phase=torch.tensor([[2.], [3.]]))
    torch.testing.assert_close(definition.terms['team_phase'].implementation(env), env.task.phase)


def test_package_metadata_import_without_gpu_libraries():
    import subprocess
    import sys
    script = '''
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco', 'mjlab', 'warp'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('cooperative', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
definition = module.build_training_package()
assert definition.tasks['team_progress'].describe()['learning_done_roles'] == ['scout']
'''
    result = subprocess.run([sys.executable, '-c', script, str(EXAMPLE / 'training_package/__init__.py')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.environ.get('MUSE_TEST_GPU') != '1', reason='Explicit serial GPU acceptance')
@pytest.mark.parametrize('sharing', [True, False])
def test_real_mappo_http_three_runs(tmp_path, sharing):
    import hashlib
    import io
    import json
    from pathlib import Path
    import socket
    import subprocess
    import sys

    import httpx2
    import torch
    from test_runs_api import zip_body
    from test_supervisor import wait_for

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with (tmp_path / 'server.log').open('wb') as log:
        server = subprocess.Popen([sys.executable, '-m', 'muse_server.app', '--port', str(port),
                                   '--data-dir', str(tmp_path / 'data')], stdout=log, stderr=log)
        try:
            with httpx2.Client(base_url=f'http://127.0.0.1:{port}', timeout=30.) as client:
                def healthy():
                    try:
                        return client.get('/health').is_success
                    except httpx2.HTTPError:
                        return False
                wait_for(healthy, timeout=30)
                request = json.loads((EXAMPLE / 'request_train.json').read_text())
                request['algorithm_config']['share_actor_parameters'] = sharing
                request['algorithm_config']['recurrent'] = {'type': 'gru' if sharing else 'lstm', 'hidden_size': 16}
                for kind, route in [('environment', 'environments'), ('training_package', 'training-packages')]:
                    response = client.post('/catalog/' + route, content=zip_body(EXAMPLE / kind),
                                           headers={'Content-Type': 'application/zip'})
                    assert response.status_code == 201, response.text
                    item = response.json()
                    request[kind + '_id'], request[kind + '_revision'] = item['resource_id'], item['revision']
                response = client.post('/experiments', json=request)
                assert response.status_code == 201, response.text
                identity = response.json()['experiment_id']
                reference, previous = None, None
                run_ids, artifact_ids = set(), set()
                for mode, steps, cumulative in [('TRAIN', 8, 2), ('TRAIN', 8, 4), ('EVALUATE', 3, 4)]:
                    response = client.post('/runs', json=dict(experiment_id=identity, mode=mode,
                        max_decision_steps=steps, checkpoint_artifact_id=reference))
                    assert response.status_code == 201, response.text
                    run_id = response.json()['run_id']
                    assert run_id not in run_ids
                    run_ids.add(run_id)
                    result = wait_for(lambda: (r if (r := client.get(f'/runs/{run_id}').json())['state']
                        in ('COMPLETED', 'FAILED', 'LOST', 'CANCELLED') else None), timeout=180)
                    assert result['state'] == 'COMPLETED', (result, Path(result['artifact_directory'], 'logs/worker.log').read_text())
                    assert result['exit_code'] == 0 and result['metrics']['steps'] == steps
                    assert result['metrics']['updates'] == (2 if mode == 'TRAIN' else 0)
                    artifacts = client.get(f'/runs/{run_id}/artifacts')
                    assert artifacts.status_code == 200
                    saved = [a for a in artifacts.json() if a['kind'] == 'checkpoint']
                    if mode == 'EVALUATE':
                        assert saved == []
                        continue
                    assert len(saved) == 1
                    item = saved[0]
                    assert item['id'] not in artifact_ids and item['id'].startswith(run_id + ':checkpoint:')
                    artifact_ids.add(item['id'])
                    response = client.get(f"/runs/{run_id}/artifacts/{item['id']}")
                    assert response.status_code == 200
                    content = response.content
                    assert len(content) == item['size'] and hashlib.sha256(content).hexdigest() == item['sha256']
                    envelope = torch.load(io.BytesIO(content), weights_only=True, map_location='cpu')
                    assert envelope['run_id'] == run_id and envelope['experiment_id'] == identity
                    assert envelope['steps'] == 8 and envelope['updates'] == 2
                    state = envelope['learning_state']
                    assert state['updates'] == cumulative
                    owners = {'scout', 'rover'} if sharing else {'scout/slot/0', 'rover/slot/0', 'rover/slot/1'}
                    assert set(state['roles']) == owners
                    assert state['signature']['config']['share_actor_parameters'] is sharing
                    for owner, learning in state['roles'].items():
                        assert learning['parameter_names']
                        adam = learning['optimizer']['state']
                        assert adam and all(set(v) == {'step', 'exp_avg', 'exp_avg_sq'} for v in adam.values())
                        assert all(torch.isfinite(t).all() for kind in ('actor', 'critic') for t in learning[kind].values())
                        if previous is not None:
                            before = previous['learning_state']['roles'][owner]
                            for kind in ('actor', 'critic'):
                                assert any(not torch.equal(t, learning[kind][name]) for name, t in before[kind].items())
                            assert learning['parameter_names'] == before['parameter_names']
                            assert adam.keys() == before['optimizer']['state'].keys()
                            assert all(v['step'] > before['optimizer']['state'][key]['step'] for key, v in adam.items())
                    if previous is not None:
                        assert envelope['learning_contract'] == previous['learning_contract']
                    previous, reference = envelope, item['id']
                assert len(run_ids) == 3 and len(artifact_ids) == 2
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
