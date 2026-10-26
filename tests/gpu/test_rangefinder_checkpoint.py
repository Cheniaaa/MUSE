"""真实Worker训练与Catalog恢复后，首次step前发布新episode的真实测距。"""
import json
from pathlib import Path
import sqlite3

import torch
from muse_sdk.run_protocol import RunRequest

from flight_helpers import archive
from test_imu_checkpoint import execute_with_preview


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / 'examples/rangefinder_flight'


def test_worker_checkpoint_restore_has_initial_rangefinder_before_first_step(tmp_path, monkeypatch):
    from muse_builtins.metadata import export_metadata
    from muse_runtime import checkpoint as checkpoint_module, driver
    from muse_server.catalog import Catalog
    from muse_server.resolver import resolve_experiment

    metadata = tmp_path / 'builtin.json'
    metadata.write_text(export_metadata(ROOT / 'examples/basic_flight/controller.json').model_dump_json())
    catalog = Catalog(tmp_path / 'catalog', metadata)
    request = json.loads((EXAMPLE / 'request_train.json').read_text())
    for kind, source in [('environment', ROOT / 'examples/basic_flight/environment'),
                         ('training_package', EXAMPLE / 'training_package')]:
        resource = catalog.register_archive(kind, archive(source, tmp_path / (kind + '.zip')))
        request[kind + '_id'], request[kind + '_revision'] = resource.resource_id, resource.revision
    request['algorithm_config'].update(collection_steps=4, num_learning_epochs=1, num_mini_batches=1)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    # 8 decisions = 160 physics ticks；4 decisions的旧IMU预算不足以触发5Hz range更新。
    source = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='TRAIN', max_decision_steps=8), device='cuda:0')

    original_run, original_load = driver.run, checkpoint_module.load_checkpoint
    loaded, published_ticks, restored = [], [], []

    def load_spy(plan, algorithm):
        original_load(plan, algorithm)
        loaded.append(plan.checkpoint.artifact_id)

    def run_spy(env, algorithm, *args, **kwargs):
        if kwargs['mode'] == 'TRAIN':
            original_sink = kwargs['checkpoint_sink']

            def checkpoint_sink(*, steps, updates):
                assert steps == 8 and updates == 2
                ticks = {group: outputs['range'].sample_tick.clone()
                         for group, outputs in env.sensor_outputs.items()}
                assert set(ticks) == {'scout_group', 'guard_group'}
                assert all((tick >= 100).all() for tick in ticks.values())
                published_ticks.append(ticks)
                return original_sink(steps=steps, updates=updates)

            kwargs['checkpoint_sink'] = checkpoint_sink
        else:
            # 位于真实load_checkpoint之后、原driver首次act/step之前；绝不手动reset。
            assert kwargs['mode'] == 'EVALUATE'
            assert loaded == [artifact['id']]
            assert published_ticks
            assert env.episode_length_buf.count_nonzero() == 0
            assert env.sim.data.time.count_nonzero() == 0
            assert set(env.sensor_outputs) == {'scout_group', 'guard_group'}
            for group_id, outputs in env.sensor_outputs.items():
                assert set(outputs) == {'gyro', 'specific_force', 'range'}
                for sample in outputs.values():
                    assert sample.valid.all()
                    assert sample.age.count_nonzero() == sample.sample_tick.count_nonzero() == 0
                sample = outputs['range']
                # 初始body z=1，名义site z偏移=-.01，下视平面z=0，沿束距离=.99m。
                torch.testing.assert_close(sample.data, torch.full_like(sample.data, .99), atol=2e-5, rtol=0)
                for slot in range(sample.data.shape[1]):
                    torch.testing.assert_close(sample.data[:, slot],
                        env.scene[f'{group_id}_{slot}/rangefinder_down'].data, atol=2e-5, rtol=0)
                # 组合profile仍保留原IMU的自由落体初值，不能填入+g。
                for signal in ('gyro', 'specific_force'):
                    torch.testing.assert_close(outputs[signal].data,
                        torch.zeros_like(outputs[signal].data), atol=1e-5, rtol=0)
            for observations in env.policy_input.observations.actors.values():
                distance = observations['rangefinder_distance']
                torch.testing.assert_close(distance, torch.full_like(distance, .99), atol=2e-5, rtol=0)
                valid = observations['rangefinder_valid']
                torch.testing.assert_close(valid, torch.ones_like(valid))
                assert observations['rangefinder_age'].count_nonzero() == 0
            restored.append(True)
        return original_run(env, algorithm, *args, **kwargs)

    monkeypatch.setattr(checkpoint_module, 'load_checkpoint', load_spy)
    monkeypatch.setattr(driver, 'run', run_spy)
    result, protocol = execute_with_preview(source, monkeypatch)
    assert result == {'state': 'COMPLETED', 'steps': 8, 'updates': 2}
    assert published_ticks
    artifact, = [payload for kind, payload in protocol.events if kind == 'checkpoint']
    assert (Path(source.artifact_directory) / artifact['path']).is_file()

    # 只登记真实Worker产物，恢复引用及兼容性检查继续走Catalog正式路径。
    record = {'run_id': source.run_id, 'artifact_directory': source.artifact_directory,
              'plan_path': str(Path(source.artifact_directory) / 'resolved_run_plan.json'),
              'artifacts': [artifact]}
    with sqlite3.connect(catalog.database) as db:
        db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, record TEXT NOT NULL)')
        db.execute('INSERT INTO runs VALUES (?, ?)', (source.run_id, json.dumps(record)))
    target = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='EVALUATE',
        checkpoint_artifact_id=artifact['id'], max_decision_steps=2), device='cuda:0')
    assert target.checkpoint.artifact_id == artifact['id']
    result, _ = execute_with_preview(target, monkeypatch)
    assert result == {'state': 'COMPLETED', 'steps': 2, 'updates': 0}
    assert restored == [True]
