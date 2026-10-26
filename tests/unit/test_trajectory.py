import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from recording_helpers import sample_values


def unpack(plan, artifact):
    directory = Path(plan.artifact_directory) / 'unpacked'
    with zipfile.ZipFile(Path(plan.artifact_directory) / artifact.path) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        archive.extractall(directory)
    return directory


def test_trajectory_layout_budget_uses_actual_fields(recording_source):
    from muse_sdk.trajectory import TrajectoryWriter
    plan, model = recording_source
    writer = TrajectoryWriter(plan, model)
    try:
        # 四body位姿48 floats；三slot各6速度+4动作+1reward；15角色bool+2world bool。
        assert writer.row_bytes == 56 + 4 * (48 + 33) + 8 + 17
        assert writer.rows_per_world == 65536 // (2 * 405)
        assert not any(field.name in ('world_reward', 'world_cost', 'action_channel') for field in writer.fields)
    finally:
        writer.close()


def test_trajectory_round_trip_keeps_boundary_and_dtypes(recording_source):
    from muse_sdk.experiment import RadioActionSpec, ResolvedCost
    from muse_sdk.trajectory import TrajectoryWriter, TrajectoryReader
    plan, model = recording_source
    plan.experiment.action_specs['scout'].radio = RadioActionSpec(
        power_minimum_w=0., power_maximum_w=1., channel_ids=[-7, 12])
    plan.experiment.costs.roles['scout'] = {'energy': ResolvedCost(cost_id='energy', scope='role',
        units='J', statistic='episode_discounted_sum', boundary='world_or_learning_done',
        discount=1., budget=1., lambda_initial=0., lambda_learning_rate=.1)}
    writer = TrajectoryWriter(plan, model)
    for world in (0, 1):
        writer.append(world, (0, 0, 0, 0, 0, 0, 0), sample_values(writer.fields))
    writer.append(0, (1, 2, 0, 2, 40, 1, 0), sample_values(writer.fields, step=2, terminal=True))
    writer.append(0, (2, 2, 1, 0, 0, 3, 1), sample_values(writer.fields))
    artifact = writer.seal('cancelled', 4)
    writer.close()
    path = Path(plan.artifact_directory) / artifact.path
    assert path.stat().st_size == artifact.size
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
    reader = TrajectoryReader(unpack(plan, artifact))
    try:
        assert reader.manifest.summary['worlds']['0']['sample_count'] == 3
        assert reader.manifest.summary['worlds']['1']['sample_count'] == 1
        _, _, initial = reader.frame(0, 0)
        assert initial['roles']['scout']['action'] is None
        assert initial['roles']['scout']['reward'] is None
        xpos, xmat, step = reader.frame(0, 1)
        np.testing.assert_array_equal(xpos[:, 0], [2, 2, 2, 2])
        np.testing.assert_array_equal(xmat[1], np.eye(3).reshape(9))
        assert step['world']['world_end_reason'] == 2**40
        assert step['roles']['scout']['action']['radio']['channel'] == [-7]
        assert step['roles']['scout']['reward'] == [2.25]
        assert step['roles']['scout']['costs']['energy'] == [2.25]
        assert step['objects']['floor']['position'] == [2.5, 0., 1.]
        assert step['objects']['guard_group:1']['role_slot'] == 1
        assert step['sim_time'] == .08
        np.testing.assert_array_equal(reader.episode_samples(0, 0), [0, 1])
        assert reader.manifest.episodes['0'][0]['complete'] is True
        assert reader.manifest.episodes['0'][1]['complete'] is False
        arrays = np.load(reader.directory / 'worlds/0/integer.npy', allow_pickle=False)
        channel = next(i for i, f in enumerate(reader.manifest.fields) if f.name == 'action_channel')
        reason = next(i for i, f in enumerate(reader.manifest.fields) if f.name == 'world_end_reason')
        assert arrays[f'field_{channel}'].dtype == np.dtype('<i4')
        assert arrays[f'field_{reason}'].dtype == np.dtype('<i8')
        with pytest.raises(ValueError):
            reader.frame(0, -1)
        with pytest.raises(ValueError):
            reader.frame(2, 0)
    finally:
        reader.close()


def test_trajectory_capacity_and_too_small_budget(recording_source):
    from muse_sdk.trajectory import TrajectoryWriter
    plan, model = recording_source
    plan.recording.max_data_bytes = 2 * 405 * 3
    writer = TrajectoryWriter(plan, model)
    try:
        assert writer.rows_per_world == 3
        writer.append(0, (0, 0, 0, 0, 0, 0, 0), sample_values(writer.fields))
        assert writer.can_append([0, 1], 2)
        writer.append(0, (1, 3, 0, 3, 60, 1, 0), sample_values(writer.fields, step=3))
        assert not writer.can_append([0, 1], 2)
        assert writer.can_append([1], 2)
    finally:
        writer.close()
    plan.recording.max_data_bytes -= 1
    with pytest.raises(ValueError, match='three|3'):
        TrajectoryWriter(plan, model)


@pytest.mark.parametrize('damage', ['revision', 'shape', 'object', 'missing', 'index', 'layout', 'model'])
def test_trajectory_reader_rejects_invalid_archive(recording_source, recorded_trajectory, damage):
    from muse_sdk.trajectory import TrajectoryReader
    plan, _ = recording_source
    directory = unpack(plan, recorded_trajectory)
    if damage in ('revision', 'layout'):
        path = directory / 'manifest.json'
        manifest = json.loads(path.read_text())
        if damage == 'revision':
            manifest['format_revision'] = 2
        else:
            manifest['fields'][0]['offset'] = 4
        path.write_text(json.dumps(manifest))
    elif damage == 'missing':
        (directory / 'worlds/0/boolean.npy').unlink()
    elif damage == 'model':
        (directory / 'model.mjb').write_bytes(b'broken')
    else:
        path = directory / 'worlds/0/index.npy'
        array = np.load(path)
        if damage == 'shape':
            array = array[:, :6]
        elif damage == 'object':
            array = np.array([{'bad': 1}], dtype=object)
        else:
            array[1, 0] = 8
        np.save(path, array)
    with pytest.raises((ValueError, FileNotFoundError)):
        TrajectoryReader(directory)
