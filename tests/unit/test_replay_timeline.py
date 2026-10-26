"""显式单调时钟的CPU历史游标；不启动窗口或物理。"""
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


@pytest.fixture
def timeline(recording_source, recorded_trajectory, tmp_path):
    from muse_sdk.trajectory import TrajectoryReader
    from muse_server.storage import extract_archive
    from muse_server.viewers.replay_process import ReplayTimeline
    plan, _ = recording_source
    directory = tmp_path / 'decoded'
    extract_archive(Path(plan.artifact_directory) / recorded_trajectory.path, directory)
    reader = TrajectoryReader(directory)
    try:
        yield ReplayTimeline(reader)
    finally:
        reader.close()


def command(sequence, kind, **target):
    from muse_sdk.replay_protocol import ReplayCommand
    return ReplayCommand(command_seq=sequence, kind=kind, **target)


def test_backward_seek_updates_scene_and_diagnostics_together(timeline):
    timeline.apply(command(1, 'seek', sample_index=8), 10.)
    before = timeline.revision
    result = timeline.apply(command(2, 'seek', sample_index=2), 11.)
    xpos, _, diagnostics = timeline.frame()
    assert result['status'] == 'applied'
    assert timeline.revision > before
    assert diagnostics['sample_index'] == 2
    assert diagnostics['run_decision'] == 6
    np.testing.assert_allclose(xpos[1], diagnostics['objects']['scout_group:0']['position'])


def test_step_visits_same_time_terminal_and_reset_rows(timeline):
    timeline.apply(command(1, 'seek', sample_index=3), 0.)
    before = timeline.frame()[2]
    timeline.apply(command(2, 'step', delta=1), 0.)
    after = timeline.frame()[2]
    assert before['run_decision'] == after['run_decision'] == 7
    assert (before['phase'], after['phase']) == ('step', 'reset_after')
    assert after['sample_index'] == 4 and after['episode'] == 1
    timeline.apply(command(3, 'step', delta=-1), 0.)
    assert timeline.frame()[2]['sample_index'] == 3


def test_world_episode_and_step_pause_and_respect_edges(timeline):
    timeline.apply(command(1, 'play'), 0.)
    timeline.apply(command(2, 'world', world_id=1), .01)
    assert not timeline.playing and timeline.frame()[2]['world_id'] == 1
    assert timeline.frame()[2]['sample_index'] == 0
    timeline.apply(command(3, 'step', delta=-1), .01)
    assert timeline.frame()[2]['sample_index'] == 0
    timeline.apply(command(4, 'play'), .01)
    timeline.apply(command(5, 'episode', episode=1), .01)
    assert not timeline.playing and timeline.frame()[2]['sample_index'] == 4
    timeline.apply(command(6, 'episode', episode=2), .01)
    timeline.apply(command(7, 'step', delta=1), .01)
    assert timeline.frame()[2]['sample_index'] == 8


def test_playback_uses_run_clock_speed_and_stops_at_last_row(timeline):
    dt = timeline.reader.manifest.physics_dt * timeline.reader.manifest.decimation
    timeline.apply(command(1, 'speed', speed=2.), 100.)
    timeline.apply(command(2, 'play'), 100.)
    assert not timeline.advance(100. + dt)
    assert timeline.advance(100. + 3.5 * dt)
    assert timeline.frame()[2]['sample_index'] == 4
    assert timeline.advance(100. + 100. * dt)
    assert timeline.frame()[2]['sample_index'] == 8 and not timeline.playing
    revision = timeline.revision
    assert not timeline.advance(1000.) and timeline.revision == revision


def test_bad_range_and_old_sequence_do_not_move_cursor(timeline):
    timeline.apply(command(1, 'seek', sample_index=2), 0.)
    for sequence, kind, target in [(2, 'seek', {'sample_index': 100}),
                                    (3, 'world', {'world_id': 99}),
                                    (4, 'episode', {'episode': 99})]:
        result = timeline.apply(command(sequence, kind, **target), 0.)
        assert result['status'] == 'error' and result['error']
        assert timeline.frame()[2]['sample_index'] == 2
    assert timeline.apply(command(1, 'seek', sample_index=0), 0.)['status'] == 'error'
    assert timeline.frame()[2]['sample_index'] == 2


def test_seek_to_last_row_immediately_pauses(timeline):
    timeline.apply(command(1, 'play'), 0.)
    timeline.apply(command(2, 'seek', sample_index=8), .01)
    assert timeline.frame()[2]['sample_index'] == 8 and not timeline.playing


def test_single_initial_row_does_not_spin(recording_source, tmp_path):
    from muse_sdk.trajectory import TrajectoryReader, TrajectoryWriter
    from muse_server.storage import extract_archive
    from muse_server.viewers.replay_process import ReplayTimeline
    from recording_helpers import sample_values
    plan, model = recording_source
    writer = TrajectoryWriter(plan, model)
    try:
        for world in plan.recording.world_ids:
            writer.append(world, (0, 0, 0, 0, 0, 0, 0), sample_values(writer.fields))
        artifact = writer.seal('capacity', 0)
    finally:
        writer.close()
    directory = tmp_path / 'single'
    extract_archive(Path(plan.artifact_directory) / artifact.path, directory)
    reader = TrajectoryReader(directory)
    try:
        timeline = ReplayTimeline(reader)
        timeline.apply(command(1, 'play'), 0.)
        revision = timeline.revision
        assert not timeline.playing and not timeline.advance(100.)
        assert timeline.revision == revision and timeline.frame()[2]['sample_index'] == 0
    finally:
        reader.close()


@pytest.mark.parametrize('payload', [
    {'kind': 'play', 'world_id': 0}, {'kind': 'seek'}, {'kind': 'world'},
    {'kind': 'episode', 'episode': -1}, {'kind': 'step', 'delta': 0},
    {'kind': 'speed', 'speed': float('inf')}, {'kind': 'speed', 'speed': 0.},
    {'kind': 'seek', 'sample_index': True}, {'kind': 'pause', 'delta': 1},
    {'kind': 'step', 'delta': True},
])
def test_command_requires_only_its_own_target(payload):
    from muse_sdk.replay_protocol import ReplayCommand
    with pytest.raises(ValidationError):
        ReplayCommand(command_seq=1, **payload)
