"""真实归档登记和CPU冻结，不以模型候选冒充激活。"""
import pytest

from muse_sdk.experiment import RunPlan, learning_contract
from wireless_package_helpers import wireless_plan


@pytest.mark.parametrize('options,expected', [({}, {'radio', 'sensing'}),
    ({'services': False, 'sensing': False}, {'radio'}),
    ({'services': False, 'sensing_only': True}, {'sensing'}),
    ({'services': False, 'sensing': False, 'background_only': True}, set())])
def test_wireless_package_roundtrip_is_descriptor_only(tmp_path, options, expected):
    _, plan = wireless_plan(tmp_path, **options)
    copy = RunPlan.model_validate_json(plan.model_dump_json())
    assert copy == plan
    assert set(plan.experiment.model_bindings) == expected
    assert plan.experiment.network.epoch_offsets == (0, 4, 8, 10)
    assert [r.channel_id for r in plan.experiment.network.resources] == [7, -3]
    assert len(plan.experiment.network.endpoints) == 4


@pytest.mark.parametrize('direction', ['downlink', 'mobile_to_mobile'])
@pytest.mark.parametrize('worlds', [1, 3])
def test_real_package_preserves_non_uplink_layout(tmp_path, direction, worlds):
    _, plan = wireless_plan(tmp_path, direction=direction, worlds=worlds, sensing=False)
    spec = plan.experiment.network
    allocated = spec.communication_allocations[0]
    assert spec.endpoints[allocated.tx].group_id == ('anchor_group' if direction == 'downlink' else 'guard_group')
    assert spec.endpoints[allocated.rx].group_id == 'scout_group'
    assert not spec.role_controls
    assert plan.experiment.topology.world_count == worlds


def test_real_package_freezes_joint_process_without_independent_draws(tmp_path):
    _, plan = wireless_plan(tmp_path, joint_markov=True, sensing=False)
    emitter, = plan.experiment.network.background_emitters
    assert emitter.endpoint_indices == (0, 1, 3)
    assert emitter.behavior == 'markov'
    assert emitter.config.transition_matrix == [[0., 1.], [1., 0.]]


def test_static_sensing_layout_survives_without_geometry_query(tmp_path):
    from pathlib import Path
    import torch
    from muse_runtime.network.runtime import WirelessRuntime
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    _, plan = wireless_plan(tmp_path, services=False, sensing_only=True, transmission_sensing=True)
    experiment = plan.experiment
    assert 'geometry' not in experiment.activation
    assert experiment.geometry.layout.sensing
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        runtime = WirelessRuntime(experiment.network, experiment.model_bindings,
            {'sensing': package.definition.sensing_models['transmission_sensing']}, batch_size=2, device='cpu',
            rng=WorldRng(7, 2, 'cpu'), geometry_layout=experiment.geometry.layout, geometry_view=None)
        assert runtime.models['sensing']._model.layout['geometry'] is experiment.geometry.layout
        runtime.reset(torch.ones(2, dtype=torch.bool), sample_tick=torch.zeros(2, dtype=torch.int64))
        runtime.begin_epoch(torch.zeros(2, dtype=torch.int64), interval_ticks=4, radio_requests={})
        runtime.end_epoch(torch.full((2,), 4, dtype=torch.int64))
        assert runtime.sensing_outputs['measurement'].data.tolist() == [[[1., 1.]], [[1., 1.]]]
        assert not hasattr(runtime.network, 'geometry')
