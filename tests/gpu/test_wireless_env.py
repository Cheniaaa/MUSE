"""真实Env的无线装配、终态快照和长期stream生命周期。"""
import torch
import warp as wp
import pytest
from pathlib import Path

from test_flight_env import actions, flight_env, flight_plan
from wireless_package_helpers import wireless_plan


@pytest.fixture
def wireless_env(tmp_path, gpu_stream):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    _, plan = wireless_plan(tmp_path)
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        definition = package.definition
        context = RuntimeContext('cuda:0', plan.resource_paths, definition.terms, model_types={
            'radio': definition.radio_models['synthetic_radio'], 'sensing': definition.sensing_models['synthetic_sensing']})
        env = MuseEnv(plan.experiment, context, WorldRng(7, 2, 'cuda:0'))
        rule = definition.algorithms['wireless_rule'](plan.experiment.algorithm_config)
        rule.initialize(plan.experiment, context, env.rng)
        yield env, rule.actions
        env.close()


def test_env_keeps_worker_stream_registered_after_step_and_reset(flight_env, gpu_stream):
    env = flight_env
    env.step(actions(env))
    env.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    # 同句柄临时wrapper析构会unregister，后续真实capture会直接失败。
    storage = wp.zeros(4, dtype=float, device=env.device)
    with wp.ScopedCapture(stream=wp.get_stream(env.device)) as capture:
        storage.zero_()
    wp.capture_launch(capture.graph)
    gpu_stream.synchronize()
    assert not wp.to_torch(storage).count_nonzero()


def test_wireless_terminal_snapshot_survives_subset_reset(wireless_env):
    env, requests = wireless_env
    radio = env.wireless.models['radio']._model
    assert len(radio.resets) == 1
    assert not radio.evaluations
    services = env.network.services['upload']
    pointers = [v.data_ptr() for v in services]
    out1 = env.step(requests)
    out2 = env.step(requests)
    assert out2.transition.world_reset.tolist() == [True, False]
    terminal = out2.transition.wireless
    assert terminal.services['upload'].remaining_bits[0].item() == 0
    assert terminal.services['upload'].interval_end_tick.tolist() == [20, 20]
    assert terminal.sensing['fim'].interval_start_tick.tolist() == [10, 10]
    assert terminal.sensing['fim'].interval_end_tick.tolist() == [20, 20]
    assert terminal.sensing['measurement'].interval_start_tick.tolist() == [18, 18]
    assert services.remaining_bits[0].item() == 1.
    assert services.served_bits[0].item() == 0
    assert env.episode_length_buf.tolist() == [0, 2]
    before = terminal.services['upload'].served_bits.clone()
    env.step(requests)
    assert radio.evaluations[-3][0].tolist() == [0, 20]
    torch.testing.assert_close(before, terminal.services['upload'].served_bits)
    assert [v.data_ptr() for v in services] == pointers
    resets = len(radio.resets)
    env.reset(torch.zeros(2, dtype=torch.bool, device=env.device))
    assert len(radio.resets) == resets


@pytest.mark.parametrize('kind', ['background', 'sensing', 'static_sensing'])
def test_real_optional_wireless_assemblies(tmp_path, gpu_stream, kind):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    _, plan = wireless_plan(tmp_path, services=False, background_only=kind == 'background',
                            sensing_only=kind != 'background', sensing=kind != 'background',
                            transmission_sensing=kind == 'static_sensing')
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        types = {name: getattr(package.definition, name + '_models')[binding.reference]
                 for name, binding in plan.experiment.model_bindings.items()}
        env = MuseEnv(plan.experiment, RuntimeContext('cuda:0', plan.resource_paths, package.definition.terms,
            model_types=types), WorldRng(4, 2, 'cuda:0'))
        try:
            assert ('radio' not in env.wireless.models)
            assert (env.geometry_query is None) == (kind != 'sensing')
            out = env.step(actions(env))
            assert out.transition.wireless.radio is None
            assert out.transition.wireless.services == {}
            assert env.network.simultaneous_transmissions.active.any()
            if kind == 'static_sensing':
                assert env.sensing_outputs['measurement'].data.eq(1.).all()
        finally:
            env.close()
