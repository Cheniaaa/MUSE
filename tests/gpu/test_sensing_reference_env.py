"""真实物理/公开缓存/局部重置；记录器仅在测试中观察epoch。"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sensing_package_helpers import SENSING, sensing_catalog
from muse_server.resolver import resolve_experiment


@contextmanager
def sensing_env(catalog, request):
    from muse_builtins.models import registrations, sensing_registrations
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    experiment = report.resolved_experiment
    identity = catalog.save_experiment(experiment)
    plan = catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0', max_decision_steps=200)
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        registered = {'builtin:' + name: cls for name, cls in
                      {**registrations(), **sensing_registrations()}.items()}
        context = RuntimeContext('cuda:0', plan.resource_paths, package.definition.terms,
            model_types={kind: registered[binding.reference] for kind, binding in experiment.model_bindings.items()})
        env = MuseEnv(experiment, context, WorldRng(experiment.root_seed,
            experiment.topology.world_count, 'cuda:0'))
        try:
            rule = package.definition.algorithms[experiment.algorithm_id](experiment.algorithm_config)
            rule.initialize(experiment, context, env.rng)
            rule.set_mode('EVALUATE')
            yield env, rule
        finally:
            env.close()


def step_sensing(env, rule):
    action = rule.act(env.policy_input.observations, env.policy_input.role_masks, env.rng, True)
    output = env.step(action.role_actions)
    rule.process_env_step(output.transition, action.act_aux)
    return output


def powered_request(request):
    """保留示例的role配置，只选择显式不同频1mW阶段。"""
    composition = yaml.safe_load((SENSING / 'training_package/training_package.yaml').read_text())
    request['algorithm_config']['phases'] = [deepcopy(composition['algorithm_config']['phases'][2])]
    request['algorithm_config']['phases'][0]['duration_decisions'] = 500


@pytest.mark.parametrize('k,last', [(7, 14), (3, 18), (1, 19)])
def test_real_completed_measurement_windows(tmp_path, gpu_stream, k, last):
    catalog, request = sensing_catalog(tmp_path, jammer=False)
    request['scenario']['network']['epoch_max_ticks'] = k
    powered_request(request)
    with sensing_env(catalog, request) as (env, rule):
        sample = env.wireless.sensing_outputs['measurement']
        assert not sample.valid.any()
        output = step_sensing(env, rule)
        saved = output.transition.wireless.sensing
        assert saved['measurement'].interval_start_tick.tolist() == [last, last]
        assert saved['measurement'].interval_end_tick.tolist() == [20, 20]
        assert saved['fim'].interval_start_tick.tolist() == [0, 0]
        assert saved['fim'].interval_end_tick.tolist() == [20, 20]
        assert sample.valid.sum(-1).eq(1).all()
        assert saved['fim'].valid.all()


def test_real_subset_reset_preserves_other_world_and_future_rng(tmp_path, gpu_stream):
    catalog, request = sensing_catalog(tmp_path, jammer=False)
    powered_request(request)
    control = None
    for reset in (False, True):
        with sensing_env(catalog, request) as (env, rule):
            step_sensing(env, rule)
            views = env.wireless.sensing_outputs
            pointers = {name: [v.data_ptr() for v in view] for name, view in views.items()}
            before = {name: [v[1].clone() for v in view] for name, view in views.items()}
            if reset:
                mask = torch.tensor([True, False], device='cuda:0')
                env.reset(mask)
                rule.reset_worlds(mask)
                for name, view in views.items():
                    assert not view.data[0].any()
                    assert not view.valid[0].any()
                    assert view.interval_start_tick[0] == view.interval_end_tick[0] == 0
                    for value, wanted in zip(view, before[name]):
                        torch.testing.assert_close(value[1], wanted, rtol=0, atol=0)
                    assert [v.data_ptr() for v in view] == pointers[name]
            step_sensing(env, rule)
            future = {name: [v[1].clone() for v in view] for name, view in views.items()}
            if reset:
                for name, values in future.items():
                    for value, expected in zip(values, control[name]):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                assert views['measurement'].interval_start_tick.tolist() == [14, 34]
                assert views['measurement'].interval_end_tick.tolist() == [20, 40]
            else:
                control = future


def test_real_auto_reset_retains_terminal_samples_cost_and_local_observations(tmp_path, gpu_stream):
    catalog, request = sensing_catalog(tmp_path, jammer=False)
    powered_request(request)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    timeout = next(s for s in report.resolved_experiment.selections if s.time_out)
    request.setdefault('term_config', {}).setdefault('world', {}).setdefault('termination', {})[
        timeout.selection_id] = {'config': {'episode_max_steps': 2}}
    with sensing_env(catalog, request) as (env, rule):
        step_sensing(env, rule)
        mask = torch.tensor([False, True], device='cuda:0')
        env.reset(mask)
        rule.reset_worlds(mask)
        terminal = step_sensing(env, rule).transition
        assert terminal.world_reset.tolist() == [True, False]
        measurement = terminal.wireless.sensing['measurement']
        fim = terminal.wireless.sensing['fim']
        assert measurement.interval_start_tick.tolist() == [34, 14]
        assert measurement.interval_end_tick.tolist() == [40, 20]
        assert measurement.valid.sum(-1).eq(1).all()
        assert fim.interval_start_tick.tolist() == [20, 0]
        assert fim.interval_end_tick.tolist() == [40, 20]
        assert fim.valid.all()
        assert terminal.costs['world']['position_bound_violation'].tolist() == [0., 0.]
        assert not env.wireless.sensing_outputs['measurement'].valid[0].any()
        assert not env.wireless.sensing_outputs['fim'].valid[0].any()
        # 依据冻结关联独立构建每role的期望输入，比较terminal与reset后的公开观察。
        geometry = env.experiment.geometry.layout
        for role, binding in env.experiment.bindings.items():
            relation = next(i for i, sensing in enumerate(geometry.sensing)
                if geometry.points[geometry.paths[sensing.outbound].tx].group_id == binding.group_id
                and geometry.points[geometry.paths[sensing.outbound].tx].slot == binding.role_slot_to_group_slot[0])
            local = next(iter(terminal.pre_reset_next_observations.actors[role].values()))
            expected_range = torch.where(measurement.valid[:, relation], measurement.data[:, relation], 0.).sum(-1)/100.
            expected = torch.stack((expected_range, torch.ones_like(expected_range),
                                    torch.full_like(expected_range, .3)), -1).float()[:, None]
            torch.testing.assert_close(local, expected, rtol=0, atol=0)
            current = next(iter(env.policy_input.observations.actors[role].values()))
            assert not current[0].any()
            torch.testing.assert_close(current[1], expected[1], rtol=0, atol=0)
        following = step_sensing(env, rule).transition
        assert following.wireless.sensing['measurement'].interval_start_tick.tolist() == [14, 34]
        assert following.wireless.sensing['measurement'].interval_end_tick.tolist() == [20, 40]


@pytest.mark.parametrize('model,fields', [
    ('free_space_path_length_measurement', {'measurement'}),
    ('free_space_position_efim_2d', {'fim'}),
    ('free_space_path_length_efim_2d', {'measurement', 'fim'})])
def test_real_powered_capabilities_have_only_selected_outputs(tmp_path, gpu_stream, model, fields):
    catalog, request = sensing_catalog(tmp_path, model=model, jammer=False)
    powered_request(request)
    with sensing_env(catalog, request) as (env, rule):
        output = step_sensing(env, rule).transition.wireless.sensing
        assert set(output) == fields
        for field, view in output.items():
            assert view.valid.any()
            assert view.data.abs().sum() > 0
            assert view.interval_end_tick.tolist() == [20, 20]


def test_real_sensing_only_without_radio_or_links(tmp_path, gpu_stream):
    catalog, request = sensing_catalog(tmp_path, jammer=False)
    powered_request(request)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    reward = next(s for s in report.resolved_experiment.selections if s.term_kind == 'reward')
    request.setdefault('term_config', {}).setdefault('world', {}).setdefault('reward', {})[
        reward.selection_id] = {'enabled': False}
    request['scenario']['radio_geometry']['links'] = []
    request['scenario']['network']['communication_allocations'] = []
    with sensing_env(catalog, request) as (env, rule):
        assert set(env.wireless.models) == {'sensing'}
        assert env.experiment.network.link_count == 0
        output = step_sensing(env, rule).transition.wireless
        assert output.radio is None
        assert output.sensing['measurement'].valid.any()
        assert output.sensing['fim'].data.abs().sum() > 0


def test_real_unused_sensing_binding_does_not_instantiate_model(tmp_path, gpu_stream):
    catalog, request = sensing_catalog(tmp_path)
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    for selection in report.resolved_experiment.selections:
        if selection.term_kind not in ('observation', 'cost'):
            continue
        config = request.setdefault('term_config', {})
        for part in selection.owner_path:
            config = config.setdefault(part, {})
        config.setdefault(selection.term_kind, {})[selection.selection_id] = {'enabled': False}
    request['cost_config'] = {'roles': {}, 'world': {}}
    with sensing_env(catalog, request) as (env, rule):
        assert set(env.wireless.models) == {'radio'}
        assert env.wireless.sensing_outputs == {}
        assert step_sensing(env, rule).transition.wireless.sensing == {}


def blocker_environment(catalog, request, root, center):
    from flight_helpers import archive
    root.mkdir()
    (root / 'environment.yaml').write_text('scene: scene.xml\nrf_geometry: {rf_blocking: [wall]}\n')
    (root / 'scene.xml').write_text(f'''<mujoco><worldbody>
        <geom name="floor" type="plane" size="20 20 .1"/>
        <geom name="wall" type="box" size=".1 .05 .5" pos="{center}"/>
        </worldbody></mujoco>''')
    record = catalog.register_archive('environment', archive(root, root.with_suffix('.zip')))
    request.update(environment_id=record.resource_id, environment_revision=record.revision)


def test_real_echo_and_jammer_occlusion_are_local(tmp_path, gpu_stream):
    catalog, source = sensing_catalog(tmp_path / 'catalog')
    powered_request(source)
    outputs, visibility = {}, {}
    for case, center in [('clear', '20 20 .5'), ('echo', '2 2 .5'), ('jammer', '2 -4 1')]:
        request = deepcopy(source)
        blocker_environment(catalog, request, tmp_path / case, center)
        with sensing_env(catalog, request) as (env, rule):
            outputs[case] = step_sensing(env, rule).transition.wireless.sensing
            layout = env.experiment.geometry.layout
            alpha = next(i for i, entry in enumerate(layout.sensing)
                         if layout.points[layout.paths[entry.outbound].tx].slot == 0)
            entry = layout.sensing[alpha]
            interference = next(x for x in layout.interference if x.kind == 'sensing' and x.consumer == alpha)
            jammer = env.experiment.network.background_emitters[0].endpoint_indices[0]
            jammer_point = env.experiment.network.endpoint_to_point[jammer]
            jammer_path = next(x.path for x in interference.sources if x.tx == jammer_point)
            visibility[case] = env.network.geometry.los[0, [entry.outbound, jammer_path]].tolist()
    assert visibility == {'clear': [True, True], 'echo': [False, True], 'jammer': [True, False]}
    clear, echo, jammer = (outputs[name] for name in ('clear', 'echo', 'jammer'))
    assert clear['measurement'].valid[:, 0, 0].all()
    assert not echo['measurement'].valid[:, 0].any()
    assert echo['fim'].valid.all()
    assert not echo['fim'].data[:, 0].any()
    assert (jammer['fim'].data[:, 0, 0].diagonal(dim1=-2, dim2=-1).sum(-1) >
            clear['fim'].data[:, 0, 0].diagonal(dim1=-2, dim2=-1).sum(-1)).all()
    for case in ('echo', 'jammer'):
        torch.testing.assert_close(outputs[case]['fim'].data[:, 1], clear['fim'].data[:, 1], rtol=1e-11, atol=0)
    (tmp_path / 'blocker-controls.json').write_text(json.dumps(dict(visibility=visibility,
        fim={name: value['fim'].data.tolist() for name, value in outputs.items()}), indent=2))


def test_real_epoch_refinement_matches_scalar_energy_and_same_physics(tmp_path, gpu_stream):
    from sensing_reference_helpers import scalar_sensing_reference
    catalog, source = sensing_catalog(tmp_path / 'catalog', worlds=1)
    powered_request(source)
    blocker_environment(catalog, source, tmp_path / 'moving-occlusion', '2 2 .5')
    emitter = source['scenario']['network']['background_emitters'][0]
    emitter['behavior'] = 'periodic'
    emitter['config'] = dict(on_ticks=137, off_ticks=211, phase_ticks=0, power_w=[.001], channel_id=[7])
    records, totals = {}, {}
    max_scaled_error = 0.
    for k in (7, 3, 1):
        request = deepcopy(source)
        request['scenario']['network']['epoch_max_ticks'] = k
        rows = []
        with sensing_env(catalog, request) as (env, rule):
            original = env.wireless.begin_epoch
            config = env.experiment.model_bindings['sensing'].config
            layout, network = env.experiment.geometry.layout, env.experiment.network

            def record_epoch(sample_tick, *, interval_ticks, radio_requests):
                nonlocal max_scaled_error
                original(sample_tick, interval_ticks=interval_ticks, radio_requests=radio_requests)
                inputs = env.wireless._inputs('sensing')
                plain = {name: {field: value.cpu().tolist() for field, value in getattr(inputs, name).items()}
                         for name in ('geometry', 'endpoint_masks', 'resources', 'allocations', 'simultaneous_transmissions')}
                plain.update(sim_time=inputs.sim_time.tolist(), interval_s=inputs.interval_s.tolist())
                reference = scalar_sensing_reference(config=config, geometry_layout=layout,
                    network_spec=network, inputs=plain)
                expected = torch.tensor(reference['fim'], dtype=torch.float64, device='cuda:0')
                actual = env.wireless.models['sensing'].output.fim
                scale = torch.linalg.matrix_norm(expected).clamp_min(1.)
                error = torch.linalg.matrix_norm(actual - expected) / scale
                assert error.le(1e-11).all()
                max_scaled_error = max(max_scaled_error, error.max().item())
                rows.append(dict(tick=int(sample_tick[0]), length=interval_ticks,
                    positions=plain['geometry']['positions_world'][0], los=plain['geometry']['los'][0],
                    active=plain['simultaneous_transmissions']['active'][0],
                    information_rate=(expected[0] / (interval_ticks * .002)).tolist()))

            env.wireless.begin_epoch = record_epoch
            total = torch.zeros_like(env.wireless.sensing_outputs['fim'].data)
            for _ in range(200):
                output = step_sensing(env, rule)
                assert not output.transition.world_reset.any()
                assert output.transition.wireless.sensing['fim'].valid.all()
                total += output.transition.wireless.sensing['fim'].data
            expected_total = sum(np.asarray(row['information_rate']) * row['length'] * .002 for row in rows)
            delta = np.linalg.norm(total.cpu().numpy()[0] - expected_total, axis=(-2, -1))
            assert np.all(delta <= 1e-11 * np.maximum(np.linalg.norm(expected_total, axis=(-2, -1)), 1.))
            totals[k] = total.cpu().numpy()[0]
            records[k] = rows
    fine = {row['tick']: row for row in records[1]}
    assert len(fine) == 4000
    rates = np.asarray([row['information_rate'] for row in records[1]])
    visibility = np.asarray([row['los'] for row in records[1]])
    active = np.asarray([row['active'] for row in records[1]])
    los_changes = np.flatnonzero(np.any(visibility[1:] != visibility[:-1], axis=1)) + 1
    emitter_changes = np.flatnonzero(np.any(active[1:] != active[:-1], axis=(1, 2))) + 1
    assert len(los_changes) >= 1
    assert len(emitter_changes) >= 2
    transitions = np.union1d(los_changes, emitter_changes)
    errors = {}
    # 两个粗网格使用同一光滑集合，避免排除窗口差异改变收敛比较。
    smooth = np.ones(4000, dtype=bool)
    for tick in transitions:
        smooth[max(0, tick-7):tick+7] = False
    for k in (7, 3):
        held = np.empty_like(rates)
        for row in records[k]:
            tick, length = row['tick'], row['length']
            np.testing.assert_array_equal(row['positions'], fine[tick]['positions'])
            np.testing.assert_array_equal(row['los'], fine[tick]['los'])
            np.testing.assert_array_equal(row['active'], fine[tick]['active'])
            np.testing.assert_allclose(row['information_rate'], fine[tick]['information_rate'], rtol=1e-11, atol=0)
            assert length == min(k, 20-tick % 20)
            held[tick:tick+length] = rates[tick]
        expected = held.sum(0) * .002
        delta = np.linalg.norm(totals[k] - expected, axis=(-2, -1))
        assert np.all(delta <= 1e-11 * np.maximum(np.linalg.norm(expected, axis=(-2, -1)), 1.))
        bound = np.linalg.norm(held-rates, axis=(-2, -1)).sum(0) * .002
        integral_error = np.linalg.norm(totals[k]-totals[1], axis=(-2, -1))
        assert np.all(integral_error <= bound + 1e-9 * np.maximum(bound, 1.))
        ticks = np.asarray([row['tick'] for row in records[k]])
        delays = [int(ticks[ticks >= tick][0]-tick) for tick in transitions]
        assert max(delays) <= k
        errors[k] = dict(integral_error=integral_error.tolist(), absolute_bound=bound.tolist(),
            switch_delays_ticks=delays,
            smooth_l1_error=float(np.linalg.norm((held-rates)[smooth], axis=(-2, -1)).sum() * .002))
    assert errors[3]['smooth_l1_error'] < errors[7]['smooth_l1_error']
    (tmp_path / 'epoch-refinement.json').write_text(json.dumps(dict(max_scaled_error=max_scaled_error,
        totals={str(k): v.tolist() for k, v in totals.items()}, errors=errors,
        los_change_ticks=los_changes.tolist(), emitter_change_ticks=emitter_changes.tolist()), indent=2))
    (tmp_path / 'epoch-trace.json').write_text(json.dumps(records[1]))
