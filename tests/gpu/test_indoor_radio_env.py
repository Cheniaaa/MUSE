"""真实物理轨迹上的科学/积分/局部reset验收；观测器只存在于测试。"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from muse_sdk.run_protocol import RunRequest

from indoor_radio_helpers import indoor_radio_catalog
from muse_server.resolver import resolve_experiment


@contextmanager
def indoor_env(catalog, request):
    from muse_builtins.models import registrations
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.package_loader import load_package
    from muse_runtime.rng import WorldRng
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    experiment = report.resolved_experiment
    identity = catalog.save_experiment(experiment, source=request)
    plan = catalog.create_run_plan(RunRequest(experiment_id=identity, mode='EVALUATE', max_decision_steps=200), device='cuda:0')
    with load_package(Path(plan.resource_paths['training_package'])) as package:
        models = {'builtin:' + name: cls for name, cls in registrations().items()}
        context = RuntimeContext('cuda:0', plan.resource_paths, package.definition.terms,
            model_types={'radio': models[experiment.model_bindings['radio'].reference]})
        env = MuseEnv(experiment, context, WorldRng(7, experiment.topology.world_count, 'cuda:0'))
        try:
            rule = package.definition.algorithms['indoor_radio_rule'](experiment.algorithm_config)
            rule.initialize(experiment, context, env.rng)
            rule.set_mode('EVALUATE')
            yield env, rule
        finally:
            env.close()


def step_rule(env, rule):
    action = rule.act(env.policy_input.observations, env.policy_input.role_masks, env.rng, True)
    out = env.step(action.role_actions)
    rule.process_env_step(out.transition, action.act_aux)
    return out


def test_builtin_subset_reset_and_fixed_buffers(tmp_path, gpu_stream):
    catalog, request = indoor_radio_catalog(tmp_path)
    request['algorithm_config']['phases'] = [dict(duration_decisions=100, motion=[0., .1, 0., 0.], power_w=.001, channel_id=7)]
    with indoor_env(catalog, request) as (env, rule):
        model = env.wireless.models['radio']
        assert model.state == ()
        assert not model.output.rate.valid.any()
        assert model.output.link.valid.all()
        assert not model.output.rate.rate_bps.any()
        pointers = [value.data_ptr() for block in model.output for value in block]
        step_rule(env, rule)
        before = [value[1].clone() for block in model.output for value in block]
        mask = torch.tensor([True, False], device='cuda:0')
        env.reset(mask)
        rule.reset_worlds(mask)
        assert rule.step.tolist() == [0, 1]
        assert not model.output.rate.valid[0].any()
        for value, wanted in zip((v for block in model.output for v in block), before):
            torch.testing.assert_close(value[1], wanted, rtol=0, atol=0)
        assert pointers == [value.data_ptr() for block in model.output for value in block]
        assert env.network.radio.sample_tick.tolist() == [0, 14]


def test_exhausted_payload_keeps_interference(tmp_path, gpu_stream):
    catalog, request = indoor_radio_catalog(tmp_path, service=True)
    request['term_config']['world']['termination']['done']['enabled'] = False
    request['scenario']['network']['services'][0]['initial_bits'] = 1.
    request['scenario']['network']['background_emitters'][0]['config']['active'] = [False]
    request['algorithm_config'].update(world_power_scales=[1., 1.], phases=[dict(
        duration_decisions=100, motion=[0., 0., 0., 0.], power_w=.001, channel_id=7)])
    with indoor_env(catalog, request) as (env, rule):
        first = step_rule(env, rule)
        assert not first.transition.world_reset.any()
        assert env.network.services['upload'].remaining_bits.eq(0).all()
        second = step_rule(env, rule)
        assert env.network.services['upload'].served_bits.eq(0).all()
        radio = second.transition.wireless.radio
        assert radio.interference.valid[:, :, 0].all()
        # 唯一干扰源为另一条已完成业务但仍发射的UAV。
        torch.testing.assert_close(radio.interference.interference_power_w[:, 0, 0],
                                   radio.link.received_power_w[:, 1, 0], rtol=1e-5, atol=0)
        assert radio.interference.interference_power_w[:, :, 0].gt(0).all()


def test_indoor_epoch_refinement_same_physics(tmp_path, gpu_stream):
    from radio_reference_helpers import scalar_radio_reference
    catalog, source = indoor_radio_catalog(tmp_path, service=True, worlds=1)
    source['term_config']['world']['termination']['done']['enabled'] = False
    # 5MHz、1mW、d>=1m的有效参考增益上界取1，8s理想rate远低于此有限预算。
    source['scenario']['network']['services'][0]['initial_bits'] = 1e10
    records, totals, hashes = {}, {}, {}
    max_relative = 0.
    for k in (7, 3, 1):
        request = deepcopy(source)
        request['scenario']['network']['epoch_max_ticks'] = k
        rows = []
        with indoor_env(catalog, request) as (env, rule):
            spec, layout = env.experiment.network, env.experiment.geometry.layout
            config = env.experiment.model_bindings['radio'].config
            hashes[str(k)] = hashlib.sha256(env.experiment.model_dump_json().encode()).hexdigest()
            original = env.wireless.begin_epoch

            def record_epoch(sample_tick, *, interval_ticks, radio_requests):
                nonlocal max_relative
                original(sample_tick, interval_ticks=interval_ticks, radio_requests=radio_requests)
                inputs = env.wireless._inputs('radio')
                plain = {name: {field: value.cpu().tolist() for field, value in getattr(inputs, name).items()}
                         for name in ('geometry', 'endpoint_masks', 'resources', 'allocations', 'simultaneous_transmissions')}
                reference = scalar_radio_reference(law='indoor_office_radio', config=config,
                    geometry_layout=layout, network_spec=spec, inputs=plain)
                output = env.wireless.models['radio'].output
                for name, block in output._asdict().items():
                    for field, value in block._asdict().items():
                        expected = torch.tensor(reference[name][field],
                            dtype=torch.bool if field == 'valid' else torch.float64, device=value.device)
                        observed = value if field == 'valid' else value.double()
                        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=0)
                        if field != 'valid':
                            active = expected != 0
                            if active.any():
                                error = ((observed[active]-expected[active]).abs()/expected[active].abs()).max().item()
                                max_relative = max(max_relative, error)
                rows.append(dict(tick=int(sample_tick[0]), length=interval_ticks,
                    positions=plain['geometry']['positions_world'][0],
                    distance=plain['geometry']['distance_m'][0], los=plain['geometry']['los'][0],
                    rate=reference['rate']['rate_bps'][0],
                    signal=reference['link']['received_power_w'][0],
                    interference=reference['interference']['interference_power_w'][0],
                    sinr=reference['interference']['sinr_linear'][0]))

            env.wireless.begin_epoch = record_epoch
            served = torch.zeros((1, 2), dtype=torch.float64, device='cuda:0')
            for _ in range(200):
                transition = step_rule(env, rule).transition
                assert not transition.world_reset.any()
                served += transition.wireless.services['upload'].served_bits
            remaining = env.network.services['upload'].remaining_bits
            torch.testing.assert_close(served + remaining, torch.full_like(served, 1e10), rtol=1e-12, atol=0)
            scalar_total = sum(np.asarray(row['rate']).sum(-1)*row['length']*.002 for row in rows)
            np.testing.assert_allclose(served.cpu().numpy()[0], scalar_total, rtol=1e-5, atol=0)
            totals[str(k)] = served.cpu().tolist()[0]
            records[k] = rows
    fine = {row['tick']: row for row in records[1]}
    assert len(fine) == 4000
    errors = {}
    fine_rates = np.array([row['rate'] for row in records[1]]).sum(-1)
    fine_los = np.array([row['los'] for row in records[1]])
    transitions = np.flatnonzero(np.any(fine_los[1:] != fine_los[:-1], axis=1)) + 1
    assert len(transitions) >= 2
    for k in (7, 3):
        held = np.empty_like(fine_rates)
        for row in records[k]:
            tick, length = row['tick'], row['length']
            np.testing.assert_array_equal(row['positions'], fine[tick]['positions'])
            np.testing.assert_array_equal(row['los'], fine[tick]['los'])
            assert length == min(k, 20 - tick % 20)
            np.testing.assert_allclose(row['rate'], fine[tick]['rate'], rtol=1e-5, atol=0)
            held[tick:tick+length] = np.asarray(fine[tick]['rate']).sum(-1)
        reference_total = held.sum(0)*.002
        np.testing.assert_allclose(totals[str(k)], reference_total, rtol=1e-5, atol=0)
        bound = np.abs(held-fine_rates).sum(0)*.002
        delta = np.abs(reference_total-fine_rates.sum(0)*.002)
        assert np.all(delta <= bound + 1e-7)
        ticks = np.array([row['tick'] for row in records[k]])
        delays = [int(ticks[ticks >= t][0]-t) for t in transitions]
        assert max(delays) <= k
        smooth = np.ones(4000, dtype=bool)
        for t in transitions:
            smooth[max(0, t-k):t+k] = False
        errors[str(k)] = dict(integral_error_bits=delta.tolist(), absolute_error_bound_bits=bound.tolist(),
            switch_delays_ticks=delays, smooth_l1_error_bits=(np.abs(held-fine_rates)[smooth].sum(0)*.002).tolist())
    positions = np.array([row['positions'] for row in records[1]])
    distances = np.array([row['distance'] for row in records[1]])
    report = dict(max_relative_error=max_relative, configuration_sha256=hashes, totals_bits=totals,
        errors=errors, height_range_m=[float(positions[:,:,2].min()), float(positions[:,:,2].max())],
        distance_range_m=[float(distances.min()), float(distances.max())], transition_ticks=transitions.tolist())
    (tmp_path / 'epoch-refinement.json').write_text(json.dumps(report, indent=2))
    (tmp_path / 'epoch-trace.json').write_text(json.dumps(records[1]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    seconds = np.arange(4000)*.002
    for ax, field in zip(axes, ('signal', 'interference', 'sinr', 'rate')):
        values = np.array([row[field] for row in records[1]])
        ax.plot(seconds, values[:, 0].sum(-1), label='link0')
        ax.plot(seconds, values[:, 1].sum(-1), label='link1')
        ax.set_ylabel(field)
        ax.legend()
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(tmp_path / 'epoch-curves.png')
    plt.close(fig)
