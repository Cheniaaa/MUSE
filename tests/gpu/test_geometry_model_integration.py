"""真实资源→Scene→几何→模型贯通；合成输出不是科学传播或测量模型。"""
from copy import deepcopy
import math
from types import MappingProxyType

import pytest
import torch
import yaml

from flight_helpers import add_fixed_anchors, archive
from muse_sdk.config import DataModel, export_config_schema
from muse_sdk.descriptor import RadioDescriptor, SensingDescriptor
from muse_sdk.geometry import geometry_input_fields
from muse_sdk.models import (ExecutionContext, ModelResetMask, ModelResult, RadioEvaluationResult,
    RadioInputs, RadioModel, RateMetrics, SensingInputs, SensingModel, SensingResult)
from muse_server.resolver import resolve_experiment


@pytest.fixture(scope='module')
def geometry_catalog(pipeline, tmp_path_factory):
    catalog, source = pipeline
    root = tmp_path_factory.mktemp('geometry-model')
    environment = root / 'environment'
    environment.mkdir()
    (environment / 'environment.yaml').write_text(yaml.safe_dump(dict(
        scene='scene.xml', rf_geometry=dict(rf_blocking=['wall']))))
    # 闭合薄壁mesh的解析范围：x=[1.9,2.1]、y=[-1,1]、z=[0,2]。
    # 透明且无collision；RF语义来自资源声明，不来自光学或碰撞属性。
    (environment / 'scene.xml').write_text('''<mujoco>
      <asset><mesh name="wall_mesh"
        vertex="-.1 -1 -1  .1 -1 -1  .1 1 -1  -.1 1 -1
                -.1 -1 1   .1 -1 1   .1 1 1   -.1 1 1"
        face="0 2 1  0 3 2  4 5 6  4 6 7  0 1 5  0 5 4
              1 2 6  1 6 5  2 3 7  2 7 6  3 0 4  3 4 7"/></asset>
      <worldbody>
        <geom name="floor" type="plane" size="20 20 .1"/>
        <geom name="wall" type="mesh" mesh="wall_mesh" pos="2 0 1"
              rgba="1 1 1 0" contype="0" conaffinity="0"/>
      </worldbody></mujoco>''')
    record = catalog.register_archive('environment', archive(environment, root / 'environment.zip'))
    request = deepcopy(source)
    request.update(environment_id=record.resource_id, environment_revision=record.revision)
    add_fixed_anchors(request)
    # 保持scout/guard已有motion roles；station没有motion role。
    anchor = request['scenario']['entity_groups'][-1]
    anchor['count'] = 1
    request['scenario']['placements']['anchor_start'] = [dict(
        position=[4., 0., 1.], orientation_wxyz=[1., 0., 0., 0.])]
    request['scenario']['placements']['guard_start'][0]['position'] = [0., 3., 1.]
    request['scenario']['placements']['guard_start'][1]['position'] = [0., -3., 1.]
    request['scenario']['radio_geometry'] = dict(
        endpoints={
            'tx': dict(group_id='scout_group', slots=[0], device_id='ideal_point_radio'),
            'rx': dict(group_id='anchor_group', slots=[0], device_id='ideal_point_radio'),
            'jammer': dict(group_id='guard_group', slots=[1], device_id='ideal_point_radio')},
        links=[dict(link_id='uplink', tx='tx', rx='rx')],
        sensing=[dict(sensing_id='echo', tx='tx', rx='rx',
                      target=dict(group_id='guard_group', slots=[0]))],
        endpoint_epsilon_m=1e-4)
    return catalog, request


def freeze_plan(catalog, source, batch_size, *, geometry=True):
    request = deepcopy(source)
    request['topology']['world_count'] = batch_size
    if not geometry:
        del request['scenario']['radio_geometry']
    report = resolve_experiment(request, catalog.snapshot())
    assert report.valid, report.errors
    identity = catalog.save_experiment(report.resolved_experiment, source=request)
    return catalog.create_run_plan(identity, mode='EVALUATE', device='cuda:0')


def assert_layout(layout):
    # 独立手算oracle：直达先出现、感知双段随后、候选干扰最后。
    assert [(p.group_id, p.slot, p.device_id) for p in layout.points] == [
        ('scout_group', 0, 'ideal_point_radio'), ('anchor_group', 0, 'ideal_point_radio'),
        ('guard_group', 1, 'ideal_point_radio'), ('guard_group', 0, None)]
    assert [(p.path_id, p.tx, p.rx) for p in layout.paths] == [
        ('p0', 0, 1), ('p1', 0, 3), ('p2', 3, 1), ('p3', 2, 1)]
    assert [(d.link_id, d.path) for d in layout.desired] == [('uplink', 0)]
    assert [(s.sensing_id, s.outbound, s.inbound, s.direct) for s in layout.sensing] == [
        ('echo', 1, 2, 0)]
    assert [(c.kind, c.consumer, [(s.tx, s.path) for s in c.sources])
            for c in layout.interference] == [('desired', 0, [(2, 3)]), ('sensing', 0, [(2, 3)])]


def test_frozen_layout_identity_is_independent_of_world_count(geometry_catalog):
    catalog, request = geometry_catalog
    plans = [freeze_plan(catalog, request, batch) for batch in (1, 2)]
    for plan in plans:
        assert_layout(plan.experiment.geometry.layout)
        assert plan.experiment.geometry.blockers == ('wall',)
        assert plan.experiment.geometry.endpoint_epsilon_m == 1e-4
        assert not plan.experiment.model_bindings and 'radio' not in plan.experiment.activation
    assert plans[0].experiment.geometry == plans[1].experiment.geometry
    # 保留M3.3既有正式Run边界；组件验收不暗中激活model binding。
    bound_request = deepcopy(request)
    bound_request['model_config'] = {'synthetic_geometry_radio': {}}
    report = resolve_experiment(bound_request, catalog.snapshot())
    assert not report.valid
    assert any(error['field'] == 'model_config' for error in report.errors), report.errors


def check_model_geometry(model, inputs):
    """两个合成消费者都检查六字段来自同一真实快照。"""
    geometry = inputs.geometry
    layout = model.layout['geometry']
    positions = geometry['positions_world']
    lengths = torch.stack([torch.linalg.vector_norm(positions[:, path.tx] - positions[:, path.rx], dim=-1)
                           for path in layout.paths], dim=1)
    torch.testing.assert_close(geometry['distance_m'], torch.where(geometry['valid'], lengths, 0.))
    assert not (geometry['los'] & ~geometry['valid']).any()
    torch.testing.assert_close(geometry['sample_time'], inputs.sim_time)
    torch.testing.assert_close(geometry['sample_time'],
        geometry['sample_tick'].to(torch.float64) * model.config['physics_dt'])


def create_geometry_models(view, physics_dt, device):
    from muse_runtime.model_execution import ModelInstance
    from muse_runtime.rng import WorldRng

    batch_size = view.distance_m.shape[0]
    layout = view.layout
    fields = [field for phase in ('reset', 'post_physics')
              for field in geometry_input_fields(layout, batch_size=batch_size, phase=phase)]
    class GeometryModelConfig(DataModel):
        physics_dt: float

    common = dict(config_schema=export_config_schema(GeometryModelConfig), input_fields=fields,
        state_scope='world', state_schema={},
        diagnostics_schema={}, assumptions={'synthetic': True, 'scientific_propagation': False})
    radio_meta = dict(common, model_id='synthetic_geometry_radio', output_spec={'output_blocks': ['rate']})
    sensing_meta = dict(common, model_id='synthetic_geometry_sensing', output_spec={'measurement': dict(
        shape=['B', 'M', 'K'], dtype='float32', units='m', frame='none', rate='radio_epoch',
        time_semantics='snapshot')})

    class SyntheticRadio(RadioModel):
        @classmethod
        def describe(cls):
            return {**radio_meta, 'config_schema': GeometryModelConfig}

        def result(self, inputs, reset):
            check_model_geometry(self, inputs)
            indices = [item.path for item in self.layout['geometry'].desired]
            distance = inputs.geometry['distance_m'][:, indices, None]
            valid = inputs.geometry['valid'][:, indices, None]
            los = inputs.geometry['los'][:, indices, None]
            # 10是测试变换系数，不是传播增益；有效NLoS不会被替换为0。
            rate = torch.zeros_like(distance) if reset else distance * 10 + los.to(torch.float32)
            return ModelResult(RadioEvaluationResult(None, None,
                RateMetrics(rate, torch.zeros_like(valid) if reset else valid.clone())), (), {})

        def reset_state(self, state, reset_mask, inputs, rng):
            return self.result(inputs, True)

        def evaluate(self, inputs, state, rng):
            return self.result(inputs, False)

    class SyntheticSensing(SensingModel):
        @classmethod
        def describe(cls):
            return {**sensing_meta, 'config_schema': GeometryModelConfig}

        def result(self, inputs):
            check_model_geometry(self, inputs)
            indices = [[item.outbound, item.inbound] for item in self.layout['geometry'].sensing]
            # 两个通道分别保留tx→target/target→rx距离，不把direct当回波。
            return ModelResult(SensingResult(inputs.geometry['distance_m'][:, indices].clone(), None,
                {'measurement': inputs.geometry['valid'][:, indices].clone()},
                {'measurement': 'snapshot'}), (), {})

        def reset_state(self, state, reset_mask, inputs, rng):
            return self.result(inputs)

        def evaluate(self, inputs, state, rng):
            return self.result(inputs)

    axes = MappingProxyType(dict(B=batch_size, L=len(layout.desired), F=1, M=len(layout.sensing), K=2))
    zero = torch.zeros(batch_size, dtype=torch.float64, device=device)
    radio_inputs = RadioInputs(view.mapping, {}, {}, {}, {}, view.sample_time, zero)
    sensing_inputs = SensingInputs(view.mapping, {}, {}, {}, {}, None, view.sample_time, zero)
    kwargs = dict(config=MappingProxyType({'physics_dt': physics_dt}), context=ExecutionContext(device),
        available_inputs=fields, rng=WorldRng(7, batch_size, device))
    radio = ModelInstance(SyntheticRadio, descriptor=RadioDescriptor.model_validate(radio_meta),
        layout=MappingProxyType(dict(axes=axes, geometry=layout, instance_id='radio')),
        bound_slots=torch.ones(batch_size, len(layout.desired), 1,
            dtype=torch.bool, device=device), **kwargs)
    sensing = ModelInstance(SyntheticSensing, descriptor=SensingDescriptor.model_validate(sensing_meta),
        layout=MappingProxyType(dict(axes=axes, geometry=layout, instance_id='sensing')),
        **kwargs)
    return radio, sensing, radio_inputs, sensing_inputs


@pytest.mark.parametrize('batch_size', [1, 2])
def test_real_scene_mesh_snapshot_is_consumed_by_model_instances(geometry_catalog, gpu_stream, batch_size):
    from mjlab.scene import Scene
    from mjlab.sim import Simulation
    from muse_runtime.network.geometry import GeometryQuery
    from muse_runtime.scene import assemble_scene

    catalog, request = geometry_catalog
    plan = freeze_plan(catalog, request, batch_size)
    experiment = plan.experiment
    resolved = experiment.geometry
    assert_layout(resolved.layout)
    assert all(binding.group_id != 'anchor_group' for binding in experiment.bindings.values())
    scene_cfg, sim_cfg = assemble_scene(experiment, plan.resource_paths)
    scene = Scene(scene_cfg, device=plan.device)
    sim = Simulation(batch_size, sim_cfg, spec=scene.spec, device=plan.device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.set_sensor_context(scene.sensor_context)
    sim.reset()
    scene.reset()
    scout = scene['scout_group_0']
    state = scout.data.default_root_state.clone()
    if batch_size == 2:
        state[1, :3] = torch.tensor([0., 4., 1.], device=plan.device)
    scout.write_root_state_to_sim(state)
    scene.write_data_to_sim()
    sim.forward()
    dt = experiment.scenario.scene.physics_dt
    query = GeometryQuery(resolved, sim, physics_dt=dt, device=plan.device)
    assert query.view.layout == resolved.layout
    assert not query.view.valid.any()
    worlds = torch.ones(batch_size, dtype=torch.bool, device=plan.device)
    ticks = torch.zeros(batch_size, dtype=torch.int64, device=plan.device)
    present = torch.ones(batch_size, len(resolved.layout.points), dtype=torch.bool, device=plan.device)
    physical = {name: getattr(sim.data, name)[:].clone() for name in ('time', 'qpos', 'qvel', 'xpos')}
    view = query.refresh(worlds, sample_tick=ticks, present=present)
    for name, before in physical.items():
        torch.testing.assert_close(getattr(sim.data, name)[:], before, rtol=0, atol=0)
    expected_positions = [[[0., 0., 1.], [4., 0., 1.], [0., -3., 1.], [0., 3., 1.]]]
    expected_distances = [[4., 3., 5., 5.]]
    expected_los = [[False, True, True, True]]
    if batch_size == 2:
        expected_positions.append([[0., 4., 1.], [4., 0., 1.], [0., -3., 1.], [0., 3., 1.]])
        expected_distances.append([math.sqrt(32), 1., 5., 5.])
        expected_los.append([True, True, True, True])
    torch.testing.assert_close(view.positions_world, torch.tensor(expected_positions, device=plan.device))
    torch.testing.assert_close(view.distance_m, torch.tensor(expected_distances, device=plan.device))
    assert view.valid.all() and view.los.tolist() == expected_los
    assert all(view.mapping[name] is getattr(view, name) for name in view.mapping)
    addresses = {name: tensor.data_ptr() for name, tensor in view.mapping.items()}
    radio, sensing, radio_inputs, sensing_inputs = create_geometry_models(view, dt, plan.device)
    radio.reset(ModelResetMask(worlds), radio_inputs)
    sensing.reset(ModelResetMask(worlds), sensing_inputs)
    assert not radio.output.rate.valid.any() and radio.output.rate.rate_bps.count_nonzero() == 0
    torch.testing.assert_close(sensing.output.measurement,
        torch.tensor([[[3., 5.]], [[1., 5.]]][:batch_size], device=plan.device))

    # 宿主移动并forward；query只读已经更新的物理状态，没有第二时钟。
    state[0, :3] = torch.tensor([0., 4., 1.], device=plan.device)
    if batch_size == 2:
        state[1, :3] = torch.tensor([0., 0., 1.], device=plan.device)
    scout.write_root_state_to_sim(state)
    sim.data.time[:] = torch.tensor([.02, .04][:batch_size], device=plan.device)
    sim.forward()
    time_before = sim.data.time[:].clone()
    ticks[:] = torch.tensor([10, 20][:batch_size], device=plan.device)
    assert query.refresh(worlds, sample_tick=ticks, present=present) is view
    torch.testing.assert_close(sim.data.time[:], time_before, rtol=0, atol=0)
    torch.testing.assert_close(view.sample_time, torch.tensor([.02, .04][:batch_size],
        dtype=torch.float64, device=plan.device))
    interval = torch.full((batch_size,), .01, dtype=torch.float64, device=plan.device)
    radio.evaluate(radio_inputs._replace(interval_s=interval))
    sensing.evaluate(sensing_inputs._replace(interval_s=interval))
    torch.testing.assert_close(radio.output.rate.rate_bps,
        torch.tensor([[[math.sqrt(32) * 10 + 1]], [[40.]]][:batch_size], device=plan.device))
    assert radio.output.rate.valid.all()
    torch.testing.assert_close(sensing.output.measurement,
        torch.tensor([[[1., 5.]], [[3., 5.]]][:batch_size], device=plan.device))

    saved_geometry = {name: tensor.clone() for name, tensor in view.mapping.items()}
    saved_rate = radio.output.rate.rate_bps.clone()
    saved_measurement = sensing.output.measurement.clone()
    mask = torch.arange(batch_size, device=plan.device) == 0
    state[0, :3] = torch.tensor([0., 0., 1.], device=plan.device)
    scout.write_root_state_to_sim(state)
    sim.data.time[0] = 0
    sim.forward()
    ticks[0] = 0
    present[0, 3] = False  # 缺失目标是invalid，不是有效NLoS。
    query.refresh(mask, sample_tick=ticks, present=present)
    radio.reset(ModelResetMask(mask), radio_inputs)
    sensing.reset(ModelResetMask(mask), sensing_inputs)
    assert not view.valid[0, 1:3].any() and not view.los[0, 1:3].any()
    assert view.valid[0, 0] and not view.los[0, 0]
    assert not sensing.output.field_valid['measurement'][0].any()
    assert sensing.output.measurement[0].count_nonzero() == 0
    assert not radio.output.rate.valid[0].any() and radio.output.rate.rate_bps[0].count_nonzero() == 0
    for name, tensor in view.mapping.items():
        assert tensor.data_ptr() == addresses[name]
        torch.testing.assert_close(tensor[1:], saved_geometry[name][1:], rtol=0, atol=0)
    torch.testing.assert_close(radio.output.rate.rate_bps[1:], saved_rate[1:], rtol=0, atol=0)
    torch.testing.assert_close(sensing.output.measurement[1:], saved_measurement[1:], rtol=0, atol=0)


def test_ordinary_flight_does_not_construct_rf_query_or_buffers(geometry_catalog, gpu_stream, monkeypatch):
    from muse_runtime.env import MuseEnv, RuntimeContext
    from muse_runtime.network.geometry import GeometryQuery
    from muse_runtime.rng import WorldRng
    from test_imu_sensors import commands

    catalog, request = geometry_catalog
    plan = freeze_plan(catalog, request, 2, geometry=False)
    assert plan.experiment.geometry is None

    def unexpected_query(*args, **kwargs):
        pytest.fail('普通flight不应构造拥有RF缓存的GeometryQuery')

    # 只拦截禁止发生的RF分配入口；真实MuseEnv/Scene/Simulation仍完整运行。
    monkeypatch.setattr(GeometryQuery, '__init__', unexpected_query)
    env = MuseEnv(plan.experiment, RuntimeContext(plan.device, plan.resource_paths, {}),
                  WorldRng(plan.experiment.root_seed, 2, plan.device))
    try:
        env.step(commands(env))
        assert not any(isinstance(value, GeometryQuery) for value in vars(env).values())
        assert env.sensor_outputs == {}
        observations = env._observe()
        assert set(observations.actors) == {'scout', 'guard'}
    finally:
        env.close()
