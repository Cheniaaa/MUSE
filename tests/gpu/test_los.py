"""真实CUDA有限线段与固定缓存；解析oracle与退化特征分开。"""
from types import SimpleNamespace
import math
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp

from muse_sdk.geometry import GeometryPoint, PropagationPath, PropagationPathLayout, ResolvedGeometry


def make_query(body, *, assets='', start=(0., 0., 0.), end=(10., 0., 0.),
               blockers=('wall',), mount=(0., 0., 0.), endpoint_geoms=('', '')):
    from muse_runtime.network.geometry import GeometryQuery
    assert torch.cuda.is_available(), 'RF acceptance requires actual CUDA'
    wp.init()
    cpu = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler fusestatic="false"/><asset>{assets}</asset>
      <worldbody>{body}
        <body name="g_0/root" pos="{start[0]} {start[1]} {start[2]}"><freejoint/><geom size=".01" mass="1"/>{endpoint_geoms[0]}</body>
        <body name="g_1/root" pos="{end[0]} {end[1]} {end[2]}"><freejoint/><geom size=".01" mass="1"/>{endpoint_geoms[1]}</body>
      </worldbody></mujoco>''')
    stream = wp.get_stream('cuda:0')
    if stream.cuda_stream != torch.cuda.current_stream().cuda_stream:
        stream = wp.stream_from_torch(torch.cuda.current_stream())
    with wp.ScopedDevice('cuda:0'), wp.ScopedStream(stream):
        model, data = mjw.put_model(cpu), mjw.make_data(cpu, nworld=2)
        mjw.forward(model, data)
    # 真实MjModel/Warp arrays，窄测试不创建显示或训练运行时。
    sim = SimpleNamespace(mj_model=cpu, wp_model=model, wp_data=data, stream=stream)
    points = tuple(GeometryPoint(group_id='g', slot=i, device_id='radio', body='root',
        position_m=mount if i == 0 else (0., 0., 0.), assumption='test mount') for i in range(2))
    layout = PropagationPathLayout(points=points, paths=(PropagationPath(path_id='p0', tx=0, rx=1),),
        desired=(), sensing=(), interference=())
    resolved = ResolvedGeometry(layout=layout, endpoint_epsilon_m=1e-4, blockers=tuple(blockers))
    return GeometryQuery(resolved, sim, physics_dt=.002, device='cuda:0'), sim


def refresh(query, mask=None, tick=(10, 20), present=None):
    return query.refresh(torch.tensor([True, True] if mask is None else mask, device='cuda:0'),
        sample_tick=torch.tensor(tick, dtype=torch.int64, device='cuda:0'),
        present=torch.ones((2, 2), dtype=torch.bool, device='cuda:0') if present is None else present)


@pytest.mark.parametrize('body,blockers,start,end,valid,los', [
    ('<geom name="environment/wall" type="box" pos="5 0 0" size="1 1 1" rgba="1 1 1 0" contype="0" conaffinity="0"/>', ('wall',), (0.,0.,0.), (10.,0.,0.), True, False),
    ('<geom name="environment/wall" type="box" pos="5 0 0" size="1 1 1"/>', (), (0.,0.,0.), (10.,0.,0.), True, True),
    ('<geom name="environment/wall" type="box" pos="12 0 0" size="1 1 1"/>', ('wall',), (0.,0.,0.), (10.,0.,0.), True, True),
    ('<geom name="environment/wall" type="box" pos="5 0 0" euler="0 0 45" size="1 1 1"/>', ('wall',), (0.,0.,0.), (10.,0.,0.), True, False),
    ('<geom name="environment/wall" type="box" pos="5 0 0" size="1 1 1"/>', ('wall',), (0.,0.,0.), (4.00005,0.,0.), True, True),
    ('<geom name="environment/wall" type="box" pos="5 0 0" size="1 1 1"/>', ('wall',), (0.,0.,0.), (4.0002,0.,0.), True, False),
    ('', (), (0.,0.,0.), (0.,0.,0.), False, False),
    ('', (), (0.,0.,0.), (.00019,0.,0.), False, False),
    ('', (), (0.,0.,0.), (.00021,0.,0.), True, True),
])
def test_analytic_segments(body, blockers, start, end, valid, los):
    query, sim = make_query(body, blockers=blockers, start=start, end=end)
    view = refresh(query)
    assert view.valid.tolist() == [[valid], [valid]]
    assert view.los.tolist() == [[los], [los]]
    expected_distance = math.dist(start, end) if valid else 0.
    torch.testing.assert_close(view.distance_m, torch.full((2, 1), expected_distance, device='cuda:0'))


def test_mesh_scale_rotation_and_backface_exit():
    assets = '<mesh name="tetra" scale="2 3 4" vertex="0 -1 -1 0 1 -1 0 0 1 1 0 0" face="0 2 1 0 1 3 1 2 3 2 0 3"/>'
    # 绕Z90度后，局部x内部点.1→世界y=.1；沿世界+y穿出mesh背面。
    query, _ = make_query('<geom name="environment/wall" type="mesh" mesh="tetra" pos="5 0 0" euler="0 0 90"/>',
        assets=assets, start=(5., .1, 0.), end=(5., 4., 0.))
    assert not refresh(query).los.any()
    assert len(query.view.layout.paths) == 1


@pytest.mark.parametrize('third_party', [False, True])
def test_body_filter_does_not_eliminate_third_party(third_party):
    # 直接从原生MJCF绑定两个endpoint所属geom；仅此kernel用例越过静态环境resolver限制。
    attached = ('<geom name="environment/tx" type="box" pos="1 0 0" size=".5 1 1"/>',
                '<geom name="environment/rx" type="box" pos="-1 0 0" size=".5 1 1"/>')
    body = '<geom name="environment/wall" type="box" pos="5 0 0" size=".5 1 1"/>'
    query, _ = make_query(body, endpoint_geoms=attached,
        blockers=('tx', 'rx', 'wall') if third_party else ('tx', 'rx'))
    assert refresh(query).los.tolist() == [[not third_party], [not third_party]]


def test_fixed_cache_subset_reset_motion_aliases_and_atomic_failure():
    query, sim = make_query('<geom name="environment/wall" type="box" pos="5 0 0" size=".5 .5 .5"/>', mount=(1.,0.,0.))
    assert not query.view.valid.any()
    view = refresh(query)
    pointers = {k: v.data_ptr() for k, v in view.mapping.items()}
    snapshot = {k: v.clone() for k, v in view.mapping.items()}
    qpos = wp.to_torch(sim.wp_data.qpos)
    qpos[1, :3] = torch.tensor([0., 3., 0.], device='cuda:0')
    qpos[1, 3:7] = torch.tensor([2**-.5, 0., 0., 2**-.5], device='cuda:0')
    with wp.ScopedStream(sim.stream):
        mjw.forward(sim.wp_model, sim.wp_data)
    refresh(query)
    torch.testing.assert_close(view.positions_world[1, 0], torch.tensor([0., 4., 0.], device='cuda:0'), atol=1e-6, rtol=0)
    assert view.los.tolist() == [[False], [True]]
    assert view.sample_time.tolist() == [.02, .04]
    other = {k: v[1].clone() for k, v in view.mapping.items()}
    refresh(query, mask=[True, False], tick=(0, 99))
    for k, value in view.mapping.items():
        torch.testing.assert_close(value[1], other[k])
        assert value.data_ptr() == pointers[k]
    assert snapshot['sample_tick'].tolist() == [10, 20]
    assert view.sample_tick.tolist() == [0, 20]
    present = torch.zeros((2, 2), dtype=torch.bool, device='cuda:0')
    # valid在提交过程中变false不能影响后续时间字段的world选择。
    query.refresh(view.valid[:, 0], sample_tick=torch.tensor([30, 40], device='cuda:0'), present=present)
    assert view.sample_tick.tolist() == [30, 40] and not view.valid.any()
    before = {k: v.clone() for k, v in view.mapping.items()}
    wp.to_torch(sim.wp_data.xpos)[1, sim.mj_model.body('g_0/root').id, 0] = float('nan')
    refresh(query, mask=[False, False])
    with pytest.raises(RuntimeError, match='finite'):
        refresh(query)
    for k, value in view.mapping.items():
        torch.testing.assert_close(value, before[k])


@pytest.mark.parametrize('start,end,los', [((4.5,1.,0.),(5.5,1.,0.),True),
    ((0.,1.,1.),(10.,1.,1.),False), ((4.,3.,1.),(8.,-1.,1.),True),
    ((4.5,0.,0.),(5.5,0.,0.),True)])
def test_fixed_backend_degenerate_characterization(start, end, los):
    """记录固定后端的具体样本，非严格集合相交oracle。"""
    query, _ = make_query('<geom name="environment/wall" type="box" pos="5 0 0" size="1 1 1"/>', start=start, end=end)
    assert refresh(query).los.tolist() == [[los], [los]]


def test_shared_and_per_world_box_sizes_on_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        query, sim = make_query('<geom name="environment/wall" type="box" pos="5 2 0" size=".5 .5 .5"/>')
        assert refresh(query).los.all()
        size = np.tile(sim.mj_model.geom_size, (2, 1, 1))
        size[1, sim.mj_model.geom('environment/wall').id, 1] = 3.
        sim.wp_model.geom_size = wp.array(size, dtype=wp.vec3, device='cuda:0')
        assert refresh(query).los.tolist() == [[True], [False]]
        for _ in range(5):
            refresh(query)
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            refresh(query)
        stream.synchronize()
        print(f'RF diagnostic B=2 P=1 blockers=1 triangles=0 mean_ms={(time.perf_counter()-start)*50:.3f}')


def test_query_preserves_host_registered_stream_for_graph_capture(gpu_stream):
    # 外层宿主已有Warp stream；query不能创建同句柄短命包装后将它unregister。
    query, _ = make_query('', blockers=())
    refresh(query)
    with wp.ScopedCapture() as capture:
        pass
    assert capture.graph is not None


@pytest.mark.parametrize('start,end', [((5.,-.1,-.5),(5.,.1,-.5)), ((0.,1.,-1.),(10.,1.,-1.))])
def test_fixed_backend_mesh_degenerate_characterization(start, end):
    assets = '<mesh name="tetra" vertex="0 -1 -1 0 1 -1 0 0 1 1 0 0" face="0 2 1 0 1 3 1 2 3 2 0 3"/>'
    query, _ = make_query('<geom name="environment/wall" type="mesh" mesh="tetra" pos="5 0 0"/>', assets=assets, start=start, end=end)
    assert refresh(query).los.all()


def test_nonfinite_distance_does_not_commit_and_unselected_world_is_untouched():
    query, sim = make_query('', blockers=())
    refresh(query)
    before = {k: v.clone() for k, v in query.view.mapping.items()}
    wp.to_torch(sim.wp_data.xpos)[1, sim.mj_model.body('g_1/root').id, 0] = 1e30
    with pytest.raises(RuntimeError, match='nonfinite'):
        refresh(query, tick=(100, 200))
    for k, value in query.view.mapping.items():
        torch.testing.assert_close(value, before[k])
    refresh(query, mask=[True, False], tick=(1, 200))
    assert query.view.sample_tick.tolist() == [1, 20]
