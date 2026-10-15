"""真实Scene的有限RF表面查询；宿主拥有forward、world tick及调度。"""
import math

import mujoco
import torch
import warp as wp
from mujoco_warp._src.ray import ray_box

from muse_sdk.geometry import GeometryView
from ..scene import entity_name

_BOX = wp.constant(int(mujoco.mjtGeom.mjGEOM_BOX))


@wp.kernel(enable_backward=False)
def _segments(
    worlds: wp.array[wp.bool], positions: wp.array2d[wp.vec3], present: wp.array2d[wp.bool],
    point_bodies: wp.array[int], paths: wp.array[wp.vec2i],
    blockers: wp.array[int], mesh_ids: wp.array[wp.uint64],
    geom_bodies: wp.array[int], geom_types: wp.array[int],
    geom_pos: wp.array2d[wp.vec3], geom_mat: wp.array2d[wp.mat33], geom_size: wp.array2d[wp.vec3],
    epsilon: float, distances: wp.array2d[float], los: wp.array2d[wp.bool],
    valid: wp.array2d[wp.bool], faults: wp.array2d[wp.bool],
):
    world, path = wp.tid()
    if not worlds[world]:
        return
    a, b = paths[path][0], paths[path][1]
    distances[world, path] = 0.0
    los[world, path] = False
    valid[world, path] = False
    faults[world, path] = False
    if not present[world, a] or not present[world, b]:
        return
    delta = positions[world, b] - positions[world, a]
    distance = wp.length(delta)
    if not wp.isfinite(distance):
        faults[world, path] = True
        return
    if distance <= 2.0 * epsilon:
        return
    distances[world, path] = distance
    valid[world, path] = True
    direction = delta / distance
    origin = positions[world, a] + epsilon * direction
    limit = distance - 2.0 * epsilon
    blocked = bool(False)
    for index in range(blockers.shape[0]):
        geom = blockers[index]
        if geom_bodies[geom] == point_bodies[a] or geom_bodies[geom] == point_bodies[b]:
            continue
        pos, mat = geom_pos[world, geom], geom_mat[world, geom]
        hit = float(-1.0)
        if geom_types[geom] == _BOX:
            hit, sides, normal = ray_box(pos, mat, geom_size[world % geom_size.shape[0], geom], origin, direction)
        else:
            inverse = wp.transpose(mat)
            result = wp.mesh_query_ray(mesh_ids[index], inverse @ (origin - pos), inverse @ direction, limit)
            if result.result:
                hit = result.t
        if not wp.isfinite(hit):
            faults[world, path] = True
        if hit >= 0.0 and hit < limit:
            blocked = True
    los[world, path] = not blocked


class GeometryQuery:
    @torch.no_grad()
    def __init__(self, resolved, simulation, *, physics_dt: float, device: str):
        self._device = torch.device(device)
        if self._device.type != 'cuda':
            raise ValueError('RF GeometryQuery requires CUDA')
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('Expected positive finite physics_dt')
        self._simulation, self._dt = simulation, physics_dt
        self._stream = wp.get_stream(device)
        torch_stream = torch.cuda.current_stream(self._device)
        if self._stream.cuda_stream != torch_stream.cuda_stream:
            self._stream = wp.stream_from_torch(torch_stream)
        self._epsilon = resolved.endpoint_epsilon_m
        model, data = simulation.mj_model, simulation.wp_data
        if str(data.xpos.device) != str(self._device):
            raise ValueError('Scene and geometry device mismatch')
        self._batch = data.nworld
        layout = resolved.layout
        q, p = len(layout.points), len(layout.paths)
        bodies = [model.body(f'{entity_name(point.group_id, point.slot)}/{point.body}').id for point in layout.points]
        blockers = [model.geom('environment/' + name).id for name in resolved.blockers]
        self._bodies = torch.tensor(bodies, dtype=torch.long, device=device)
        self._blocker_indices = torch.tensor(blockers, dtype=torch.long, device=device)
        self._mounts = torch.tensor([point.position_m for point in layout.points], dtype=torch.float32, device=device)
        self._point_bodies = wp.array(bodies, dtype=int, device=device)
        self._paths = wp.array([(path.tx, path.rx) for path in layout.paths], dtype=wp.vec2i, device=device)
        self._blockers = wp.array(blockers, dtype=int, device=device)
        # mesh对象持有BVH及底层points/indices；同一编译mesh只建立一次。
        self._meshes = {}
        mesh_ids = []
        with wp.ScopedStream(self._stream):
            for geom in blockers:
                mesh_id = 0
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(model.geom_dataid[geom])
                    if mid not in self._meshes:
                        va, fa = model.mesh_vertadr[mid], model.mesh_faceadr[mid]
                        vertices = model.mesh_vert[va:va+model.mesh_vertnum[mid]]
                        faces = model.mesh_face[fa:fa+model.mesh_facenum[mid]]
                        self._meshes[mid] = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device),
                            indices=wp.array(faces.flatten(), dtype=int, device=device))
                    mesh_id = self._meshes[mid].id
                mesh_ids.append(mesh_id)
            self._mesh_ids = wp.array(mesh_ids, dtype=wp.uint64, device=device)
        values = dict(positions_world=torch.zeros((self._batch, q, 3), device=device),
            distance_m=torch.zeros((self._batch, p), device=device),
            los=torch.zeros((self._batch, p), dtype=torch.bool, device=device),
            valid=torch.zeros((self._batch, p), dtype=torch.bool, device=device),
            sample_tick=torch.zeros(self._batch, dtype=torch.int64, device=device),
            sample_time=torch.zeros(self._batch, dtype=torch.float64, device=device))
        self.view = GeometryView(**values, layout=layout)
        self._scratch = {name: torch.empty_like(value) for name, value in values.items()}
        self._faults = torch.zeros((self._batch, p), dtype=torch.bool, device=device)
        self._wp_positions = wp.from_torch(self._scratch['positions_world'], dtype=wp.vec3)
        self._wp_outputs = [wp.from_torch(self._scratch[name]) for name in ('distance_m', 'los', 'valid')]
        self._wp_outputs.append(wp.from_torch(self._faults))
        self._checked_inputs = False

    @torch.no_grad()
    def refresh(self, world_mask, *, sample_tick, present):
        if torch.cuda.current_stream(self._device).cuda_stream != self._stream.cuda_stream:
            raise ValueError('GeometryQuery must run on its bound Worker stream')
        if not self._checked_inputs:
            for name, value, shape, dtype in (
                ('world_mask', world_mask, (self._batch,), torch.bool),
                ('sample_tick', sample_tick, (self._batch,), torch.int64),
                ('present', present, (self._batch, len(self.view.layout.points)), torch.bool)):
                if value.shape != shape or value.dtype != dtype or value.device != self._device:
                    raise ValueError(f'{name}: tensor shape/dtype/device mismatch')
            self._checked_inputs = True
        worlds = world_mask.clone()
        if not worlds.any():
            return self.view
        if (sample_tick[worlds] < 0).any():
            raise ValueError('sample_tick must be nonnegative')
        data, model = self._simulation.wp_data, self._simulation.wp_model
        body_pos = wp.to_torch(data.xpos)[:, self._bodies]
        body_mat = wp.to_torch(data.xmat)[:, self._bodies]
        geom_pos = wp.to_torch(data.geom_xpos)[:, self._blocker_indices]
        geom_mat = wp.to_torch(data.geom_xmat)[:, self._blocker_indices]
        if not all(torch.isfinite(value[worlds]).all() for value in (body_pos, body_mat, geom_pos, geom_mat)):
            raise RuntimeError('RF physical pose must be finite')
        positions = self._scratch['positions_world']
        positions.copy_(body_pos + torch.einsum('bqij,qj->bqi', body_mat, self._mounts))
        if not torch.isfinite(positions[worlds]).all():
            raise RuntimeError('RF mount position must be finite')
        positions.masked_fill_(~present.unsqueeze(-1), 0.)
        with wp.ScopedStream(self._stream):
            wp.launch(_segments, dim=self.view.valid.shape, inputs=[wp.from_torch(worlds),
                self._wp_positions, wp.from_torch(present), self._point_bodies, self._paths,
                self._blockers, self._mesh_ids, model.geom_bodyid, model.geom_type,
                data.geom_xpos, data.geom_xmat, model.geom_size, self._epsilon],
                outputs=self._wp_outputs, device=str(self._device))
        if self._faults[worlds].any():
            raise RuntimeError('RF query produced nonfinite values')
        self._scratch['sample_tick'].copy_(sample_tick)
        self._scratch['sample_time'].copy_(sample_tick.to(torch.float64) * self._dt)
        if not torch.isfinite(self._scratch['sample_time'][worlds]).all():
            raise RuntimeError('RF sample time must be finite')
        # 所有source位于私有scratch；world选择在首次写缓存前已固定。
        for name, value in self.view.mapping.items():
            value[worlds] = self._scratch[name][worlds]
        return self.view
