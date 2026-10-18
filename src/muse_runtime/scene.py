from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import MujocoCfg

from muse_sdk.scene_edit import PreviewEntity

from .environment_edit import load_environment_spec


@dataclass(kw_only=True)
class FlightOptions(MujocoCfg):
    flight_integrator: str
    density: float
    viscosity: float

    def apply(self, model):
        super().apply(model)
        model.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + self.flight_integrator)
        model.opt.density = self.density
        model.opt.viscosity = self.viscosity


def entity_name(group, slot):
    return f'{group}_{slot}'


def make_scene_config(environment_root: Path, environment_scene: str, entities: list[PreviewEntity],
                      resource_paths: dict[str, str], *, world_count: int) -> SceneCfg:
    """预览与 Run 的唯一冷装配路径；只构造原生模型，不初始化执行状态。"""
    configured = {}
    placements = {}
    for entity in entities:
        asset = Path(resource_paths['entity_model:' + entity.model.resource_id]) / entity.model.physics_asset
        root_body = entity.model.root_body
        range_outputs = [output for output in entity.sensor_outputs if output['sensor_type'] == 'rangefinder']
        name = entity_name(entity.group_id, entity.slot)
        def load(path=asset, root=root_body, outputs=range_outputs):
            spec = mujoco.MjSpec.from_file(str(path))
            for key in list(spec.keys):
                spec.delete(key)
            for output in outputs:
                spec.body(root).add_site(name=output['mount'],
                    pos=output['position'], quat=output['orientation_wxyz'])
                sensor = spec.add_sensor(name=output['sensor_name'],
                    type=mujoco.mjtSensor.mjSENS_RANGEFINDER, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname=output['mount'])
                # 3.11要求显式dist；在Entity装配时添加，Scene才会自动包装BuiltinSensor。
                sensor.intprm[0] = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)
            return spec
        configured[name] = EntityCfg(spec_fn=load, init_state=EntityCfg.InitialStateCfg(
            pos=tuple(entity.pose.position), rot=tuple(entity.pose.orientation_wxyz)))
        placements[name] = (entity.pose, root_body)
    def configure(spec):
        environment = load_environment_spec(environment_root, environment_scene)
        spec.attach(environment, prefix='environment/', frame=spec.worldbody.add_frame())
        for name, (pose, root_body) in placements.items():
            body = spec.body(f'{name}/{root_body}')
            # 固定体的placement已由mjlab写入mocap根；自由体还需同步编译初态。
            if any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in body.joints):
                body.pos = pose.position
                body.quat = pose.orientation_wxyz
    return SceneCfg(num_envs=world_count, entities=configured, spec_fn=configure)


def assemble_scene(experiment, resource_paths):
    entities = []
    for group in experiment.scenario.entity_groups:
        outputs = list(experiment.sensors[group.group_id]['outputs'].values()) if (
            'sensor:' + group.group_id in experiment.activation) else []
        for slot, pose in enumerate(experiment.scenario.placements[group.placement_id]):
            entities.append(PreviewEntity(object_id=entity_name(group.group_id, slot),
                group_id=group.group_id, slot=slot, model=experiment.entity_models[group.model_id],
                pose=pose, sensor_outputs=outputs))
    scene = make_scene_config(Path(resource_paths['environment']), experiment.environment_scene,
                              entities, resource_paths, world_count=experiment.topology.world_count)
    cfg = experiment.scenario.scene
    simulation = SimulationCfg(nconmax=128, njmax=512, mujoco=FlightOptions(
        timestep=cfg.physics_dt, flight_integrator=cfg.integrator,
        density=cfg.density, viscosity=cfg.viscosity))
    return scene, simulation
