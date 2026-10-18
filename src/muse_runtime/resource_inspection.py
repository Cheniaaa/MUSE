"""one-shot CPU原生环境检查；事实与资源内容摘要在同一进程产生。"""
import argparse
import json
from pathlib import Path

import mujoco

from muse_sdk.geometry import GeometryAssetFact, GeometryAssetSummary
from muse_sdk import content


def inspect_environment(root: Path, scene: str) -> GeometryAssetSummary:
    model = mujoco.MjModel.from_xml_path(str(root / scene))
    static = [True] * model.nbody
    for body in range(1, model.nbody):
        static[body] = (static[model.body_parentid[body]]
                        and model.body_jntnum[body] == 0 and model.body_mocapid[body] < 0)
    geoms = {}
    for index in range(model.ngeom):
        geom = model.geom(index)
        if not geom.name:
            continue
        body = int(model.geom_bodyid[index])
        kind = mujoco.mjtGeom(int(model.geom_type[index])).name.removeprefix('mjGEOM_').lower()
        geoms[geom.name] = GeometryAssetFact(geom_type=kind, body=model.body(body).name,
            is_static=bool(static[body]),
            mesh=model.mesh(int(model.geom_dataid[index])).name if kind == 'mesh' else None)
    flex_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, i) for i in range(model.nflex)]
    return GeometryAssetSummary(geoms=geoms, flex_names=[name for name in flex_names if name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = inspect_environment(args.root, args.scene)
    args.output.write_text(json.dumps({'geometry_summary': summary.model_dump(mode='json'),
                                       'sha256': content.hash_package(args.root)}))


if __name__ == '__main__':
    main()
