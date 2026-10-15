import shutil

import pytest

from test_supervisor import make_plan


@pytest.mark.parametrize('kind', ['environment', 'entity_model:bitcraze_crazyflie_2'])
def test_worker_rejects_changed_frozen_physics_resource_before_cuda(tmp_path, kind):
    from muse_runtime.worker import execute

    _, plan = make_plan(tmp_path)
    # 修改临时副本，绝不改生产资产；CPU device确保反例不会意外启动物理。
    copied = tmp_path / 'changed-resource'
    shutil.copytree(plan.resource_paths[kind], copied)
    plan.resource_paths[kind] = str(copied)
    scene = copied / (plan.experiment.environment_scene if kind == 'environment' else 'cf2.xml')
    scene.write_text(scene.read_text() + '\n<!-- changed after resolution -->\n')
    plan.device = 'cpu'
    with pytest.raises(ValueError, match=f'Frozen resource hash mismatch: {kind}'):
        execute(plan, None)
