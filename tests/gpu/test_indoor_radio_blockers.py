"""分别只遮挡desired与jammer，验证真实几何→科学模型的不同SINR效应。"""
from copy import deepcopy
import json

import torch

from flight_helpers import archive
from indoor_radio_helpers import indoor_radio_catalog
from test_indoor_radio_env import indoor_env, step_rule


def test_only_desired_or_jammer_blocker_changes_its_own_path(tmp_path, gpu_stream):
    catalog, source = indoor_radio_catalog(tmp_path)
    source['algorithm_config']['phases'] = [dict(duration_decisions=100, motion=[0., 0., 0., 0.],
                                                  power_w=.001, channel_id=7)]
    outputs, visibility = {}, {}
    for case, center in [('clear', '15 9 1.5'), ('desired', '10 0 1.5'), ('jammer', '15 4 1.5')]:
        root = tmp_path / case
        root.mkdir()
        (root / 'environment.yaml').write_text('scene: scene.xml\nrf_geometry: {rf_blocking: [wall]}\n')
        (root / 'scene.xml').write_text(f'''<mujoco><worldbody>
            <geom name="floor" type="plane" size="15 10 .1" pos="10 0 0"/>
            <geom name="wall" type="box" size=".1 .3 1.5" pos="{center}"/>
            </worldbody></mujoco>''')
        record = catalog.register_archive('environment', archive(root, tmp_path / (case + '.zip')))
        request = deepcopy(source)
        request.update(environment_id=record.resource_id, environment_revision=record.revision)
        with indoor_env(catalog, request) as (env, rule):
            output = step_rule(env, rule).transition.wireless.radio
            outputs[case] = tuple(t[0, 0, 0].item() for t in (
                output.link.received_power_w, output.interference.interference_power_w,
                output.interference.sinr_linear, output.rate.rate_bps))
            assert all(block.valid[0, 0, 0] for block in output)
            geometry = env.experiment.geometry.layout
            desired_path = geometry.desired[0].path
            source_paths = next(c.sources for c in geometry.interference if c.kind == 'desired' and c.consumer == 0)
            jammer_path = next(s.path for s in source_paths if geometry.points[s.tx].group_id == 'anchor_group'
                               and geometry.points[s.tx].slot == 1)
            visibility[case] = env.network.geometry.los[0, [desired_path, jammer_path]].tolist()
    assert visibility == {'clear': [True, True], 'desired': [False, True], 'jammer': [True, False]}
    clear, desired, jammer = (outputs[name] for name in ('clear', 'desired', 'jammer'))
    assert desired[0] < clear[0] and desired[2] < clear[2] and desired[3] < clear[3]
    assert jammer[1] < clear[1] and jammer[2] > clear[2] and jammer[3] > clear[3]
    torch.testing.assert_close(torch.tensor(desired[1]), torch.tensor(clear[1]), rtol=1e-5, atol=0)
    torch.testing.assert_close(torch.tensor(jammer[0]), torch.tensor(clear[0]), rtol=1e-5, atol=0)
    (tmp_path / 'blocker-controls.json').write_text(json.dumps(dict(
        fields=['signal_w', 'interference_w', 'sinr', 'rate_bps'], visibility=visibility, outputs=outputs), indent=2))
