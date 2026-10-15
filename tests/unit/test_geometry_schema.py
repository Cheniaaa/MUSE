"""公开设备/实验边界；派生摘要和索引不能由实验输入注入。"""
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError


def test_geometry_sdk_import_is_backend_free():
    result = subprocess.run([sys.executable, '-c', '''
import sys
import muse_sdk.geometry
import muse_server.catalog
assert not {'torch', 'mujoco', 'warp', 'muse_runtime'} & sys.modules.keys()
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('epsilon', [0., -1., float('nan'), float('inf'), 1e-50, 1e40])
def test_epsilon_must_remain_positive_finite_float32(epsilon):
    from muse_sdk.geometry import GeometrySelection
    with pytest.raises(ValidationError):
        GeometrySelection(endpoints={}, links=[], sensing=[], endpoint_epsilon_m=epsilon)


@pytest.mark.parametrize('extra', ['geometry_summary', 'paths', 'handles'])
def test_public_selection_forbids_platform_derived_data(extra):
    from muse_sdk.geometry import GeometrySelection
    with pytest.raises(ValidationError):
        GeometrySelection(endpoints={}, links=[], sensing=[], **{extra: {}})


@pytest.mark.parametrize('change', [dict(position_m=[0., 0.]),
    dict(orientation_wxyz=[0., 0., 0., 0.]), dict(orientation_wxyz=[2., 0., 0., 0.]),
    dict(position_m=[float('nan'), 0., 0.])])
def test_device_mount_is_finite_and_rigid(change):
    from muse_sdk.geometry import RadioDeviceDefinition
    data = dict(body='root', can_transmit=True, can_receive=True,
                position_m=[0., 0., 0.], orientation_wxyz=[1., 0., 0., 0.], assumption='ideal')
    with pytest.raises(ValidationError):
        RadioDeviceDefinition(**(data | change))


def test_missing_rf_and_explicit_empty_rf_are_distinct():
    from muse_sdk.experiment import EnvironmentBundle
    missing = EnvironmentBundle(scene='scene.xml')
    empty = EnvironmentBundle.model_validate(dict(scene='scene.xml', rf_geometry={'rf_blocking': []}))
    assert json.loads(missing.model_dump_json())['rf_geometry'] is None
    assert json.loads(empty.model_dump_json())['rf_geometry'] == {'rf_blocking': []}
