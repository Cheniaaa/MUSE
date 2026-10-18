"""CPU生成builtin metadata；Server只读取生成文件。"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
from pydantic import Field

from muse_sdk.config import DataModel, export_config_schema
from muse_sdk.descriptor import build_descriptor
from muse_sdk.experiment import BuiltinMetadata, ResourceRecord
from muse_sdk.geometry import RadioDeviceDefinition
from muse_sdk.registration import TrainingPackageDefinition
from muse_sdk.content import hash_package

from .algorithms import registrations as algorithm_registrations
from .models import registrations as radio_registrations, sensing_registrations
from .terms import EmptyConfig, registrations


class ImuConfig(DataModel):
    update_rate_hz: float = Field(gt=0)


class ImuRangefinderConfig(DataModel):
    imu_update_rate_hz: float = Field(gt=0)
    rangefinder_update_rate_hz: float = Field(gt=0)


def export_metadata(controller_path: Path) -> BuiltinMetadata:
    controller = json.loads(controller_path.read_text())
    asset = Path(__file__).parent / 'assets/bitcraze_crazyflie_2'
    asset_hash = hash_package(asset)
    definition = TrainingPackageDefinition(registrations(), {}, algorithm_registrations(), radio_registrations(), sensing_registrations())
    descriptor = build_descriptor(definition, {}, asset_hash, default_composition_id=None)
    controllers = {controller['id']: controller, 'none': {'id': 'none'}}
    controllers['diff_drive_controller'] = {
        'id': 'diff_drive_controller',
        'config': {'wheel_radius_m': .03, 'wheel_separation_m': .12,
                   'wheel_damping_nm_s': .03, 'wheel_velocity_kp_nm_s': .01},
        'motion': {'fields': ['v_forward', 'yaw_rate'], 'units': ['m/s', 'rad/s'],
                   'frames': ['body_flu', 'world_heading'],
                   'minimum': [-.1, -.8], 'maximum': [.1, .8]},
    }
    models = {'bitcraze_crazyflie_2': ResourceRecord(
        resource_id='bitcraze_crazyflie_2', revision=1, sha256=asset_hash,
        kind='entity_model', path=str(asset.resolve()), metadata={
            'entity_kind': 'aerial_vehicle', 'physics_asset': 'cf2.xml',
            'root_body': 'cf2', 'support_geoms': [],
            'action_interfaces': {'motion': 'body_velocity'}, 'controller_ids': [controller['id']],
            'sensor_profile_ids': ['none', 'cf2_imu', 'cf2_imu_rangefinder_down'],
        })}
    car_asset = Path(__file__).parent / 'assets/ugv_diff_drive'
    models['ugv_diff_drive'] = ResourceRecord(
        resource_id='ugv_diff_drive', revision=1, sha256=hash_package(car_asset),
        kind='entity_model', path=str(car_asset.resolve()), metadata={
            'entity_kind': 'ground_vehicle', 'physics_asset': 'car.xml',
            'root_body': 'car', 'support_geoms': ['front wheel', 'left_wheel', 'right_wheel'],
            'action_interfaces': {'motion': 'diff_drive'}, 'controller_ids': ['diff_drive_controller'],
            'sensor_profile_ids': ['none'],
        })
    station_asset = Path(__file__).parent / 'assets/radio_base_station'
    models['radio_base_station'] = ResourceRecord(
        resource_id='radio_base_station', revision=1, sha256=hash_package(station_asset),
        kind='entity_model', path=str(station_asset.resolve()), metadata={
            'entity_kind': 'infrastructure', 'physics_asset': 'station.xml',
            'root_body': 'station', 'support_geoms': [],
            'action_interfaces': {}, 'controller_ids': ['none'],
            'sensor_profile_ids': ['none'],
        })
    for model_id in ('bitcraze_crazyflie_2', 'radio_base_station'):
        record = models[model_id]
        body = record.metadata['root_body']
        native = mujoco.MjModel.from_xml_path(str(Path(record.path) / record.metadata['physics_asset']))
        native.body(body)
        record.metadata['radio_devices'] = {'ideal_point_radio': RadioDeviceDefinition(
            can_transmit=True, can_receive=True, body=body, position_m=[0., 0., 0.],
            orientation_wxyz=[1., 0., 0., 0.],
            assumption='理想root点参考，不代表真实硬件天线相位中心').model_dump(mode='json')}
    imu_outputs = {
        'gyro': {'sensor_name': 'body_gyro', 'units': 'rad/s', 'frame': 'imu_flu',
                 'width': 3, 'sensor_type': 'gyro', 'mount': 'imu', 'rate_key': 'update_rate_hz'},
        'specific_force': {'sensor_name': 'body_linacc', 'units': 'm/s²', 'frame': 'imu_flu',
                           'width': 3, 'sensor_type': 'accelerometer', 'mount': 'imu',
                           'rate_key': 'update_rate_hz'},
    }
    sensors = {
        'none': {'sensor_profile_id': 'none', 'outputs': {},
                 'config_schema': export_config_schema(EmptyConfig)},
        'cf2_imu': {'sensor_profile_id': 'cf2_imu',
                    'config_schema': export_config_schema(ImuConfig),
                    'outputs': imu_outputs},
        'cf2_imu_rangefinder_down': {
                    'sensor_profile_id': 'cf2_imu_rangefinder_down',
                    'config_schema': export_config_schema(ImuRangefinderConfig),
                    'outputs': {
                        **{signal: {**output, 'rate_key': 'imu_update_rate_hz'}
                           for signal, output in imu_outputs.items()},
                        'range': {'sensor_name': 'rangefinder_down', 'units': 'm', 'frame': 'none',
                                  'width': 1, 'sensor_type': 'rangefinder', 'mount': 'rangefinder_down',
                                  'rate_key': 'rangefinder_update_rate_hz',
                                  'position': [0., 0., -.01], 'orientation_wxyz': [0., 1., 0., 0.],
                                  'min_range_m': .04, 'max_range_m': 4., 'minimum_period_s': .144},
                    }},
    }
    # 平台revision覆盖四个实际包及controller参数，不依赖Git。
    digest = hashlib.sha256()
    source = Path(__file__).parent.parent
    for package in ('muse_sdk', 'muse_server', 'muse_runtime', 'muse_builtins'):
        digest.update(hash_package(source / package).encode())
    digest.update(json.dumps(controllers, sort_keys=True).encode())
    return BuiltinMetadata(platform_revision=digest.hexdigest(), descriptor=descriptor,
                           entity_models=models, controllers=controllers, sensors=sensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        stream.write(export_metadata(args.controller).model_dump_json(indent=2))


if __name__ == '__main__':
    main()
