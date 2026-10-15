"""builtin Radio/Sensing类登记；metadata与Worker共用，不加载数值后端。"""
from muse_sdk.models import RadioModel, SensingModel

from .analytical_radio import FreeSpaceRadio, IndoorOfficeRadio
from .sensing import FreeSpacePathLengthEFIM2D, FreeSpacePathLengthMeasurement, FreeSpacePositionEFIM2D


def registrations() -> dict[str, type[RadioModel]]:
    return {'free_space_radio': FreeSpaceRadio, 'indoor_office_radio': IndoorOfficeRadio}


def sensing_registrations() -> dict[str, type[SensingModel]]:
    return {'free_space_path_length_measurement': FreeSpacePathLengthMeasurement,
            'free_space_position_efim_2d': FreeSpacePositionEFIM2D,
            'free_space_path_length_efim_2d': FreeSpacePathLengthEFIM2D}
