"""M3.5调度oracle真实注册包；不提供科学传播或学习算法。"""
from muse_sdk.registration import TrainingPackageDefinition
from .algorithm import WirelessRule
from .models import SyntheticRadio, SyntheticSensing, TransmissionSensing
from .terms import registrations


def build_training_package():
    return TrainingPackageDefinition(registrations(), {}, {'wireless_rule': WirelessRule},
        {'synthetic_radio': SyntheticRadio}, {'synthetic_sensing': SyntheticSensing,
                                              'transmission_sensing': TransmissionSensing})
