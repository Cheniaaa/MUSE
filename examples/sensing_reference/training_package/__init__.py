"""科学感知参考包；具体law由显式builtin binding提供。"""
from muse_sdk.registration import TrainingPackageDefinition
from .algorithm import SensingReferenceRule
from .terms import registrations


def build_training_package():
    return TrainingPackageDefinition(registrations(), {}, {'sensing_reference_rule': SensingReferenceRule}, {}, {})
