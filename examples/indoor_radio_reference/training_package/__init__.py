"""科学Radio验证包：规则动作和只读Term，不内置传播实现。"""
from muse_sdk.registration import TrainingPackageDefinition
from .algorithm import IndoorRadioRule
from .terms import registrations


def build_training_package():
    return TrainingPackageDefinition(registrations(), {}, {'indoor_radio_rule': IndoorRadioRule}, {}, {})
