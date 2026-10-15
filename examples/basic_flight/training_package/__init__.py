"""真实包注册入口，内置实现通过显式builtin引用选取。"""
from muse_sdk.registration import TrainingPackageDefinition


def build_training_package():
    return TrainingPackageDefinition({}, {}, {}, {}, {})
