"""两机ISAC参考包；登记不加载Torch/Warp。"""
from muse_sdk.registration import TrainingPackageDefinition
from .task import IsacTask
from .terms import registrations


def build_training_package():
    return TrainingPackageDefinition(registrations(), {'isac_task': IsacTask}, {}, {}, {})
