"""IPPO混合动作与隐藏Markov干扰参考包。"""
from muse_sdk.registration import TrainingPackageDefinition
from .task import HistoryTask
from .terms import registrations


def build_training_package():
    return TrainingPackageDefinition(registrations(), {'history_task': HistoryTask}, {}, {}, {})
