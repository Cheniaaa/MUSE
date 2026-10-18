"""编辑检查结果；只保存本次解析视图，不成为另一份运行配置。"""
from pydantic import ConfigDict, Field

from .config import DataModel
from .experiment import ExperimentDefinition


class InspectionSection(DataModel):
    model_config = ConfigDict(serialize_by_alias=True)

    path: list[str | int]
    config_schema: dict = Field(alias='schema')
    effective: dict
    origins: dict[str, str]
    missing: list[str]
    errors: list[dict[str, str]]


class InspectionReport(DataModel):
    sections: list[InspectionSection]
    errors: list[dict[str, str]]
    resolved_experiment: ExperimentDefinition | None
