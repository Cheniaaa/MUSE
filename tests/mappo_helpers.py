"""合作示例的注册入口，供 CPU 与 GPU 验收共用。"""
import importlib.util
import json
from pathlib import Path

from flight_helpers import archive

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/cooperative_mappo'


def example_definition():
    spec = importlib.util.spec_from_file_location('cooperative_package', EXAMPLE / 'training_package/__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_training_package()


def cooperative_pipeline(pipeline, tmp_path):
    catalog, _ = pipeline
    request = json.loads((EXAMPLE / 'request_train.json').read_text())
    for kind, directory in [('environment', 'environment'), ('training_package', 'training_package')]:
        item = catalog.register_archive(kind, archive(EXAMPLE / directory, tmp_path / f'{kind}.zip'))
        request[kind + '_id'] = item.resource_id
        request[kind + '_revision'] = item.revision
    return catalog, request
