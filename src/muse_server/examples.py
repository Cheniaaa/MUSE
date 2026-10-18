"""固定参考模板；只登记资源、填入引用，不启动Run或导入示例代码。"""
import json
from pathlib import Path
import sys
import tempfile
import zipfile

from muse_sdk.config import ConfigError
from muse_sdk.experiment import ResourceRef

from .catalog import Catalog


_EXAMPLES = {
    'basic_flight': {'label': '基础飞行', 'package': 'training_package_ippo', 'learning': 'not_evaluated'},
    'isac_cost': {'label': 'ISAC 与 cost 对照', 'package': 'training_package', 'learning': 'not_met'},
    'hybrid_history_markov': {
        'label': '混合动作、历史与 Markov 干扰', 'package': 'training_package', 'learning': 'incomplete'},
    'empty_environment': {'label': '空白地面环境', 'package': None, 'learning': 'not_evaluated'},
}


def list_examples() -> list[dict]:
    return [{'id': identity, 'label': example['label'],
             'research_status': {'engineering': 'available', 'learning': example['learning']}}
            for identity, example in _EXAMPLES.items()]


def import_example(catalog: Catalog, example_id: str) -> dict:
    if example_id not in _EXAMPLES:
        raise ConfigError('example_id', f'Unknown example: {example_id}')
    example = _EXAMPLES[example_id]
    root = Path(__file__).resolve().parents[2] / 'examples'
    if not root.is_dir():
        root = Path(sys.prefix) / 'share/muse/examples'
    directory = root / example_id
    package = example['package']
    request = json.loads((directory / 'request_train.json').read_text()) if package else {}
    inputs = [('environment', directory / 'environment' if package else directory)]
    if package:
        inputs.append(('training_package', directory / package))
    resources = []
    with tempfile.TemporaryDirectory(dir=catalog.directory) as temporary:
        for kind, source in inputs:
            archive_path = Path(temporary) / f'{kind}.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                for path in sorted(source.rglob('*')):
                    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                        archive.write(path, path.relative_to(source))
            record = catalog.register_archive(kind, archive_path)
            request[kind + '_id'] = record.resource_id
            request[kind + '_revision'] = record.revision
            resources.append(ResourceRef(resource_id=record.resource_id, revision=record.revision,
                                         sha256=record.sha256).model_dump(mode='json'))
    # 预算沿RunRequest必填契约，由客户端明确填写，不在模板中另设默认值。
    return {'request': request, 'run_options': {'mode': 'TRAIN'} if package else {}, 'resources': resources,
            'research_status': {'engineering': 'available', 'learning': example['learning']}}
