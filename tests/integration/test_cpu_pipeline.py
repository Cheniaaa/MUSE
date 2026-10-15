"""两个独立解释器间的真实数据契约，不借用父进程注册对象。"""
import json
import os
from pathlib import Path
import subprocess
import sys

from muse_sdk.selection import ResolvedPackageSelection


SOURCE = Path(__file__).resolve().parents[2] / 'src'


def run_python(args, cwd):
    env = {**os.environ, 'PYTHONPATH': str(SOURCE), 'PYTHONDONTWRITEBYTECODE': '1'}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=15)


def test_independent_validation_and_blocked_resolver(fixture_root, tmp_path):
    output = tmp_path / 'validation'
    validation = run_python(['-m', 'muse_runtime.validation', '--package', str(fixture_root), '--output-dir', str(output)], tmp_path)
    assert validation.returncode == 0, validation.stderr
    report = json.loads((output / 'validation-report.json').read_text())
    assert report['valid'] is True and report['resolved_experiment'] is None
    config = tmp_path / 'override.json'
    config.write_text(json.dumps({'roles': {'scout': {'reward': {'progress': {'config': {'scale': 3.0}, 'weight': 0.25}}}}}))
    result_path = tmp_path / 'selection.json'
    blocker = '''
import importlib.abc, runpy, sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'muse_runtime', 'muse_builtins', 'torch', 'mjlab', 'mujoco', 'warp'} or fullname.startswith('_muse_package_'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
runpy.run_module('muse_server.resolver', run_name='__main__')
'''
    resolved = run_python(['-c', blocker, '--descriptor', str(output / 'descriptor.json'),
                          '--term-config', str(config), '--output', str(result_path)], tmp_path)
    assert resolved.returncode == 0, resolved.stderr
    assert result_path.exists(), 'Resolver CLI must write its result'
    result = ResolvedPackageSelection.model_validate_json(result_path.read_text())
    assert result.package_hash == report['package_hash']
    assert result.selections[0].config == {'scale': 3.0}
    assert result.selections[0].weight == 0.25
    assert result.selections[2].config == {'scale': 1.0}
    assert result.termination_reasons == {1: ('world', 'termination', 'collision'), 2: ('world', 'termination', 'deadline')}
    assert json.loads(result_path.read_text())['termination_reasons']['2'] == ['world', 'termination', 'deadline']


def test_sdk_import_is_cpu_only(tmp_path):
    script = '''
import importlib.abc, sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','mujoco','warp','mjlab','fastapi','sqlite3','muse_runtime','muse_builtins'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockImports())
import muse_sdk.config, muse_sdk.registration, muse_sdk.descriptor, muse_sdk.selection
'''
    result = run_python(['-c', script], tmp_path)
    assert result.returncode == 0, result.stderr


def test_resolver_bad_override_never_publishes_result(descriptor, tmp_path):
    data = tmp_path / 'descriptor.json'
    data.write_text(descriptor.model_dump_json())
    config = tmp_path / 'override.json'
    config.write_text('{"roles":{"scout":{"reward":{"progress":{"enabled":false}}}}}')
    target = tmp_path / 'result.json'
    result = run_python(['-m', 'muse_server.resolver', '--descriptor', str(data), '--term-config', str(config), '--output', str(target)], tmp_path)
    assert result.returncode != 0
    assert not target.exists()
    errors = json.loads(result.stderr)['errors']
    assert errors and 'scout' in errors[0]['field']


def test_duplicate_json_keys_rejected(descriptor, tmp_path):
    data = tmp_path / 'descriptor.json'
    data.write_text(descriptor.model_dump_json())
    config = tmp_path / 'override.json'
    config.write_text('{"roles":{},"roles":{}}')
    target = tmp_path / 'result.json'
    result = run_python(['-m', 'muse_server.resolver', '--descriptor', str(data), '--term-config', str(config), '--output', str(target)], tmp_path)
    assert result.returncode != 0 and not target.exists()
    assert 'Duplicate' in result.stderr
