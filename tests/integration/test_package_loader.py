"""真实目录与独立进程覆盖包生命周期和验证产物。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from muse_sdk.config import ConfigError
from muse_sdk.content import hash_package
from muse_runtime.package_loader import load_package


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def package_dir(tmp_path):
    return shutil.copytree(ROOT / 'tests/fixtures/cpu_package', tmp_path / 'package')


def invoke(package, output, cwd):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    return subprocess.run(
        [sys.executable, '-m', 'muse_runtime.validation', '--package', str(package),
         '--output-dir', str(output)], cwd=cwd, env=env, capture_output=True,
        text=True, timeout=30,
    )


def call_reward(loaded):
    reward = loaded.definition.terms['progress'].implementation(None, None)
    return reward(3, role_id='scout', scale=1.0)


def test_relative_import_and_package_data(package_dir):
    (package_dir / 'asset.txt').write_text('resource', encoding='utf-8')
    entry = package_dir / '__init__.py'
    entry.write_text(entry.read_text() + '\nfrom pathlib import Path\n'
                     'assert Path(__file__).with_name("asset.txt").read_text() == "resource"\n')
    before_path, before_cwd = sys.path[:], Path.cwd()
    with load_package(package_dir) as loaded:
        namespace = loaded.definition.terms['progress'].implementation.__module__.split('.')[0]
        assert namespace + '.deferred' not in sys.modules
        assert call_reward(loaded) == 6
        assert namespace + '.deferred' in sys.modules
        assert loaded.root == package_dir
        assert loaded.package_hash == hash_package(package_dir)
        assert loaded.default_composition_id == 'default'
        assert loaded.compositions['default']['algorithm'] == 'cpu_rule'
    assert not any(n == namespace or n.startswith(namespace + '.') for n in sys.modules)
    assert (sys.path, Path.cwd()) == (before_path, before_cwd)


def test_same_named_packages_remain_isolated(package_dir, tmp_path):
    second = shutil.copytree(package_dir, tmp_path / 'other' / 'package')
    (second / 'deferred.py').write_text('def transform(value):\n    return value * 5\n')
    with load_package(package_dir) as first:
        with load_package(second) as other:
            assert call_reward(first) == 6
            assert call_reward(other) == 15
        assert call_reward(first) == 6


def test_context_exception_cleans_modules(package_dir):
    namespace = None
    with pytest.raises(RuntimeError, match='caller failed'):
        with load_package(package_dir) as loaded:
            namespace = loaded.definition.terms['progress'].implementation.__module__.split('.')[0]
            assert call_reward(loaded) == 6
            raise RuntimeError('caller failed')
    assert not any(n == namespace or n.startswith(namespace + '.') for n in sys.modules)


def test_root_symlink_is_rejected(package_dir, tmp_path):
    link = tmp_path / 'linked-package'
    link.symlink_to(package_dir, target_is_directory=True)
    with pytest.raises(ConfigError):
        hash_package(link)


def test_workspace_alias_does_not_change_package_hash(package_dir, tmp_path):
    # 工作区祖先路径可以是链接；约束的是包内容，不是整个主机文件系统。
    alias = tmp_path / 'workspace-alias'
    alias.symlink_to(package_dir.parent, target_is_directory=True)
    assert hash_package(alias / package_dir.name) == hash_package(package_dir)


def test_missing_registration_function(package_dir):
    (package_dir / '__init__.py').write_text('from . import terms\n')
    with pytest.raises(ConfigError, match='build_training_package'):
        load_package(package_dir)


@pytest.mark.parametrize('name', ['algorithms.py', 'deferred.py', 'assets/data.bin', 'tests/check.py'])
def test_hash_covers_persistent_files(package_dir, name):
    initial = hash_package(package_dir)
    path = package_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        stream.write(b'\n# changed\n')
    assert hash_package(package_dir) != initial


def test_hash_framing_order_and_caches(package_dir, tmp_path):
    initial = hash_package(package_dir)
    expected = hashlib.sha256()
    files = sorted(p for p in package_dir.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.suffix != '.pyc')
    original = {p.relative_to(package_dir): p.read_bytes() for p in files}
    for p in files:
        name = p.relative_to(package_dir).as_posix().encode('utf-8')
        data = p.read_bytes()
        expected.update(len(name).to_bytes(8, 'big') + name)
        expected.update(len(data).to_bytes(8, 'big') + data)
    assert initial == expected.hexdigest()
    reverse = tmp_path / 'reverse'
    reverse.mkdir()
    for name, data in reversed(list(original.items())):
        (reverse / name).parent.mkdir(parents=True, exist_ok=True)
        (reverse / name).write_bytes(data)
    assert hash_package(reverse) == initial
    cache = package_dir / '__pycache__'
    cache.mkdir(exist_ok=True)
    (cache / 'arbitrary').write_bytes(b'cache')
    (package_dir / 'ignored.pyc').write_bytes(b'cache')
    with load_package(package_dir):
        pass
    assert hash_package(package_dir) == initial
    assert all((package_dir / p).read_bytes() == data for p, data in original.items())


@pytest.mark.parametrize('name', ['link', '__pycache__/link', 'ignored.pyc'])
@pytest.mark.parametrize('kind', ['file_link', 'directory_link', 'broken_link', 'fifo'])
def test_rejects_unsafe_entries_even_in_cache(package_dir, tmp_path, name, kind):
    path = package_dir / name
    path.parent.mkdir(exist_ok=True)
    if kind == 'fifo':
        os.mkfifo(path)
    else:
        target = {'file_link': package_dir / 'terms.py',
                  'directory_link': tmp_path, 'broken_link': tmp_path / 'missing'}[kind]
        path.symlink_to(target)
    with pytest.raises(ConfigError):
        hash_package(package_dir)
    with pytest.raises(ConfigError):
        load_package(package_dir)


@pytest.mark.parametrize('name', ['__init__.py', 'training_package.yaml'])
def test_missing_required_file(package_dir, name):
    (package_dir / name).unlink()
    with pytest.raises(ConfigError, match=name):
        load_package(package_dir)


@pytest.mark.parametrize('yaml', ['algorithm: [', 'algorithm: cpu_rule\nalgorithm: alternate\n',
                                  'nested: {x: 1, x: 2}', '!!python/object:builtins.object {}', '[]'])
def test_yaml_rejected(package_dir, yaml):
    (package_dir / 'training_package.yaml').write_text(yaml)
    with pytest.raises(ConfigError):
        load_package(package_dir)


@pytest.mark.parametrize('tail', ['raise RuntimeError("import exploded")',
                                 'def build_training_package():\n    raise SystemExit("build exploded")',
                                 'def build_training_package():\n    return {}'])
def test_load_failure_cleans_own_namespace(package_dir, tail):
    entry = package_dir / '__init__.py'
    entry.write_text(entry.read_text() + '\n' + tail + '\n')
    before = {n for n in sys.modules if n.startswith('_muse_package_')}
    with pytest.raises((RuntimeError, SystemExit, ConfigError)):
        load_package(package_dir)
    assert {n for n in sys.modules if n.startswith('_muse_package_')} == before


POLLUTION = '''
import os, sys, ctypes
print('python stdout')
print('python stderr', file=sys.stderr)
os.write(1, b'native stdout\\n')
os.write(2, b'native stderr\\n')
ctypes.CDLL(None).puts(b'libc buffered stdout')
'''


def test_cli_success_is_clean_jsonl(package_dir, tmp_path):
    entry = package_dir / '__init__.py'
    entry.write_text(entry.read_text() + POLLUTION)
    output = tmp_path / 'result'
    result = invoke(package_dir, output, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    terminal, = [json.loads(line) for line in result.stdout.splitlines()]
    report = json.loads((output / 'validation-report.json').read_text())
    descriptor = json.loads((output / 'descriptor.json').read_text())
    assert terminal['valid'] is report['valid'] is True
    assert report['errors'] == []
    assert report['resolved_experiment'] is None
    assert terminal['package_hash'] == report['package_hash'] == descriptor['package_hash'] == hash_package(package_dir)
    assert {p.name for p in output.iterdir()} == {'descriptor.json', 'validation-report.json', 'user.log'}
    log = (output / 'user.log').read_text()
    for text in ('python stdout', 'python stderr', 'native stdout', 'native stderr', 'libc buffered stdout'):
        assert text in log


def test_old_output_untouched_before_import(package_dir, tmp_path):
    output = tmp_path / 'result'
    output.mkdir()
    old = output / 'descriptor.json'
    old.write_bytes(b'old descriptor')
    marker = tmp_path / 'entry-ran'
    (package_dir / '__init__.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    result = invoke(package_dir, output, tmp_path)
    assert result.returncode != 0
    assert 'exist' in result.stderr.lower()
    assert old.read_bytes() == b'old descriptor'
    assert list(output.iterdir()) == [old]
    assert not marker.exists()


@pytest.mark.parametrize('failure', ['import', 'system_exit', 'reference', 'yaml', 'missing'])
def test_cli_failure_has_report_log_no_descriptor(package_dir, tmp_path, failure):
    entry = package_dir / '__init__.py'
    entry.write_text(entry.read_text() + POLLUTION)
    if failure in ('import', 'system_exit'):
        with entry.open('a') as stream:
            stream.write('\nraise ' + ('RuntimeError' if failure == 'import' else 'SystemExit') + '("user exploded")\n')
    elif failure == 'reference':
        path = package_dir / 'training_package.yaml'
        path.write_text(path.read_text().replace('algorithm: cpu_rule', 'algorithm: absent'))
    elif failure == 'yaml':
        (package_dir / 'training_package.yaml').write_text('x: 1\nx: 2\n')
    else:
        entry.unlink()
    output = tmp_path / 'result'
    result = invoke(package_dir, output, tmp_path)
    assert result.returncode != 0
    terminal, = [json.loads(line) for line in result.stdout.splitlines()]
    report = json.loads((output / 'validation-report.json').read_text())
    assert terminal['valid'] is report['valid'] is False
    assert report['resolved_experiment'] is None
    assert report['errors'] and all(set(e) == {'field', 'message'} for e in report['errors'])
    assert {p.name for p in output.iterdir()} == {'validation-report.json', 'user.log'}
    log = (output / 'user.log').read_text()
    assert 'Traceback' in log
    if failure in ('import', 'system_exit'):
        assert 'user exploded' in report['errors'][0]['message']
        assert 'native stdout' in log and 'user exploded' in log


def test_atomic_write_failure_leaves_no_partial_artifact(tmp_path, monkeypatch):
    from muse_runtime import validation

    artifact = tmp_path / 'descriptor.json'
    def fail_publish(source, destination):
        assert not artifact.exists()
        assert Path(source).read_text() == '{"complete": true}'
        raise OSError('publication failed')

    monkeypatch.setattr(validation.os, 'replace', fail_publish)
    with pytest.raises(OSError, match='publication failed'):
        validation._atomic_write(artifact, '{"complete": true}')
    assert list(tmp_path.iterdir()) == []
def test_existing_bytecode_cannot_override_hashed_source(fixture_root, tmp_path):
    import os
    import py_compile
    import shutil
    from muse_sdk.content import hash_package
    from muse_runtime.package_loader import load_package
    package = tmp_path / 'stale-cache'
    shutil.copytree(fixture_root, package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source = package / 'deferred.py'
    py_compile.compile(str(source), doraise=True)
    initial_hash = hash_package(package)
    before = source.stat()
    source.write_text(source.read_text().replace('* 2', '* 5'))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert hash_package(package) != initial_hash
    with load_package(package) as loaded:
        reward = loaded.definition.terms['progress'].implementation(None, None)
        assert reward(3, role_id='scout', scale=1.0) == 15
