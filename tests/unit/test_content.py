"""内容摘要的固定字节证据；期望值经旧入口和手工 framing 独立核对。"""
import pytest


@pytest.mark.parametrize('reverse', [False, True])
def test_content_hash_matches_known_archive_bytes(tmp_path, reverse):
    from muse_sdk.content import hash_package

    # 排序必须基于完整 POSIX 路径，而非目录遍历顺序或路径组件元组。
    files = [
        ('é.txt', b'\x00\xff'),
        ('a/β.bin', b'BETA'),
        ('a.txt', b'A'),
        ('a-1', b'-'),
        ('.data', b'hidden'),
        ('notes.pyc.txt', b'keep'),
    ]
    for name, data in reversed(files) if reverse else files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    expected = '6ee4024fcf75e6edcd05bfcc07380cfcedc5b975d7bf6b405077732628588b67'
    assert hash_package(tmp_path) == expected

    cache = tmp_path / '__pycache__' / 'nested'
    cache.mkdir(parents=True)
    (cache / 'ignored').write_bytes(b'cache')
    (tmp_path / 'ignored.pyc').write_bytes(b'cache')
    assert hash_package(tmp_path) == expected


@pytest.mark.parametrize('name,data,expected', [
    ('a', b'bc', '3fafa1cf2f19a7c1129beb20cf0983f73a489a221fc0dd2f16d1be292d089205'),
    ('ab', b'c', '601d5476e2ccfe2c87a2bba7a322659734a05749d5b5aa781f513e4912db0d5f'),
])
def test_content_hash_distinguishes_filename_and_content_boundaries(tmp_path, name, data, expected):
    from muse_sdk.content import hash_package

    # 不加长度时两组都是 abc；固定摘要锁定路径及内容的8字节大端 framing。
    (tmp_path / name).write_bytes(data)
    assert hash_package(tmp_path) == expected
