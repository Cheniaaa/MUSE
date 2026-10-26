"""Server组合边界：独立owner清理、真实查找缺失与内部错误归属。"""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failures', [(), ('replays',), ('previews',), ('supervisor',),
                                    ('replays', 'previews', 'supervisor')])
def test_lifespan_attempts_every_owner_and_preserves_failure_identity(tmp_path, monkeypatch, failures):
    from muse_server import app as module

    calls = []
    catalog = SimpleNamespace(directory=tmp_path)
    monkeypatch.setattr(module, 'bootstrap', lambda *args: catalog)

    def shutdown(owner):
        calls.append(owner)
        if owner in failures:
            raise RuntimeError(owner + ' shutdown failed')

    class Supervisor:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def close(self, *, before_unlock):
            shutdown('supervisor')
            if before_unlock is not None:
                before_unlock()

    monkeypatch.setattr(module, 'Supervisor', Supervisor)
    monkeypatch.setattr(module, 'ReplayManager', lambda *args:
        SimpleNamespace(shutdown=lambda: shutdown('replays')))
    monkeypatch.setattr(module, 'PreviewManager', lambda *args:
        SimpleNamespace(shutdown=lambda: shutdown('previews')))
    app = module.create_app(tmp_path)

    async def enter_and_leave():
        async with app.router.lifespan_context(app):
            assert (tmp_path / 'server-instance.json').is_file()

    if failures:
        with pytest.raises(RuntimeError) as raised:
            asyncio.run(enter_and_leave())
        messages, error = [], raised.value
        while error is not None:
            messages.append(str(error))
            error = error.__context__
        assert set(messages) == {owner + ' shutdown failed' for owner in failures}
    else:
        asyncio.run(enter_and_leave())
    assert calls == ['replays', 'previews', 'supervisor']
    assert (tmp_path / 'server-instance.json').exists() == bool(failures)
