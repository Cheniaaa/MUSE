"""协议测试的完整双槽控制；默认只启用本用例关注的Viser。"""
from muse_sdk.run_protocol import DisplaySettings, PreviewSelection, PreviewSelections, WorkerControl


def preview_control(run_id, seq, *, world=0, rate=10., enabled=True):
    return WorkerControl(run_id=run_id, kind='preview', preview=PreviewSelections(
        viser=PreviewSelection(selection_seq=seq,
            display=DisplaySettings(enabled=enabled, world_id=world, rate_hz=rate)),
        native=PreviewSelection(selection_seq=0,
            display=DisplaySettings(enabled=False, world_id=0, rate_hz=10.))))
