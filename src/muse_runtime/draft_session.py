"""单所有者的长期 CPU 草稿会话；stdin JSONL，stdout 专用协议 fd。"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view-id', required=True)
    args = parser.parse_args()
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    protocol_lock = threading.Lock()

    def emit(message):
        encoded = (json.dumps(message, allow_nan=False) + '\n').encode('utf-8')
        with protocol_lock:
            while encoded:
                encoded = encoded[os.write(protocol_fd, encoded):]

    server, interaction = None, None
    try:
        import mujoco
        import viser
        from muse_sdk.view_protocol import SceneObject, ViewIntent
        from muse_runtime.scene_preview import compile_preview
        from muse_server.viewers.scene_interaction import SceneInteraction

        for line in sys.stdin:
            message = json.loads(line)
            if message['kind'] == 'close':
                break
            if message['kind'] == 'intent':
                try:
                    if interaction is None:
                        raise ValueError('No compiled view is ready')
                    interaction.apply_intent(ViewIntent.model_validate(message['intent']))
                except ValueError as exc:
                    emit({'kind': 'view', 'event': {'kind': 'error', 'message': str(exc)}})
                continue
            if message['kind'] != 'compile':
                raise ValueError(f"Unknown draft message: {message['kind']}")
            request = message['request']
            candidate_server, candidate_interaction = None, None
            try:
                result = compile_preview(request, message['resource_paths'], Path(message['destination']))
                model = mujoco.MjModel.from_binary_path(result['model_path'])
                candidate_server = viser.ViserServer(host='127.0.0.1', port=0, verbose=False)
                candidate_interaction = SceneInteraction(candidate_server, model,
                    [SceneObject.model_validate(obj) for obj in result['objects']], view_id=args.view_id,
                    emit=emit, revision=result['applied_revision'], editable=set(result['editable']))
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                candidate_interaction.update_frame(data.xpos, data.xmat.reshape(-1, 3, 3),
                                                     {'revision': result['applied_revision']})
            except Exception as exc:
                if candidate_interaction is not None:
                    candidate_interaction.close()
                if candidate_server is not None:
                    candidate_server.stop()
                traceback.print_exc()
                emit({'kind': 'error', 'revision': request['revision'], 'message': str(exc)})
                continue
            # 固定版没有公开注销 connect callbacks：用 server 生命周期替换，避免私有列表操作。
            if interaction is not None:
                interaction.close()
            if server is not None:
                server.stop()
            server, interaction = candidate_server, candidate_interaction
            emit({'kind': 'ready', 'revision': result['applied_revision'],
                'viewer_url': f'http://{server.get_host()}:{server.get_port()}',
                'objects': result['objects']})
        return 0
    finally:
        if interaction is not None:
            interaction.close()
        if server is not None:
            server.stop()
        with protocol_lock:
            os.close(protocol_fd)


if __name__ == '__main__':
    raise SystemExit(main())
