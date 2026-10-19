"""通过正式HTTP API提交示例；不创建本地环境或训练循环。"""
import argparse
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8001')
    parser.add_argument('--mode', choices=('TRAIN', 'EVALUATE'), default='TRAIN')
    parser.add_argument('--max-decision-steps', type=int, default=2400,
                        help='本次运行的决策步数上限；每次同时推进所有world，不计内部物理子步')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--experiment-id', help='复用已登记 Experiment，跳过资源上传和解析')
    parser.add_argument('--checkpoint-artifact-id', help='从已登记 checkpoint 恢复；须同时提供 --experiment-id')
    parser.add_argument('--checkpoint-interval-updates', type=int,
                        help='TRAIN 每 N 次成功 update 发布 checkpoint；默认仅终点保存')
    args = parser.parse_args()
    if args.checkpoint_artifact_id and not args.experiment_id:
        parser.error('--checkpoint-artifact-id requires --experiment-id')
    if args.checkpoint_interval_updates is not None:
        if args.checkpoint_interval_updates <= 0:
            parser.error('--checkpoint-interval-updates must be positive')
        if args.mode != 'TRAIN':
            parser.error('--checkpoint-interval-updates requires --mode TRAIN')
    example = Path(__file__).resolve().parent

    def api(method, route, payload=None, binary=False):
        body = payload if binary else (json.dumps(payload).encode() if payload is not None else None)
        request = Request(args.server.rstrip('/') + route, data=body, method=method,
                          headers={'Content-Type': 'application/zip' if binary else 'application/json'})
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    def upload(route, directory):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for path in sorted(directory.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(path, path.relative_to(directory))
        return api('POST', route, buffer.getvalue(), binary=True)

    experiment_id = args.experiment_id
    if experiment_id is None:
        train = args.mode == 'TRAIN'
        environment = upload('/catalog/environments', example / 'environment')
        package = upload('/catalog/training-packages', example / ('training_package_ippo' if train else 'training_package'))
        request = json.loads((example / ('request_train.json' if train else 'request.json')).read_text())
        request.update(environment_id=environment['resource_id'], environment_revision=environment['revision'],
                       training_package_id=package['resource_id'], training_package_revision=package['revision'])
        experiment_id = api('POST', '/experiments', request)['experiment_id']
    payload = dict(experiment_id=experiment_id, mode=args.mode, max_decision_steps=args.max_decision_steps)
    if args.checkpoint_artifact_id is not None:
        payload['checkpoint_artifact_id'] = args.checkpoint_artifact_id
    if args.checkpoint_interval_updates is not None:
        payload['checkpoint_interval_updates'] = args.checkpoint_interval_updates
    run = api('POST', '/runs', payload)
    route = '/runs/' + run['run_id']
    print(json.dumps({'experiment_id': experiment_id, 'run_id': run['run_id'],
                      'run_url': args.server + route}, indent=2), flush=True)
    if args.no_display:
        return
    deadline = time.monotonic() + 360
    requested = False
    while time.monotonic() < deadline:
        run = api('GET', route)
        if run['state'] in ('COMPLETED', 'CANCELLED', 'FAILED', 'LOST'):
            print(json.dumps({'state': run['state'], 'metrics': run['metrics'], 'failure': run['failure']}, indent=2))
            return
        if run['state'] != 'QUEUED':
            if not requested:
                api('PATCH', route + '/display', {'viser': {'enabled': True}})
                requested = True
            display = api('GET', route + '/display')['viser']
            if display['status'] == 'error':
                raise RuntimeError(display['error'])
            if display['status'] == 'ready' and display['viewer_url']:
                print(json.dumps({'viewer_url': display['viewer_url']}, indent=2))
                return
        time.sleep(.1)
    raise TimeoutError('Run did not become display-ready; inspect its API status')


if __name__ == '__main__':
    main()
