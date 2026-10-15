"""本机控制面；上传为application/zip二进制body，不使用multipart。"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import sys
import tempfile

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from muse_sdk.config import ConfigError
from muse_sdk.run_protocol import DisplayPatch, InteractiveCommand, RunRequest, RunTimeouts

from .catalog import Catalog
from .resolver import resolve_experiment
from .supervisor import Supervisor, TERMINAL_STATES, worker_command as default_worker_command


def bootstrap(data_dir, controller_path=None):
    directory = Path(data_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'builtin-metadata.json'
    if not metadata.exists():
        controller = Path(controller_path) if controller_path else Path(__file__).resolve().parents[2] / 'examples/basic_flight/controller.json'
        if not controller.exists() and controller_path is None:
            controller = Path(sys.prefix) / 'share/muse/controller.json'
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            output = Path(temporary) / 'metadata.json'
            subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--controller', str(controller),
                            '--output', str(output)], check=True, capture_output=True, text=True)
            return Catalog(directory, output)
    return Catalog(directory, metadata)


def create_app(data_dir, *, controller_path=None, timeouts=None, worker_command=default_worker_command,
               access_host='127.0.0.1', viewer_port=8080):
    limits = timeouts or RunTimeouts()

    @asynccontextmanager
    async def lifespan(app):
        catalog = bootstrap(data_dir, controller_path)
        supervisor = Supervisor(catalog, worker_command=worker_command, access_host=access_host, viewer_port=viewer_port)
        app.state.catalog = catalog
        app.state.supervisor = supervisor
        supervisor.start()
        try:
            yield
        finally:
            await asyncio.to_thread(supervisor.close)

    app = FastAPI(title='Muse M1', lifespan=lifespan)

    @app.exception_handler(ConfigError)
    async def config_error(request, exc):
        status = 404 if any(e['message'].startswith('Unknown experiment') for e in exc.errors) else 422
        return JSONResponse(status_code=status, content={'errors': exc.errors})

    @app.exception_handler(KeyError)
    async def missing_record(request, exc):
        return JSONResponse(status_code=404, content={'detail': 'Unknown resource'})

    @app.get('/health')
    def health():
        error = app.state.supervisor.error
        return JSONResponse(status_code=503 if error else 200,
                            content={'status': 'error' if error else 'ok', 'error': error,
                                     'gpu_busy': app.state.supervisor.gpu_busy})

    @app.get('/catalog/environments')
    def environments():
        return [r for r in app.state.catalog.snapshot().resources if r.kind == 'environment']

    @app.get('/catalog/training-packages')
    def training_packages():
        return [r for r in app.state.catalog.snapshot().resources if r.kind == 'training_package']

    async def upload(request, kind):
        if request.headers.get('content-type', '').split(';')[0] != 'application/zip':
            raise HTTPException(415, 'Expected application/zip binary body (not multipart)')
        content = await request.body()
        def register():
            with tempfile.TemporaryDirectory(dir=app.state.catalog.directory) as temporary:
                path = Path(temporary) / 'upload.zip'
                path.write_bytes(content)
                try:
                    return app.state.catalog.register_archive(kind, path)
                except ConfigError:
                    raise
                except (ValueError, FileNotFoundError) as exc:
                    raise ConfigError('archive', str(exc)) from exc
        return await asyncio.to_thread(register)

    @app.post('/catalog/environments', status_code=201,
              openapi_extra={'requestBody': {'required': True, 'content': {'application/zip': {
                  'schema': {'type': 'string', 'format': 'binary'}}}}})
    async def register_environment(request: Request):
        return await upload(request, 'environment')

    @app.post('/catalog/training-packages', status_code=201,
              openapi_extra={'requestBody': {'required': True, 'content': {'application/zip': {
                  'schema': {'type': 'string', 'format': 'binary'}}}}})
    async def register_training_package(request: Request):
        return await upload(request, 'training_package')

    @app.get('/catalog/entity-models')
    def entity_models():
        return app.state.catalog.builtins.entity_models

    @app.get('/catalog/controllers')
    def controllers():
        return app.state.catalog.builtins.controllers

    @app.get('/catalog/sensors')
    def sensors():
        return app.state.catalog.builtins.sensors

    @app.post('/experiments/validate')
    def validate_experiment(request: dict):
        return resolve_experiment(request, app.state.catalog.snapshot())

    @app.post('/experiments', status_code=201)
    def save_experiment(request: dict):
        report = resolve_experiment(request, app.state.catalog.snapshot())
        if not report.valid:
            return JSONResponse(status_code=422, content=report.model_dump(mode='json'))
        return {'experiment_id': app.state.catalog.save_experiment(report.resolved_experiment)}

    @app.get('/experiments/{experiment_id}')
    def get_experiment(experiment_id: str):
        return app.state.catalog.get_experiment(experiment_id)

    @app.post('/runs', status_code=201)
    def create_run(request: RunRequest):
        plan = app.state.catalog.create_run_plan(**request.model_dump(), device='cuda:0', timeouts=limits)
        return app.state.supervisor.submit(plan)

    @app.get('/runs')
    def list_runs():
        return app.state.supervisor.list_runs()

    @app.get('/runs/{run_id}')
    def get_run(run_id: str):
        return app.state.supervisor.get_run(run_id)

    @app.post('/runs/{run_id}/cancel')
    def cancel(run_id: str):
        return app.state.supervisor.cancel(run_id)

    @app.post('/runs/{run_id}/commands')
    def command(run_id: str, request: InteractiveCommand):
        try:
            return app.state.supervisor.submit_command(run_id, request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/runs/{run_id}/display')
    def get_display(run_id: str):
        return app.state.supervisor.get_display(run_id)

    @app.patch('/runs/{run_id}/display')
    def patch_display(run_id: str, request: DisplayPatch):
        try:
            return app.state.supervisor.patch_display(run_id, request.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/runs/{run_id}/artifacts')
    def artifacts(run_id: str):
        return app.state.supervisor.artifacts(run_id)

    @app.get('/runs/{run_id}/artifacts/{artifact_id}')
    def artifact_file(run_id: str, artifact_id: str):
        supervisor = app.state.supervisor
        artifact = next((a for a in supervisor.artifacts(run_id) if a['id'] == artifact_id), None)
        if artifact is None:
            raise HTTPException(404, 'Unknown published artifact')
        root = Path(supervisor.get_run(run_id)['artifact_directory']).resolve()
        path = (root / artifact['path']).resolve()
        if artifact['run'] != run_id or not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, 'Artifact unavailable or outside run directory')
        return FileResponse(path, filename=path.name)

    @app.websocket('/streams/runs/{run_id}/events')
    async def events(websocket: WebSocket, run_id: str):
        supervisor = app.state.supervisor
        try:
            supervisor.get_run(run_id)
        except KeyError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        seq = 0
        try:
            while True:
                for event in supervisor.events(run_id, seq):
                    await websocket.send_json(event)
                    seq = event['event_seq']
                    if event['kind'] == 'terminal':
                        await websocket.close()
                        return
                # 主动接收disconnect，同时不阻塞Supervisor读取Worker。
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=.05)
                    if message['type'] == 'websocket.disconnect':
                        return
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            return

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--access-host', default='127.0.0.1')
    parser.add_argument('--viewer-port', type=int, default=8080)
    for name, field in RunTimeouts.model_fields.items():
        parser.add_argument('--' + name.replace('_', '-') + '-seconds', type=float, default=field.default)
    args = parser.parse_args()
    limits = RunTimeouts(**{name: getattr(args, name + '_seconds') for name in RunTimeouts.model_fields})
    import uvicorn
    uvicorn.run(create_app(args.data_dir, controller_path=args.controller, timeouts=limits,
                          access_host=args.access_host, viewer_port=args.viewer_port),
                host=args.host, port=args.port)


if __name__ == '__main__':
    main()
