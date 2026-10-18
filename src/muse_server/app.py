"""本机控制面；上传为application/zip二进制body，不使用multipart。"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from muse_sdk.config import ConfigError
from muse_sdk.experiment import ExperimentRequest, ScenarioDefinition
from muse_sdk.scene_edit import EnvironmentEdit, ScenePreviewRequest
from muse_sdk.run_protocol import DisplayPatch, InteractiveCommand, RunRequest, RunTimeouts

from .catalog import API_REVISION, Catalog
from .resolver import resolve_experiment
from .inspection import inspect_experiment
from .run_data import read_log_page, read_metric_page
from .previews import PreviewManager
from .storage import write_json_new
from .supervisor import Supervisor, TERMINAL_STATES, process_identity, worker_command as default_worker_command


PageOffset = Annotated[int, Query(ge=0)]
PageLimit = Annotated[int, Query(ge=1, le=200)]
LogChannel = Literal['worker', 'viser']


def _catalog_page(items, offset, limit):
    return {'items': items[offset:offset + limit],
            'next_offset': offset + limit if offset + limit < len(items) else None}


def _run_log_page(supervisor, run_id, channel, offset, limit, *, follow=False):
    record = supervisor.get_run(run_id)
    terminal = record['state'] in TERMINAL_STATES
    started = record['pid'] is not None if channel == 'worker' else record['viser_log_started']
    if not started:
        return dict(data_base64='', next_offset=0 if offset is None else offset, eof=True,
                    availability='not_started', terminal=terminal)
    directory = Path(record['artifact_directory'])
    gap = None
    if follow:
        size = (directory / 'logs' / f'{channel}.log').stat().st_size
        if offset is None:
            offset = size
        elif size - offset > 262144:
            gap = dict(previous_offset=offset, next_offset=size - 262144)
            offset = gap['next_offset']
    page = read_log_page(directory, channel=channel, offset=offset, limit=limit)
    page['terminal'] = terminal
    if gap is not None:
        page['gap'] = gap
    return page


def bootstrap(data_dir, controller_path=None):
    directory = Path(data_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'builtin-metadata.json'
    if not metadata.exists():
        controller = Path(controller_path) if controller_path else Path(__file__).resolve().parents[2] / 'examples/basic_flight/controller.json'
        if not controller.exists() and controller_path is None:
            controller = Path(sys.prefix) / 'share/muse/controller.json'
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'metadata.json'
            subprocess.run([sys.executable, '-m', 'muse_builtins.metadata', '--controller', str(controller),
                            '--output', str(output)], check=True, capture_output=True, text=True)
            return Catalog(directory, output)
    return Catalog(directory, metadata)


def create_app(data_dir, *, controller_path=None, timeouts=None, worker_command=default_worker_command,
               access_host='127.0.0.1', viewer_port=8080, server_address=None):
    limits = timeouts or RunTimeouts()

    @asynccontextmanager
    async def lifespan(app):
        catalog = bootstrap(data_dir, controller_path)
        supervisor = Supervisor(catalog, worker_command=worker_command, access_host=access_host, viewer_port=viewer_port)
        app.state.catalog = catalog
        app.state.supervisor = supervisor
        app.state.instance_id = uuid4().hex
        supervisor.start()
        app.state.previews = PreviewManager(catalog)
        instance_path = catalog.directory / 'server-instance.json'
        published = False
        try:
            instance = dict(instance_id=app.state.instance_id, pid=os.getpid(), process_start=process_identity(os.getpid()),
                            address=server_address, state='running', shutdown_timeout=limits.cancel + limits.stop + 5.)
            # 目录锁已取得；替换崩溃遗留身份，status只会看到完整旧记录或完整新记录。
            with tempfile.TemporaryDirectory(dir=catalog.directory) as temporary:
                path = Path(temporary) / 'instance.json'
                write_json_new(path, json.dumps(instance))
                path.replace(instance_path)
                published = True
            yield
        finally:
            # 删除身份必须在清理完成之后、目录锁释放之前，避免删掉下一实例的文件。
            previews_closed = False
            try:
                await asyncio.to_thread(app.state.previews.shutdown)
                previews_closed = True
            finally:
                await asyncio.to_thread(supervisor.close,
                    before_unlock=instance_path.unlink if published and previews_closed else None)

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
                                     'gpu_busy': app.state.supervisor.gpu_busy,
                                     'api_revision': API_REVISION, 'catalog_id': app.state.catalog.catalog_id,
                                     'platform_revision': app.state.catalog.builtins.platform_revision,
                                     'instance_id': app.state.instance_id})

    @app.get('/catalog/environments')
    def environments(offset: PageOffset = 0, limit: PageLimit = 50):
        return app.state.catalog.list_resources('environment', offset=offset, limit=limit)

    @app.get('/catalog/schema')
    def catalog_schema():
        return {'api_revision': API_REVISION, 'experiment_request': ExperimentRequest.model_json_schema(),
                'scenario': ScenarioDefinition.model_json_schema(), 'run_request': RunRequest.model_json_schema(),
                'environment_edit': EnvironmentEdit.model_json_schema(),
                'preview_request': ScenePreviewRequest.model_json_schema()}

    @app.get('/catalog/environments/{resource_id}/{revision}/editing')
    def environment_editing(resource_id: str, revision: int):
        return app.state.catalog.environment_editing(resource_id, revision)

    @app.post('/catalog/environments/derive', status_code=201)
    def derive_environment(edit: EnvironmentEdit):
        return app.state.catalog.derive_environment(edit)

    @app.post('/previews', status_code=201)
    def create_preview(request: ScenePreviewRequest):
        return app.state.previews.create(request)

    @app.put('/previews/{session_id}')
    def update_preview(session_id: str, request: ScenePreviewRequest):
        return app.state.previews.update(session_id, request)

    @app.get('/previews/{session_id}')
    def get_preview(session_id: str):
        return app.state.previews.get(session_id)

    @app.delete('/previews/{session_id}', status_code=204)
    def close_preview(session_id: str):
        app.state.previews.close(session_id)

    @app.websocket('/streams/previews/{session_id}')
    async def preview_control(websocket: WebSocket, session_id: str):
        previews = app.state.previews
        try:
            previews.attach(session_id)
        except (KeyError, ConfigError):
            await websocket.close(code=1008)
            return
        previous = None
        try:
            await websocket.accept()
            while True:
                state = previews.get(session_id)
                if state != previous:
                    await websocket.send_json({'kind': 'state', **state})
                    previous = state
                for event in previews.events(session_id):
                    await websocket.send_json(event)
                try:
                    intent = await asyncio.wait_for(websocket.receive_json(), timeout=.05)
                    try:
                        current = previews.view_intent(session_id, intent)
                        if current is not None:
                            await websocket.send_json({'kind': 'state', **current})
                    except ValueError as exc:
                        await websocket.send_json({'kind': 'error', 'message': str(exc)})
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect, KeyError):
            return
        finally:
            await asyncio.to_thread(previews.close, session_id)

    @app.get('/catalog/networks')
    def networks(offset: PageOffset = 0, limit: PageLimit = 50):
        descriptor = app.state.catalog.builtins.descriptor
        items = [dict(kind=kind, reference='builtin:' + identity, descriptor=entries[identity])
                 for kind, entries in (('radio', descriptor.radio_models), ('sensing', descriptor.sensing_models))
                 for identity in sorted(entries)]
        return _catalog_page(items, offset, limit)

    @app.get('/catalog/training-packages')
    def training_packages(offset: PageOffset = 0, limit: PageLimit = 50):
        return app.state.catalog.list_resources('training_package', offset=offset, limit=limit)

    async def upload(request, kind, resource_id):
        if request.headers.get('content-type', '').split(';')[0] != 'application/zip':
            raise HTTPException(415, 'Expected application/zip binary body (not multipart)')
        content = await request.body()
        def register():
            with tempfile.TemporaryDirectory(dir=app.state.catalog.directory) as temporary:
                path = Path(temporary) / 'upload.zip'
                path.write_bytes(content)
                try:
                    return app.state.catalog.register_archive(kind, path, resource_id=resource_id)
                except ConfigError:
                    raise
                except (ValueError, FileNotFoundError) as exc:
                    raise ConfigError('archive', str(exc)) from exc
        return await asyncio.to_thread(register)

    @app.post('/catalog/environments', status_code=201,
              openapi_extra={'requestBody': {'required': True, 'content': {'application/zip': {
                  'schema': {'type': 'string', 'format': 'binary'}}}}})
    async def register_environment(request: Request, resource_id: str | None = None):
        return await upload(request, 'environment', resource_id)

    @app.post('/catalog/training-packages', status_code=201,
              openapi_extra={'requestBody': {'required': True, 'content': {'application/zip': {
                  'schema': {'type': 'string', 'format': 'binary'}}}}})
    async def register_training_package(request: Request, resource_id: str | None = None):
        return await upload(request, 'training_package', resource_id)

    @app.get('/catalog/entity-models')
    def entity_models(offset: PageOffset = 0, limit: PageLimit = 50):
        entries = app.state.catalog.builtins.entity_models
        return _catalog_page([entries[key] for key in sorted(entries)], offset, limit)

    @app.get('/catalog/controllers')
    def controllers(offset: PageOffset = 0, limit: PageLimit = 50):
        entries = app.state.catalog.builtins.controllers
        return _catalog_page([entries[key] for key in sorted(entries)], offset, limit)

    @app.get('/catalog/sensors')
    def sensors(offset: PageOffset = 0, limit: PageLimit = 50):
        entries = app.state.catalog.builtins.sensors
        return _catalog_page([entries[key] for key in sorted(entries)], offset, limit)

    @app.get('/catalog/examples')
    def examples(offset: PageOffset = 0, limit: PageLimit = 50):
        from .examples import list_examples
        return _catalog_page(list_examples(), offset, limit)

    @app.post('/catalog/examples/{example_id}/import', status_code=201)
    def import_template(example_id: str):
        from .examples import import_example
        return import_example(app.state.catalog, example_id)

    @app.post('/experiments/validate')
    def validate_experiment(request: dict):
        return resolve_experiment(request, app.state.catalog.snapshot())

    @app.post('/experiments/inspect')
    def inspect_draft(request: dict):
        return inspect_experiment(request, app.state.catalog.snapshot())

    @app.post('/experiments', status_code=201)
    def save_experiment(request: dict, label: str = ''):
        report = resolve_experiment(request, app.state.catalog.snapshot())
        if not report.valid:
            return JSONResponse(status_code=422, content=report.model_dump(mode='json'))
        return {'experiment_id': app.state.catalog.save_experiment(report.resolved_experiment,
                                                                  source=request, label=label)}

    @app.get('/experiments')
    def list_experiments(offset: PageOffset = 0, limit: PageLimit = 50):
        return app.state.catalog.list_experiments(offset=offset, limit=limit)

    @app.get('/experiments/{experiment_id}/source')
    def get_experiment_source(experiment_id: str):
        return app.state.catalog.get_experiment_source(experiment_id)

    @app.get('/experiments/{experiment_id}')
    def get_experiment(experiment_id: str):
        return app.state.catalog.get_experiment(experiment_id)

    @app.post('/runs', status_code=201)
    def create_run(request: RunRequest):
        plan = app.state.catalog.create_run_plan(**request.model_dump(), device='cuda:0', timeouts=limits)
        return app.state.supervisor.submit(plan)

    @app.get('/runs')
    def list_runs(experiment_id: str | None = None, mode: Literal['TRAIN', 'EVALUATE', 'INTERACTIVE'] | None = None,
                  state: Literal['QUEUED', 'PREPARING', 'COMPILING', 'RUNNING', 'STOP_REQUESTED', 'STOPPING',
                                 'COMPLETED', 'CANCELLED', 'FAILED', 'LOST'] | None = None,
                  offset: PageOffset = 0, limit: PageLimit = 50):
        return app.state.supervisor.list_runs(experiment_id=experiment_id, mode=mode, state=state,
                                             offset=offset, limit=limit)

    @app.get('/runs/{run_id}')
    def get_run(run_id: str):
        return app.state.supervisor.get_run(run_id)

    @app.get('/runs/{run_id}/metrics')
    def metrics(run_id: str, offset: PageOffset = 0, limit: Annotated[int, Query(ge=1, le=2000)] = 500):
        record = app.state.supervisor.get_run(run_id)
        terminal = record['state'] in TERMINAL_STATES
        if record['pid'] is None:
            return dict(records=[], next_offset=offset, eof=True, terminal=terminal, availability='not_started')
        return read_metric_page(Path(record['artifact_directory']), offset=offset, limit=limit, terminal=terminal)

    @app.post('/runs/{run_id}/display/clients')
    def register_display_client(run_id: str):
        return app.state.supervisor.register_view_client(run_id)

    @app.delete('/runs/{run_id}/display/clients/{view_id}', status_code=204)
    def remove_display_client(run_id: str, view_id: str):
        app.state.supervisor.remove_view_client(run_id, view_id)

    @app.get('/runs/{run_id}/entities')
    def entities(run_id: str):
        return app.state.supervisor.entities(run_id)

    @app.get('/runs/{run_id}/logs')
    def logs(run_id: str, channel: LogChannel = 'worker', offset: PageOffset = 0,
             limit: Annotated[int, Query(ge=1, le=262144)] = 65536):
        return _run_log_page(app.state.supervisor, run_id, channel, offset, limit)

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

    @app.websocket('/streams/runs/{run_id}/logs')
    async def log_stream(websocket: WebSocket, run_id: str, channel: LogChannel = 'worker'):
        supervisor = app.state.supervisor
        try:
            supervisor.get_run(run_id)
        except KeyError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        offset = None
        try:
            while True:
                # 只取快照时短暂持锁；文件读取和慢网络发送均在锁外。
                page = await asyncio.to_thread(_run_log_page, supervisor, run_id, channel, offset, 65536, follow=True)
                await websocket.send_json(page)
                offset = page['next_offset']
                if page['terminal'] and page['eof']:
                    await websocket.close()
                    return
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=.1)
                    if message['type'] == 'websocket.disconnect':
                        return
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            return

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

    @app.websocket('/streams/runs/{run_id}/view')
    async def view_stream(websocket: WebSocket, run_id: str, view_id: str):
        supervisor = app.state.supervisor
        try:
            initial = supervisor.view_events(run_id, view_id)
        except (KeyError, ValueError):
            await websocket.close(code=1008)
            return
        try:
            await websocket.accept()
            for event in initial:
                await websocket.send_json(event)
            while True:
                for event in supervisor.view_events(run_id, view_id):
                    await websocket.send_json(event)
                try:
                    intent = await asyncio.wait_for(websocket.receive_json(), timeout=.05)
                    try:
                        supervisor.view_intent(run_id, view_id, intent)
                    except ValueError as exc:
                        await websocket.send_json({'kind': 'error', 'view_id': view_id, 'message': str(exc)})
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            return
        finally:
            supervisor.remove_view_client(run_id, view_id)

    @app.websocket('/streams/runs/{run_id}/entities')
    async def entity_stream(websocket: WebSocket, run_id: str):
        try:
            app.state.supervisor.get_run(run_id)
        except KeyError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        previous = None
        try:
            while True:
                current = app.state.supervisor.entities(run_id)
                if current != previous:
                    await websocket.send_json(current)
                    previous = current
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=.1)
                    if message['type'] == 'websocket.disconnect':
                        return
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            return

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--access-host', default='127.0.0.1')
    parser.add_argument('--viewer-port', type=int, default=8080)
    for name, field in RunTimeouts.model_fields.items():
        parser.add_argument('--' + name.replace('_', '-') + '-seconds', type=float, default=field.default)
    args = parser.parse_args(argv)
    limits = RunTimeouts(**{name: getattr(args, name + '_seconds') for name in RunTimeouts.model_fields})
    import uvicorn
    uvicorn.run(create_app(args.data_dir, controller_path=args.controller, timeouts=limits,
                          access_host=args.access_host, viewer_port=args.viewer_port,
                          server_address=f'http://{args.access_host}:{args.port}'),
                host=args.host, port=args.port)


if __name__ == '__main__':
    main()
