"""CPU-only Run控制与事件线格式。"""
from typing import Literal
import socket
import struct

from pydantic import Field, field_validator, model_validator

from .config import DataModel, check_json
from .recording_protocol import RecordingConfig


class RunTimeouts(DataModel):
    startup: float = Field(default=60., gt=0)
    compile: float = Field(default=300., gt=0)
    heartbeat: float = Field(default=30., gt=0)
    heartbeat_interval: float = Field(default=1., gt=0)
    cancel: float = Field(default=5., gt=0)
    stop: float = Field(default=5., gt=0)

    @model_validator(mode='after')
    def heartbeat_deadline(self):
        if self.heartbeat <= self.heartbeat_interval:
            raise ValueError('heartbeat timeout must exceed heartbeat_interval')
        return self


class InteractiveCommand(DataModel):
    command_seq: int = Field(gt=0, strict=True)
    command: Literal['pause', 'resume', 'step', 'reset_world']
    world_id: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode='after')
    def command_target(self):
        if (self.command == 'reset_world') != (self.world_id is not None):
            raise ValueError('Only reset_world requires world_id; other commands forbid it')
        return self


class CommandResult(InteractiveCommand):
    status: Literal['applied', 'rejected']
    paused: bool
    steps: int
    updates: int
    error: str | None = None


class WorkerEvent(DataModel):
    event_seq: int = Field(gt=0)
    run_id: str
    kind: Literal['preparing', 'compiling', 'running', 'heartbeat', 'metric', 'failure', 'terminal',
                  'preview_model', 'preview_applied', 'preview_error', 'checkpoint', 'command_result',
                  'recording_state', 'trajectory']
    sim_time: float = Field(ge=0)
    wall_time: float = Field(ge=0)
    payload: dict = Field(default_factory=dict)

    @field_validator('payload')
    @classmethod
    def finite_json_payload(cls, value):
        check_json(value, 'payload')
        return value


class WorkerControl(DataModel):
    run_id: str
    kind: Literal['cancel', 'preview', 'command']
    command: InteractiveCommand | None = None
    preview: 'PreviewSelections | None' = None

    @model_validator(mode='after')
    def preview_fields(self):
        if (self.kind == 'command') != (self.command is not None):
            raise ValueError('Only command controls require command')
        if (self.kind == 'preview') != (self.preview is not None):
            raise ValueError('Only preview controls require preview selections')
        return self


class RunRequest(DataModel):
    experiment_id: str
    mode: Literal['TRAIN', 'EVALUATE', 'INTERACTIVE']
    max_decision_steps: int = Field(gt=0, description='本次运行的决策步数上限；一次同时推进全部world，不乘world数，不计内部物理子步')
    checkpoint_artifact_id: str | None = None
    checkpoint_interval_updates: int | None = Field(default=None, gt=0)
    recording: RecordingConfig | None = None

    @model_validator(mode='after')
    def saving_requires_training(self):
        if self.mode != 'TRAIN' and self.checkpoint_interval_updates is not None:
            raise ValueError('checkpoint_interval_updates requires TRAIN')
        return self


class CheckpointReference(DataModel):
    artifact_id: str
    path: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class CheckpointArtifact(DataModel):
    id: str
    run: str
    kind: Literal['checkpoint'] = 'checkpoint'
    path: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class DisplaySettings(DataModel):
    enabled: bool
    world_id: int = Field(ge=0)
    rate_hz: float = Field(ge=1, le=30)


ViewerName = Literal['viser', 'native']


class ViewerPatch(DataModel):
    enabled: bool | None = None
    world_id: int | None = Field(default=None, ge=0)
    rate_hz: float | None = Field(default=None, ge=1, le=30)

    @model_validator(mode='after')
    def no_explicit_null(self):
        if any(getattr(self, key) is None for key in self.model_fields_set):
            raise ValueError('Display fields cannot be null')
        return self


class DisplayPatch(DataModel):
    viser: ViewerPatch | None = None
    native: ViewerPatch | None = None

    @model_validator(mode='after')
    def one_viewer(self):
        if len(self.model_fields_set) != 1 or getattr(self, next(iter(self.model_fields_set))) is None:
            raise ValueError('Display patch requires exactly one non-null viewer')
        return self


class PreviewSelection(DataModel):
    selection_seq: int = Field(ge=0)
    display: DisplaySettings


class PreviewSelections(DataModel):
    viser: PreviewSelection
    native: PreviewSelection


class SceneObject(DataModel):
    """草稿与 Run 共用的静态身份；原生映射仅供内部显示消费者。"""
    object_id: str
    label: str
    kind: Literal['environment', 'entity']
    group_id: str | None
    group_slot: int | None
    role_id: str | None
    model_id: str | None
    body_id: int | None
    geom_ids: list[int] | None
    # 环境 geom 相对所属 body 的冷映射；矩阵为 row-major 9 元素，entity root 不需要。
    local_position: list[float] | None = None
    local_orientation_matrix: list[float] | None = None


class PreviewModel(DataModel):
    path: str
    sha256: str = Field(pattern='^[0-9a-f]{64}$')
    nbody: int = Field(gt=0)
    native_payload_floats: int = Field(gt=0)
    objects: list[SceneObject]


class PreviewApplied(DataModel):
    viewer: ViewerName
    selection_seq: int = Field(gt=0)
    display: DisplaySettings


class PreviewFrame(DataModel):
    viewer: ViewerName
    run_id: str
    world_id: int = Field(ge=0)
    selection_seq: int = Field(gt=0)
    episode: int = Field(ge=0)
    decision: int = Field(ge=0)
    tick: int = Field(ge=0)
    sim_time: float = Field(ge=0)
    scene_hash: str = Field(pattern='^[0-9a-f]{64}$')
    nbody: int = Field(gt=0)
    payload_floats: int = Field(gt=0)

    @model_validator(mode='after')
    def viewer_layout(self):
        if self.viewer == 'viser' and self.payload_floats != self.nbody * 12:
            raise ValueError('Viser payload must contain body xpos and xmat')
        return self


WorkerControl.model_rebuild()

# 两段preview共享包边界；Viser为body pose，Native为固定状态，均为little-endian float32。
PREVIEW_CAPACITY = 128 * 1024
PREVIEW_HEADER_CAPACITY = 4096


def native_state_layout(*, nq: int, nv: int, na: int, nu: int, nbody: int,
                        nmocap: int, neq: int, nsensordata: int) -> tuple[tuple[str, int], ...]:
    """Native固定状态布局；尺寸来自本Run编译模型。"""
    return (('qpos', nq), ('qvel', nv), ('act', na), ('ctrl', nu),
            ('qfrc_applied', nv), ('xfrc_applied', 6 * nbody),
            ('mocap_pos', 3 * nmocap), ('mocap_quat', 4 * nmocap),
            ('eq_active', neq), ('sensordata', nsensordata))


def preview_socketpair():
    pair = socket.socketpair(type=socket.SOCK_SEQPACKET)
    for sock in pair:
        sock.setblocking(False)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, PREVIEW_CAPACITY * 2)
    return pair


def check_preview_capacity(sock: socket.socket, *, payload_floats: int) -> None:
    size = 4 + PREVIEW_HEADER_CAPACITY + payload_floats * 4
    if size > min(PREVIEW_CAPACITY, sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) - 32):
        raise ValueError('Scene exceeds preview packet capacity')


def encode_preview(header, arrays):
    metadata = header.model_dump_json().encode()
    if len(metadata) > PREVIEW_HEADER_CAPACITY or len(arrays) != header.payload_floats * 4:
        raise ValueError('Invalid preview packet size')
    packet = struct.pack('<I', len(metadata)) + metadata + arrays
    if len(packet) > PREVIEW_CAPACITY:
        raise ValueError('Scene exceeds preview packet capacity')
    return packet


def decode_preview(packet):
    if len(packet) > PREVIEW_CAPACITY:
        raise ValueError('Scene exceeds preview packet capacity')
    if len(packet) < 4:
        raise ValueError('truncated preview header')
    size, = struct.unpack('<I', packet[:4])
    if size > PREVIEW_HEADER_CAPACITY or len(packet) < size + 4:
        raise ValueError('truncated preview header')
    header = PreviewFrame.model_validate_json(packet[4:4 + size])
    arrays = packet[4 + size:]
    if len(arrays) != header.payload_floats * 4:
        raise ValueError('truncated preview arrays')
    return header, arrays


def send_preview(sock, pending):
    if pending is not None:
        try:
            sock.send(pending)
        except BlockingIOError:
            return pending
    return None


def drain_preview(sock, *, run_id, selections: dict[ViewerName, PreviewSelection], scene_hash,
                  capacity=PREVIEW_CAPACITY):
    latest = {}
    while True:
        try:
            packet, _, flags, _ = sock.recvmsg(capacity)
        except BlockingIOError:
            return latest
        if not packet:
            return latest
        if flags & socket.MSG_TRUNC:
            raise ValueError('truncated preview packet')
        header, arrays = decode_preview(packet)
        selection = selections.get(header.viewer)
        if selection is None or not selection.display.enabled:
            continue
        if (header.run_id, header.selection_seq, header.world_id, header.scene_hash) != (
                run_id, selection.selection_seq, selection.display.world_id, scene_hash):
            continue
        previous = latest.get(header.viewer)
        if previous is None or (header.episode, header.decision) > (previous[0].episode, previous[0].decision):
            latest[header.viewer] = header, arrays
