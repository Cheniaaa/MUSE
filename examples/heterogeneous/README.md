# UAV + UGV 异构 IPPO 示例

每个 world 包含一架 scout Crazyflie 和一辆 rover 官方 car，B=2，共享同一物理与 reset 边界。scout 使用 body_velocity 四维动作与 17 维观测，rover 使用 diff_drive 二维动作与 9 维观测。`collection_steps=4`，8 个 decision steps 对应两个学习 block；timeout 为每个 world 8 步，两个 role 分别绑定独立 collision selection。

本示例已通过实际混合物理、学习更新和 HTTP checkpoint 恢复验收，结果见 [M2.1 验收报告](../../docs/validation/m2-heterogeneous.md)。UGV当前验收范围为静态水平地面；转向存在滑移误差，不作为实车标定结果。

## 动作和观测

rover motion 为 `[v_forward, yaw_rate]`，units 为 `[m/s, rad/s]`，frames 为 `[body_flu, world_heading]`，范围为 ±.1 m/s、±.8 rad/s。`diff_drive_controller` 的轮半径 .03 m、轮距 .12 m、阻尼 .03、轮速比例增益 .01 由 builtin metadata 固定。原 car 的 motor ±1 与 joint actuator force ±.5 保留。

| rover selection | 输出维度 | 含义与 frame |
|---|---:|---|
| target | 2 | target-position 的 world xy，m，world_z_up |
| velocity | 2 | world xy 速度，m/s，world_z_up |
| heading | 2 | `[cos(yaw), sin(yaw)]`，读取 orientation_wxyz，world_z_up |
| yaw_rate | 1 | body angular velocity 的 z 分量，rad/s，body_flu；适用当前平面场景 |
| previous_action | 2 | 上一步 diff_drive motion，mixed units/frame |

rover root placement 与 target z 都为 .03 m。目标持续跟踪，没有“到达即失活”逻辑。距离/progress 复用既有 Terms；UAV 观测和 controller 数据复制自 basic_flight，原示例不变。车模型及来源许可位于 `src/muse_builtins/assets/ugv_diff_drive/`。

## 正式 Server 入口

在 `/project/mjc_ws` 激活环境，使用新的 data-dir 启动本版 Server（避免旧 metadata/Experiment 与必填 root/support DTO 混用）：

```bash
source /project/mjc_ws/.myenv/bin/activate
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m muse_server.app \
  --host 127.0.0.1 --port 8001 --viewer-port 8081 --data-dir /tmp/muse-heterogeneous
```

在另一终端通过正式 HTTP API 上传并保存 Experiment；此片段仅做注册，不运行物理或另建 driver：

```bash
source /project/mjc_ws/.myenv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

root = Path('examples/heterogeneous')
server = 'http://127.0.0.1:8001'

def post(route, body, content_type):
    with urlopen(Request(server + route, data=body, method='POST',
                         headers={'Content-Type': content_type}), timeout=60) as response:
        return json.load(response)

def upload(route, directory):
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(directory))
    return post(route, data.getvalue(), 'application/zip')

environment = upload('/catalog/environments', root / 'environment')
package = upload('/catalog/training-packages', root / 'training_package')
request = json.loads((root / 'request_train.json').read_text())
request.update(environment_id=environment['resource_id'], environment_revision=environment['revision'],
               training_package_id=package['resource_id'], training_package_revision=package['revision'])
print(post('/experiments', json.dumps(request).encode(), 'application/json')['experiment_id'])
PY
```

以输出的真实 ID 替换下面 `EXPERIMENT_ID`，复用已有 CLI 提交：

```bash
python examples/basic_flight/submit.py --server http://127.0.0.1:8001 \
  --experiment-id EXPERIMENT_ID --mode TRAIN --max-decision-steps 8 --no-display
```

checkpoint 恢复继续使用同一 Experiment ID 及已登记的 artifact ID：

```bash
python examples/basic_flight/submit.py --server http://127.0.0.1:8001 \
  --experiment-id EXPERIMENT_ID --checkpoint-artifact-id CHECKPOINT_ARTIFACT_ID \
  --mode TRAIN --max-decision-steps 8 --no-display
python examples/basic_flight/submit.py --server http://127.0.0.1:8001 \
  --experiment-id EXPERIMENT_ID --checkpoint-artifact-id CHECKPOINT_ARTIFACT_ID \
  --mode EVALUATE --max-decision-steps 8 --no-display
```

CPU 检查命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/integration/test_heterogeneous.py tests/integration/test_experiment_pipeline.py \
  tests/unit/test_checkpoint_contract.py --tb=short
```
