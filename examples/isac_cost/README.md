# ISAC＋world cost参考

两架Crazyflie分别属于`alpha/beta`，联合学习4维motion、RF功率和频道`[7,-3]`。
使用`builtin:mappo`，每role独立actor29维，集中critic58维；actor只读自身导航、上次motion、剩余时间、所选链路反馈和本地path-length测量。station坐标是已知导航目标，不向actor暴露感知target真值、FIM或位置界。

## 交付状态

2026-10-16完成M4.3工程收口：混合动作／cost真实参与学习、规则可行性、HTTP训练／恢复／冻结评价通过；**学习策略的任务效果未达标**。既有seed7的cost-on/off最终策略分别403／368步失败；另行验证gamma=1、budget25后仍在330步失败，未通过后续配对门槛。示例默认仍为原gamma=.99配置。

原90Run矩阵保留15完成／1取消／74未开始。用户已确认将飞行安全学习、完整cost-on/off及MAPPO/IPPO效果对照转入后续研究／M6。机制通过不表示策略收敛，详见本地[交付记录](../../docs/validation/m4-reference-scenarios.md)与[全时域验证](../../docs/validation/m4-full-horizon-pilot.md)。

## 注册与运行

在项目根执行（不新增依赖）：

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m muse_server.app --port 8001 --data-dir build/reference-server --controller examples/basic_flight/controller.json
```

将`environment/`和`training_package/`分别打为根目录含`environment.yaml`／`__init__.py`的ZIP，通过`POST /catalog/environments`、`POST /catalog/training-packages`上传（`Content-Type: application/zip`）；用返回的ID/revision替换`request_train.json`占位值，再将完整JSON提交`POST /experiments`。沿`examples/basic_flight/README.md`的Server启动与HTTP流程操作，向`POST /runs`提交训练body：

```json
{"experiment_id":"REGISTERED_EXPERIMENT_ID","mode":"TRAIN","max_decision_steps":4096}
```

运行默认不打开显示；`preview.world_ids=[0]`只是预览配置，训练时不发启用显示请求。学习EVALUATE需使用已登记checkpoint，通过同一接口仅修改B/root_seed；不直接用未初始化的学习策略做规则基线。

请求固定B16/root_seed7、H32、MLP[64,64]、gamma=.99；4096 decision对应128次update、65536 world transitions。原预注册研究矩阵使用seed7/17/27；初始化checkpoint、恢复、独立seed101/211评价沿正式API。剩余矩阵的后续执行需重新确认研究协议与预算，此包本身不声明训练达标。

完整固定对照作为可选研究入口保留。以下命令启动一批新的90Run试验，使用新的输出目录：

```bash
source /project/mjc_ws/.myenv/bin/activate
python examples/reference_benchmark.py --server http://127.0.0.1:8001 --output build/reference-benchmark-01
```

脚本统一执行两个参考任务的5配置×3训练seed，共90个初始化/训练/评价Run；总预算240分钟，主训练每Run15分钟、初始化/评价每Run5分钟。预算用尽即取消并保留未完成状态，不重试或扩训。`ledger.json`关联每个Run、artifact与`first_episode.json`；只统计各world首episode，失败后新episode不替换样本。1步TRAIN用于发布0更新初始化checkpoint；训练更新仍要求完整H步，半段rollout不进checkpoint。

## 科学量与评分

- dt=.002s、decision20ticks=.04s，epoch7/7/6；两service分别绑定`alpha_up/beta_up`，每展开链路4e9bit，跨频道共享backlog。
- `reward=served_bits/5e6−E_RF/8e−5`。服务是完整decision累计，链路反馈只是末epoch快照；能量只对受控实际E×F发射求和，不按link/target重复计。
- `mission_position_violation`为world、`episode_discounted_sum`，discount=.99、budget5、lambda初值0／lr.01。正常步复用builtin逐目标FIM位置界判据（.01m）；缺测／不可逆均违反。这不是实际定位误差。
- 500步为真实有限任务终止，`time_out=false`。任一slot crash或越过所属工作盒立即终止；失败reward压缩为`−R_abs*sum(gamma**j)`、mission cost为同一几何和，保留实际服务／能量diagnostic，不补物理transition。
- Task diagnostics固定为episode_decision、served_bits、service_valid、rf_energy_j、failed、task_done、task_reward、position_violation、mission_cost。`service_valid=false`仍保留已完成的已知bits，科学评价必须单独判失败。

## 固定对照

同一包显式声明`mappo_cost_on`（默认）、`mappo_cost_off`和`ippo`三个完整组合，请求通过`composition_id`选择，不重写YAML或重新生成归档。MAPPO cost-off组合默认关闭`world.cost.mission` selection，请求清空`cost_config`，Task仍报告原始约束diagnostic。IPPO组合使用`builtin:ippo`，启用各role的team reward，移除world reward/cost与critic selections及MAPPO专属配置，请求同样清空`cost_config`。环境／Task／actor权限不变。

`tests/reference_scenario_helpers.py::reference_catalog`选择上述包内组合，供测试及验收复用；不是平台算法切换兼容层。CPU验证：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/unit/test_reference_tasks.py tests/integration/test_reference_scenarios_pipeline.py
```
