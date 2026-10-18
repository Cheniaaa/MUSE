# 室内 Radio 科学参考

本例是可重复的模型验证包：两架CF2以body +y方向0.1m/s低速移动、向固定站上传，固定背景源同频发射，静态墙体提供可控遮挡。节点数、拓扑、频率、动作和部署均在本包配置，不是平台限制。

## 模型与近似

默认 `builtin:indoor_office_radio`：TR38.901 V16.1.0 InH-Office确定性dB中心路径损耗，`1≤d≤150m`、`0.5<f<100GHz`；几何LoS选择分支，无随机阴影/衰落、材料/反射、多径或天线方向图。1m终端/3m站高沿标准名义部署尺度，不声称任意高度/真实建筑已标定。

`builtin:free_space_radio` 是ITU-R P.525-5解析对照，忽略有效路径的遮挡分类，仍拒绝无效距离；`minimum_distance_m=1` 是本例假想远场下界，不是实物天线标定。两者都采用理想各向同性端点、相同接收机参数、同频半双工、干扰视为噪声的Shannon理想速率，不模拟Wi-Fi协议/编码/重传。

## 配置

- UAV初始 `(0,-.3,1)/(0,-5,1)`，站 `(20,0,3)`，背景源 `(10,8,1)`，墙中心 `(10,0,1.5)`、half-size `(.1,.1,1.5)`；地板覆盖约30×20m，3m为名义层高，无完整建筑模型。原提案0.4m/s超过已验证controller的0.1m/s边界，因此收窄示例速度/墙宽与起点，不改变平台controller。
- 频率2.412/2.437GHz，每resource 5MHz、reuse，channel ID 7/−3；发射预算1mW。
- T=290K为输入热噪声参考温度，NF=7dB，噪声每resource约 `1.003347297e-13 W`，NF不再次从信号中扣除。
- dt=.002s，decision=20ticks，radio最大7ticks，偏移 `[0,7,14,20]`；四个50decision动作段：0W/channel7、0.1mW/channel7、1mW/channel7、1mW/channel−3，均保持同一运动请求。段结束后持有最后动作；episode上限500decisions。
- `request_evaluate.json`：B=2、同功率、无service；只读rate评分显式激活Radio，无消息queue。规则动作不读取radio/geometry truth，收据仅用于诊断。
- `request_service.json`：world power scales `[1,.25]`，每条up链路 `2,000,000 bit`，两条链路均完成才结束world。业务耗尽仍非零RF时继续产生干扰。

业务预算由独立Python math名义定高直线轨迹积分选择：沿最终0.1m/s配置，t=6s时快world两链路约2.091/5.203Mbit、慢world约0.742/1.616Mbit，故2Mbit可区分快慢world；此数值是名义轨迹预算选择，不是物理飞行验收真值。实际验收应使用真实geometry逐epoch标量对照。

## 运行

先 `source /project/mjc_ws/.myenv/bin/activate`。沿[Basic Flight API流程](../basic_flight/README.md)启动Server，controller仍用 `examples/basic_flight/controller.json`。

1. 将本例 `environment/` 内容打成ZIP，经 `POST /catalog/environments` 登记。
2. 将 `training_package/` 内容打成ZIP，经 `POST /catalog/training-packages` 登记；归档根层直接放入口文件。
3. 将返回的ID/revision填入选择的request JSON占位字段，`POST /experiments`。service请求启用`world.termination.done`，主例保持关闭。
4. 创建 `{"experiment_id":"实际ID","mode":"EVALUATE","max_decision_steps":200}`。规则算法不需checkpoint，不支持TRAIN。
5. Worker日志每decision输出 `INDOOR_RADIO_RESULT`：三块结果/valid、采样tick/interval、pre-reset service与world reset。真实几何逐epoch误差和曲线见GPU验收，不用每decision末快照冒充全区间积分。

free-space对照：在包YAML的`compositions.default.model_bindings.radio.options`声明`builtin:free_space_radio: {config: {}}`，重新登记包，在request设置`model_choices.radio`为`builtin:free_space_radio`，并在`model_config.radio`增加`minimum_distance_m:1`后重新resolve；不能在运行中替换冻结模型。背景源关闭/换频及墙体对照同样通过实验配置/环境新revision完成。

## 验证入口

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m pytest -q tests/integration/test_indoor_radio_pipeline.py
python -m pytest -q tests/gpu/test_indoor_radio_worker.py
```

第二条需独占GPU串行运行。[设计](../../docs/superpowers/specs/2026-10-13-muse-analytical-radio-design.md)与[来源记录](../../docs/validation/m3-radio-reference-research.md)列出公式和适用性。

回归测试的数值JSON/曲线写入各用例的pytest临时目录，Worker日志留在对应run的 `logs/worker.log`。正式归档可指定新的 `--basetemp` 目录并先创建父目录；pytest会清空已存在的basetemp，不要指定历史证据目录。具体见[M3收口记录](../../docs/validation/m3-complexity-cleanup.md)。

## 实测证据

[M3.6验收报告](../../docs/validation/m3-analytical-radio.md)：两模型各200decision真实Worker、world0第160步/world1第161步业务完成、K7/3/1时间细化及只遮挡desired/jammer对照通过。最终联合349项＋独立遮挡1项通过；科学batch对未量化scalar最大相对误差5.72e−8。曲线/原始轨迹在 `.planning/mjlab-spike/analytical_radio/`，不把该结果当成实测无线吞吐或训练收敛。
