# 固定基站端点标记

`station.xml`为Muse项目原创的示意资产，未复制外部设备模型或物理参数，沿用项目许可范围。

- `station`固定在Scenario指定的世界位姿，单位m、Z向上、四元数wxyz；无joint、actuator或controller状态。
- 半径0.1m的蓝色球只帮助观察位置，不表示真实天线、机柜或塔体尺寸。没有碰撞体，不为radio或sensing提供传播/遮挡参数。
- 真实场景需要的塔杆、机柜、建筑应在环境资产中显式建模；近距离避障不能把此marker当成真实设备。
- `controller_id: none`、`sensor_profile_id: none`；实体名称不触发无线发射。Radio device绑定属于后续M3工作包。

建模依据与取舍见`docs/superpowers/plans/2026-10-04-muse-m3-fixed-entity.md`。
