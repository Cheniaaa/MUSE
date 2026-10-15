# Hybrid＋history＋Markov参考（无cost）

`pilot`单Crazyflie通过`builtin:ippo`学习motion、RF功率和非连续频道`[7,-3,19]`。固定jammer以三态Markov过程切换频道，转移矩阵`[[.8,.2,0],[0,.8,.2],[.2,0,.8]]`、均匀初态、dwell20ticks；jammer状态／切换时刻／完整频谱真值不进actor。

## 交付状态

2026-10-16完成M4.3工程收口：history的内容／时序／reset、混合动作更新、无cost路径、真实HTTP训练／恢复／冻结评价已验证。**L1/L4有限学习收益对照尚未完成**；可观测信息增加的oracle不等于PPO已学会利用历史。

原研究矩阵保留15完成／1取消／74未开始，完整history配对收益及多seed验证经用户确认转入后续研究／M6。详见本地[交付记录](../../docs/validation/m4-reference-scenarios.md)。

## 注册与运行

```bash
source /project/mjc_ws/.myenv/bin/activate
python -m muse_server.app --port 8001 --data-dir build/reference-server --controller examples/basic_flight/controller.json
```

将本例`environment/`和`training_package/`各打为根目录含manifest的ZIP，分别提交`POST /catalog/environments`和`POST /catalog/training-packages`（`Content-Type: application/zip`），用返回ID/revision替换`request_train.json`占位值，再提交完整JSON到`POST /experiments`。Server启动沿`examples/basic_flight/README.md`，向`POST /runs`创建训练Run：

```json
{"experiment_id":"REGISTERED_EXPERIMENT_ID","mode":"TRAIN","max_decision_steps":4096}
```

默认不启用显示；`preview.world_ids=[0]`仅指定打开显示时的预览world。请求固定B16/root_seed7、H32、MLP[64,64]；每seed最多4096 decision／128update。限定对照seed7/17/27，最终及初始化checkpoint通过同一EVALUATE路径在seed101/211、B4各评价500步；此包不预言历史必然带来学习收益。

完整的L1/L4与ISAC固定对照由`examples/reference_benchmark.py`提供可选研究入口，命令、90个Run及240分钟预算见[ISAC运行说明](../isac_cost/README.md#注册与运行)。执行前确认新的研究协议与预算；脚本保存所有seed首episode结果，达到运行预算不等于观察到历史收益。

## L1／L4输入

自身导航状态＋上次motion＋剩余时间18维，均不堆history。只有`feedback` selection设置`history_length: 4`，公开shape为`[B,N,4,3,4]`，原始Term输出仍为`[B,N,3,4]`；L4共66维。L1共30维，可在Experiment请求覆盖：

```json
{"term_config":{"roles":{"pilot":{"observation":{"feedback":{"history_length":1}}}}}}
```

history从旧到新，起始零／invalid帧由native backfill重复；不是重复采集，不增加FIM或Markov采样。每resource反馈是选中链路末epoch的rate/1e8、valid、自己power/.001及sample age。未选资源为零；不观察remaining_bits等隐式历史。

当当前resource0无干扰时，state1/2别名；上一帧resource1或2有干扰可令下一步resource0受干扰条件概率分别.04/.2。这是有限历史有额外信息的oracle，不是PPO效果保证。

## 任务与边界

dt=.002s、decision20ticks=.04s、epoch7/7/6；station=(4,8,1)m、jammer=(4,-8,1)m／1mW，单`uplink` service跨三个resource共用4e9bit业务。

`reward=served_bits/2.5e6−E_RF/4e−5`，只算pilot实际发射能量，不算jammer。工作盒`[-2,-2,.5]..[2.5,3,1.5]`，任一slot crash／越界立即失败；失败reward按剩余有限任务折扣尾项压缩。500步正常终止也是`time_out=false`，不bootstrap。Task仅报告固定实际ledger，Terms只读，不重复计分。

包不声明cost selections、mission_cost字段或cost learner；请求cost_config明确为空。独立CPU检查及两种history真实Catalog构建由`tests/reference_scenario_helpers.py::reference_catalog`提供；GPU／Worker／HTTP已有机制验收，有限学习效果按上述研究待办继续。
