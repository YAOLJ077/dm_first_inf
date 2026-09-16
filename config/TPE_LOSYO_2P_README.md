# 两参数 TPE：四参数运行的同预算对照

新增入口 `src/calibrate_m0_losyo_tpe_2p.py`，配置 `config/fsim_m0_losyo_tpe_2p.json`。
只采样 VPD_w（1–8 hPa）和 I_req（40–450 °C h），均连续；固定 RH_sp=80%、MP_sp=3 h。
种子20260914，六折LOSYO，每折1024次，其中128次初始探索（包含同折历史参考点），4进程。
初始参考点与四参数运行相同；不读取四参数最优组合来初始化两参数搜索。

目标保持五个训练site-year均可预测条件下的区间边界MAE。训练选择冻结后计算留出预测。
并列时按VPD、I、seed、trial顺序；固定RH/MP使原四参数并列规则自然退化为该规则。
使用相同种子不意味着逐组合配对：维度改变会改变随机采样和TPE反馈路径。

模型和评估辅助代码冻结在 `src/tpe_m0_reference/`，来自已完成的
`result/fsim_m0_losyo_tpe_4p_n1024/source_snapshot/`，复制时逐文件核对原run_identity的SHA256。
该私有目录是有意保留的历史代码，不应随当前model目录修改。原四参数结果与入口不修改。
主进程和spawn子进程均从私有目录导入模型。天气路径仍指向项目输入文件。
运行前核对快照哈希、原四参数输入哈希、软件版本、搜索范围、目标和预算；不一致时停止。
设计检查、四进程合成天气检查、两维TPE采样及断点重放检查已完成，未启动正式校准。

在项目根目录运行：

```bash
dm_first_os/bin/python src/calibrate_m0_losyo_tpe_2p.py --settings config/fsim_m0_losyo_tpe_2p.json --workers 4 --output-dir result/fsim_m0_losyo_tpe_2p_n1024
```

只检查配置与输入：在上述指令末尾加 `--design-only`。中断后恢复：加 `--resume`，保持配置和源码不变。
进度条每折显示1024次参数试验、当前最优训练MAE和可行数。
总计6144次参数试验，30720次训练模型计算、10240次重复天气策略核查，最多另加8次留出及其核查。
不跨试验缓存，因此预算最多40968次；中断中尚未完整保存的试验重算会增加实际调用数。

输出结构与四参数一致：`fold_summary.csv`、`oof_predictions.csv`、`oof_summary.json`，
每折训练试验/收敛/近优集合/最终选择/有效运行配置，以及完整试验JSON和恢复身份信息。
比较应使用六折配对留出误差、预测失败、7/10天准确数及训练收敛，不依据留出表现重新选参。
模型评估器的约束数值占位字段optimizer_value不是无预测组合的真实MAE，误差分析使用training_MAE_days。
