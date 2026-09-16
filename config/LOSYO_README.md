# M0 LOSYO 校准

入口为 `src/calibrate_m0_losyo.py`，设置为 `config/fsim_m0_losyo.json`。脚本只运行M0，不运行M1或FSIM-S。读取专用参数表 `config/fsim_m0_losyo_parameters.csv` 中M0的两个开放参数，其余数值保持固定；风速门控关闭。

## 默认规则

- 六折：每次留出一个完整站点年份，用其余五个训练。
- 范围：VPD_w 1–8 hPa、I_req 40–400 °C h，读取参数表，不在代码中另设范围。
- 粗网格15×37，共555点，间距0.5 hPa、10 °C h。
- 最多两轮二等分加密；第一次间距0.25 hPa、5 °C h，第二次间距0.125 hPa、2.5 °C h。如果全域都加密，第一轮为29×73=2,117点，最终为57×145=8,265点；包含此前网格点，不逐轮累加。
- 训练可行性：五个站点均产生预测。不可行组合的训练MAE/RMSE保持缺失，不以成功站点平均误差代替。程序错误立即停止，不当作无预测。
- 目标：五个训练站点的绝对区间边界距离MAE。RMSE作为补充，不参与并列选择。
- 加密：所有接触训练最佳MAE+1天以内已采样顶点的叶网格单元，包括彼此分离的区域。无可行点时全域加密。它不能保证发现粗网格顶点之间的所有狭窄低误差区域。
- 每折候选上限9,000；若下一轮超过上限，整轮不执行，明确记录candidate_budget，不能把停止当作收敛。默认最细全域网格低于这一上限。
- 并列最优：MAE距离最低值不超过1e-9天，按VPD_w、I_req升序选第一个已计算组合。仅为确定性选择规则，不引入“较低参数更合理”的假设。参数平均值不作为未经计算的新组合。
- 保存训练最佳+0.5天、+1天内的参数集合，以及边界触及情况。这些是近优集合，不是置信区间。范围不会自动扩充。
- 先写入selection.json冻结选择，再请求留出评价。其他折生成的确定性模型结果可按同一参数和站点复用，但加密、可行性和选参仅读取本折训练记录。

两次加密后仍只有有限分辨率结果。refinement_history.csv报告最佳训练MAE的变化；若变化较大或近优区域触边，需要另行检查，不能直接宣称连续空间已收敛。

## 运行

正式运行，四进程并行，默认显示每折和模型评估进度：

```bash
dm_first_os/bin/python src/calibrate_m0_losyo.py --workers 4 --output-dir result/fsim_m0_losyo
```

仅核对输入并生成配置和粗网格，不运行模型：

```bash
dm_first_os/bin/python src/calibrate_m0_losyo.py --design-only --output-dir result/fsim_m0_losyo_design_new
```

六折端到端小规模检查，9个粗网格点，最多加密到25点；不能用于解释正式性能：

```bash
dm_first_os/bin/python src/calibrate_m0_losyo.py --smoke --workers 4 --output-dir result/fsim_m0_losyo_smoke
```

输出目录须不存在或为空。可用 `--workers 2` 或 `--no-progress`。自定义设置用 `--settings`，设置文件中的项目相对路径从仓库根目录解析。

当前不提供断点恢复；evaluation_checkpoint.jsonl用于运行审计，不应把它当作可恢复的完整状态文件。默认会对两个有重复天气时间戳的输入做keep-first复算，与主策略keep-last比较，差异会停止分析，原始数据不改动。模型评估总数取决于各折加密区域和缓存命中率，不能简单按候选数×六折×六站点计算。主缓存和重复天气复算合计，默认设置的理论上限为8,265×8=66,120次模型评估。

## 输出

- `fold_summary.csv`：六折选参、训练MAE、并列数、停止原因。
- `oof_predictions.csv`：六个留出结果，含预测窗口、误差、无预测或校准失败。
- `oof_summary.json`：六折预测数、失败数、MAE、RMSE、7/10天准确预测比例。
- `folds/<站点>/loss_surface_stage_*.csv`、`loss_surface.csv`：各阶段/最终训练损失面，保留不可行组合。
- `refinement_history.csv`：候选数、可行数、最佳训练MAE变化。
- `selection.json`：训练站点名单、冻结参数、边界与停止诊断。
- `tied_best_parameters.csv`、`near_optimal_parameters.csv`：有可行参数时生成。
- `training_predictions.csv`、`held_out_prediction.csv`：训练明细和单独留出结果；校准失败不生成留出预测。
- `run_metadata.json`、`source_snapshot/`：配置、输入和源码指纹、版本、实际计算量。

若任一折无预测或校准失败，完整六站点MAE/RMSE为null。conditional误差只描述成功预测的站点，不能替代完整结果。7/10天准确比例分母始终为六，失败不满足准确条件，但单独报告校准失败数和无预测数。观测(L,U]与预测窗口的距离沿用现有模型评分函数；不是显症中点误差。

## 第三轮加密试验

`config/fsim_m0_losyo_refine3.json` 保持相同范围和555点粗网格，增加第三轮局部加密，最细步长变为VPD_w 0.0625 hPa、I_req 1.25 °C h。全域最细网格为113×289=32,657点，候选预算35,000；实际仍仅加密训练近优区域。该配置作为独立试验保留，不覆盖原两轮配置。

```bash
dm_first_os/bin/python src/calibrate_m0_losyo.py --settings config/fsim_m0_losyo_refine3.json --workers 4 --output-dir result/fsim_m0_losyo_Ireq40_400_refine3
```

运行完成后用 `dm_first_os/bin/python src/compare_m0_losyo_runs.py` 对照原运行。比较脚本核对模型、输入、范围及原有各阶段损失面，输出训练改善、参数变化和留出误差变化。不得根据留出改善与否选择参数，不能把局部加密停止直接称为全域收敛。

三轮试验已完成：8,380次评估（含1,961次天气策略复算），六折最佳训练MAE在新增一轮均未实质下降。留出MAE约33.74天，7/10天准确数仍为0/6、1/6。三个折的并列代表参数改变，不能将损失稳定等同于参数唯一。详细对照见 `result/fsim_m0_losyo_Ireq40_400_refine3/refinement_comparison.md`。

## 新增RH_sp的离散候选试验

独立入口为 `src/calibrate_m0_rh_trial.py`，配置为 `config/fsim_m0_losyo_rh_trial.json`。首轮RH候选为75%、80%、85%、90%，依据现有80%默认值和当前无雨夜间湿度分布设置探索对照，不视为文献规定的可校准区间。RH不在这一轮继续细分。每个候选值下均对VPD_w、I_req执行同样的三轮二维加密，范围仍为1–8、40–400。MP_sp等其他参数保持固定。

RH=80复用经源码、输入及配置指纹核验的三轮加密基线。每个新RH使用单独运行配置和进程池，模型缓存不跨RH共用；运行值记录在每个切片的effective_runtime_config.json。仅改变湿润小时判定的RH阈值，原来的降雨/叶湿替代条件和其他过程方程均保留。

每折先比较各RH切片的训练可行组合，在全部五个训练站点可预测的组合中按MAE选参。并列时优先最接近原始80%的RH，再按RH、VPD_w、I_req升序。三参数选择只接收训练损失面，冻结后读取相应留出预测；切片内为审计生成的其他留出结果不参与选择。

```bash
dm_first_os/bin/python src/calibrate_m0_rh_trial.py --workers 4 --output-dir result/fsim_m0_losyo_RH_trial
```

结果为 `RH_training_profiles.csv`（每折各RH的训练最低损失）、三参数损失面、并列和近优集合、六折留出预测，以及相对原RH=80基线的训练/留出误差对照。运行后用 `dm_first_os/bin/python src/plot_m0_rh_trial.py` 生成图件到 `fig/fsim_m0_losyo_RH_trial/`。目录须为空，不提供断点恢复。

RH试验已完成：新增26,443次评估，选定RH依manifest顺序为75/90/80/75/90/75。五折训练MAE降低，一折不变，但OOF MAE由33.74升至45.43天，RMSE由39.61升至49.27天，10天准确数由1/6降至0/6。主要恶化发生在DA_MPT_2023和NN_JF_2024。本次设置下不支持将RH开放版本替换原基线；原两参数M0及RH=80默认值保留。完整对照和图件见对应结果目录与 `fig/fsim_m0_losyo_RH_trial/`。

## 四参数TPE（512次/种子/折，尚未正式运行）

新增 `src/calibrate_m0_losyo_tpe.py` 与 `config/fsim_m0_losyo_tpe.json`，开放VPD_w、I_req、RH_sp、MP_sp。
六折LOSYO，两个种子，每种子每折512次试验，64次启动探索包含在内，4个进程，带tqdm和可复现的断点恢复。
详细范围、约束、预算、输出与命令见 [TPE_LOSYO_README.md](TPE_LOSYO_README.md)。本次仅编写并测试脚本，没有产生正式校准结果。

## 两参数TPE同预算对照

新增 `src/calibrate_m0_losyo_tpe_2p.py`，只开放VPD_w和I_req，固定RH=80、MP=3。
使用四参数n1024运行的冻结模型快照、同种子20260914、每折1024次（初始128次）与四进程。
说明及运行命令见 [TPE_LOSYO_2P_README.md](TPE_LOSYO_2P_README.md)。已检查实现，尚未正式运行。

## 两参数grid的地点/年份整体留出

新增 `src/calibrate_m0_grouped_grid.py` 与 `config/fsim_m0_grouped_grid.json`。
按DA/GL/NN或2022/2023/2024整体留出，分别三折，采用原三轮grid的冻结模型和范围，训练MAE适配每折3/4/5个site-year。
详细设计、命令和与原LOSYO配对输出见 [GROUPED_GRID_README.md](GROUPED_GRID_README.md)。未启动正式校准。

## 四参数分组TPE，单种子64+512

新增 `src/calibrate_m0_grouped_tpe.py` 和 `config/fsim_m0_grouped_tpe_4p.json`。
地点/年份各三折，每折64次初始探索+512次TPE，共576次，单种子20260914，四进程。
范围及指令见 [GROUPED_TPE_README.md](GROUPED_TPE_README.md)。正式校准尚未启动。

当前分组四参数TPE已调整为每折总计1024次（包含128次初始探索，之后896次TPE），单种子，六折并发共用16进程。最新命令见 GROUPED_TPE_README.md。
