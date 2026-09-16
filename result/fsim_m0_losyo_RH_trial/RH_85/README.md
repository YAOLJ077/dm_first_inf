# M0 LOSYO 校准

运行类型：正式校准。
每折只使用五个训练站点年份；要求五个均产生预测，在可行组合中最小化绝对区间距离的平均值（MAE）。
所有达到训练最佳MAE +1天的已采样顶点所接触的网格单元均参与加密；无可行点时全域加密。
并列最优在数值容差内按 VPD_w、I_req 升序选取已计算的实际组合；不按留出结果选择。该规则仅使结果可复现，不表示低参数值更可信。
网格结果依赖分辨率；局部加密可能遗漏粗网格顶点之间狭窄的低误差区域，不能称为连续空间的精确全局最优。
近优集合是训练损失集合，不是后验分布或置信区间。触边只记录，不自动扩展参数范围。

| 留出站点年份 | 状态 | 最佳训练MAE（天） | 候选数 |
|---|---|---:|---:|
| DA_MPT_2022 | selected | 18.9 | 674 |
| DA_MPT_2023 | selected | 10.333333333333334 | 790 |
| DA_MPT_2024 | selected | 10.825 | 719 |
| GL_WK_2022 | selected | 13.866666666666665 | 674 |
| NN_JF_2023 | selected | 9.674999999999999 | 638 |
| NN_JF_2024 | selected | 17.06666666666667 | 687 |

OOF汇总：

```json
{
  "n_folds": 6,
  "n_calibration_failures": 0,
  "n_predictions": 6,
  "n_no_predictions": 0,
  "prediction_rate_fixed_six": 1.0,
  "MAE_days": 40.80555555555556,
  "RMSE_days": 44.79194444358313,
  "conditional_MAE_days": 40.80555555555556,
  "conditional_RMSE_days": 44.79194444358313,
  "accurate_prediction_rate_7d_fixed_six": 0.16666666666666666,
  "accurate_prediction_rate_10d_fixed_six": 0.16666666666666666
}
```

任何留出预测缺失时，完整六站点MAE/RMSE为null；conditional指标仅针对成功预测，必须与失败数同时查看。
7/10天准确比例分母固定六；校准失败也不计为准确预测，但单独列出失败数。没有给缺失预测赋予日期或误差。
文件：folds/*/loss_surface*.csv 为训练损失面；refinement_history.csv 为加密变化；selection.json 为冻结选择；
tied_best_parameters.csv / near_optimal_parameters.csv 为并列及近优集合；held_out_prediction.csv 为单个留出评价。
oof_predictions.csv 为六折结果，fold_summary.csv 为参数与训练目标汇总。模型计算缓存仅复用固定参数和站点的确定性结果，不复用其他折的选参决策。
