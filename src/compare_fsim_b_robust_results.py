#!/usr/bin/env python3
"""Audit saved base/robust B searches and export paired comparisons; no simulation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_fsim_abc_selection_comparison import load

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'result/fsim_b_robust_comparison'
    output.mkdir(parents=True, exist_ok=True)
    searches = [
        ('base_1024', 'fsim_abc_tpe_seed123', 1024),
        ('robust_1024', 'fsim_b_robust_tpe_248_2048', 1024),
        ('robust_2048', 'fsim_b_robust_tpe_248_2048', 2048),
    ]
    folds, cases, hashes = [], [], {}
    for label, directory, budget in searches:
        f, c, _, audit = load(ROOT / 'result' / directory, 'B', 123, budget)
        folds.append(f.assign(search=label))
        cases.append(c.assign(search=label))
        hashes[label] = audit
    folds, cases = pd.concat(folds, ignore_index=True), pd.concat(cases, ignore_index=True)
    folds.to_csv(output / 'fold_comparison.csv', index=False)
    cases.to_csv(output / 'all_predictions.csv', index=False)
    rows = []
    for (search, strategy, scheme), block in cases.groupby(['search', 'strategy', 'scheme']):
        errors = block.loc[block.status.eq('predicted'), 'signed_interval_error_days'].abs()
        rows.append(dict(search=search, strategy=strategy, scheme=scheme,
                         n_predicted=len(errors), n_test=len(block),
                         conditional_MAE_days=errors.mean(),
                         full_MAE_days=errors.mean() if len(errors) == len(block) else np.nan))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'method_comparison.csv', index=False)
    keys = ['strategy', 'scheme', 'held_out', 'input_key']
    keep = keys + ['status', 'predicted_start', 'predicted_end', 'signed_interval_error_days']
    paired = cases[cases.search.eq('base_1024')][keep].merge(
        cases[cases.search.eq('robust_2048')][keep], on=keys,
        suffixes=('_base', '_robust'), validate='one_to_one')
    paired['delta_absolute_error_days'] = (
        paired.signed_interval_error_days_robust.abs() - paired.signed_interval_error_days_base.abs())
    paired.to_csv(output / 'paired_predictions.csv', index=False)
    diagnostics = []
    for row in folds[(folds.search == 'robust_2048') & (folds.strategy == 'minimum')].itertuples():
        path = ROOT / 'result/fsim_b_robust_tpe_248_2048/B' / row.scheme / 'seed_123/folds' / row.held_out / 'trials.csv'
        t = pd.read_csv(path)
        assert len(t) == 2048 and set(t.trial) == set(range(2048)), path
        feasible = t[t.feasible.eq(True)]
        best = feasible.training_MAE_days.min()
        assert np.isclose(best, row.training_MAE_days, atol=1e-9, rtol=0), path
        ties = feasible[np.isclose(feasible.training_MAE_days, best, atol=1e-9, rtol=0)]
        near = feasible[feasible.training_MAE_days <= best + 1 + 1e-9]
        d = dict(scheme=row.scheme, held_out=row.held_out, n_trials=len(t),
                 n_feasible=len(feasible), final_best_training_MAE_days=best,
                 first_final_best_trial=int(ties.trial.min()), n_exact_best=len(ties),
                 n_within_one_day=len(near),
                 training_improvement_after_1024_days=feasible[feasible.trial < 1024].training_MAE_days.min()-best)
        for name, block in [('exact_best', ties), ('within_one_day', near)]:
            for parameter in ['VPD_w', 'I_req']:
                d[f'{name}_{parameter}_min'] = block[parameter].min()
                d[f'{name}_{parameter}_max'] = block[parameter].max()
        diagnostics.append(d)
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(output / 'search_diagnostics.csv', index=False)
    (output / 'input_table_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    table = ['|选择规则|验证|基础 B 1024|升级搜索 1024|升级搜索 2048|',
             '|---|---|---:|---:|---:|']
    for strategy in ['minimum', 'tolerance']:
        for scheme in ['losyo', 'site', 'year']:
            values = []
            for search, _, _ in searches:
                r = summary[(summary.search == search) & (summary.strategy == strategy) & (summary.scheme == scheme)].iloc[0]
                values.append(f'{r.conditional_MAE_days:.4f} ({r.n_predicted}/6)')
            table.append('|'+ '|'.join([strategy, scheme] + values) + '|')
    shared = paired[(paired.strategy == 'minimum') & (paired.scheme == 'site')].dropna(
        subset=['signed_interval_error_days_base', 'signed_interval_error_days_robust'])
    report = '''# 基础 B 与扩展搜索 B 的结果核查

本报告只读取保存结果，不运行 FSIM，不修改后台孢子试验。
基础搜索为每折总计 1024 次（含 128 次启动探索）；升级搜索为总计 2048 次（含 248 次启动探索）。
二者使用同一模型和固定参数空间。升级搜索的前 1024 次是新的搜索轨迹，不是基础搜索的原始 1024 次。
程序核实升级搜索全部 12 折各有完整 trial 0–2047，且保存选择对应最终可行训练最小值。

## 汇总

单位：天；括号为有预测的测试 site-year 数。缺失预测不按零计；不足 6/6 时只报告条件 MAE。
minimum 仅以训练结果选参；tolerance 在训练最小值 +1 天内使用测试误差选参，属于探索性结果。

''' + '\n'.join(table) + f'''

## 主要变化

- LOSYO 最低训练 MAE 方案仅 NN 2023 的测试误差发生实质变化：24.58 → 62.88 天，均为偏早；训练 MAE 却从 15.39 降到 9.675 天。其余五个测试误差不变。因此汇总恶化不是所有 site-year 普遍恶化。
- 留 site 的基础结果缺少 DA 2023 预测；升级搜索给出偏晚 89.33 天的预测。共同有预测的五个案例，MAE 为 {shared.signed_interval_error_days_base.abs().mean():.4f} → {shared.signed_interval_error_days_robust.abs().mean():.4f} 天。不能直接把基础条件 MAE 与升级完整 MAE 相减作为性能变化。
- 留 year 的改善来自 2024 折：NN 2024 从偏晚 102.33 天变为偏早 1.42 天，但 DA 2024 从偏晚 3.63 天变为偏晚 57.71 天。平均改善同时掩盖了站点之间的误差转移。
- 相比基础搜索，12 折只有 LOSYO NN 2023 和留年 2024 两折的最终最小训练 MAE 实质降低，其余折基本相同。
- 升级搜索 1024 → 2048 次时，最低训练规则的 LOSYO 和留 site 测试误差不变；留 year 仅改善约 0.014 天。主要差异已经在新的搜索轨迹前 1024 次出现，不能归因于单纯把预算翻倍。

## 可提取的额外信息与边界

训练拟合更好不保证留出案例更准。NN 2023 的变化表明训练目标偏好的参数区间和留出 site-year 的需要可能不一致；仅凭六个案例不能进一步断言生物机制错误或已证明普遍过拟合。

相同训练 MAE 可以对应不同参数和明显不同的测试输出。search_diagnostics.csv 给出训练最优平台及 +1 天候选集合内的参数范围和数量。这些是本次搜索访问到的范围，不是置信区间，也不是完整参数可辨识性分析；重复候选计入试验数。

容错候选集合以当前训练最小值为基准。随着训练最小值下降，该集合会收缩或改变，所以增加搜索预算也不保证探索性测试最优结果改善。右列使用测试结果选参，不可作为独立泛化性能或孢子增益的主证据。

目前只观察到单一种子、两条搜索轨迹，不能声称优化已全局收敛或升级版本统计上优于/劣于基础版。后半段预算的边际变化很小，但这不排除其他种子找到不同区域。

正在运行的孢子增益试验应保留预先固定的基础 B 参照。扩展搜索结果可作为优化敏感性证据；不应因为看过测试误差而替换参照或选取某折更有利的基线。若后续另做敏感性评估，应对同一组新固定参数成对比较有/无孢子信息。

## 文件

- method_comparison.csv：三种搜索预算/轨迹、两种选参规则的汇总。
- fold_comparison.csv：逐折训练误差、测试误差和所选参数。
- paired_predictions.csv：基础与升级的逐案例窗口、状态、带符号误差及绝对误差变化。
- search_diagnostics.csv：完整性、可行性、最优平台、候选范围和后半程训练收益。
- input_table_hashes.json：本次读取结果表的 SHA256。
- 图：../../fig/fsim_b_robust_comparison/B_2048_selection.png（另有 PDF）。

复现：`dm_first_os/bin/python src/compare_fsim_b_robust_results.py`
'''
    (output / 'README.md').write_text(report, encoding='utf-8')
    print(summary.to_string(index=False))
    print(diagnostics[['scheme', 'held_out', 'n_exact_best', 'first_final_best_trial', 'training_improvement_after_1024_days']].to_string(index=False))
    print(output / 'README.md')


if __name__ == '__main__':
    main()
