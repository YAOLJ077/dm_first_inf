#!/usr/bin/env python3
"""Compare a nested I_req range expansion with the preserved baseline grid.

The analysis is descriptive. Grid minima are full-data diagnostics, not
calibrated parameters or estimates of held-out performance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.report_fsim_parameter_analysis import plt
PARAMETERS = ['model','VPD_w','I_req','u_tr']
MATCH = PARAMETERS + ['design','input_key']


def read_run(path):
    meta = json.loads((path/'run_metadata.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError(f'Incomplete run: {path}')
    return {'metadata': meta,
            'manifest_sha256': hashlib.sha256((path/'site_year_manifest.csv').read_bytes()).hexdigest(),
            'table': pd.read_csv(path/'parameter_table.csv').fillna(''),
            'predictions': pd.read_csv(path/'predictions.csv'),
            'metrics': pd.read_csv(path/'sample_metrics.csv'),
            'profiles': pd.read_csv(path/'parameter_profiles.csv'),
            'near': pd.read_csv(path/'near_optimal_sets.csv')}


def check_common_predictions(baseline, extended):
    if baseline['metadata']['input_sha256'] != extended['metadata']['input_sha256']:
        raise ValueError('Input fingerprints differ between runs')
    if baseline['manifest_sha256'] != extended['manifest_sha256']:
        raise ValueError('Endpoint/input manifest differs between runs')
    old_table = baseline['table'].set_index('parameter')
    new_table = extended['table'].set_index('parameter')
    stable_fields = ['config_key','value_json','role','scope','unit']
    pd.testing.assert_frame_equal(old_table[stable_fields], new_table[stable_fields])
    other = old_table.index != 'I_req'
    fields = ['lower','upper','grid_levels']
    pd.testing.assert_frame_equal(old_table.loc[other, fields],new_table.loc[other, fields])
    # The configuration loader and design code may change; biological source
    # and controller implementations must remain identical in this experiment.
    for name, digest in baseline['metadata']['source_sha256'].items():
        if name.startswith('model/') and name != 'model/parameter_table.py':
            if extended['metadata']['source_sha256'].get(name) != digest:
                raise ValueError(f'Model source changed: {name}')
    a,b = baseline['predictions'],extended['predictions']
    paired = a.merge(b, on=MATCH, suffixes=('_baseline','_extended'),
                     how='left', validate='one_to_one', indicator=True)
    if not (paired['_merge']=='both').all():
        raise ValueError('Expanded design does not contain the full baseline design')
    columns = ['observed_L','observed_U','status','predicted_start','predicted_end','n_windows','overlap',
               'interval_distance_days','signed_interval_error_days',
               'window_width_days','first_positive_distance_days']
    for col in columns:
        left,right = paired[col+'_baseline'],paired[col+'_extended']
        if pd.api.types.is_numeric_dtype(left):
            same = np.isclose(left, right, rtol=0, atol=1e-9, equal_nan=True)
        else:
            same = left.fillna('MISSING') == right.fillna('MISSING')
        if not np.all(same):
            raise ValueError(f'Baseline reproducibility check failed: {col}')
    return {'n_common_records': len(paired), 'all_common_predictions_equal': True,
            'n_common_grid_records': int((paired.design=='grid').sum())}


def select_best(block):
    """Coverage first, then timing; stable sorting resolves numerical ties."""
    good = block[(block.n_errors == 0) & (block.n_predictions > 0)]
    if good.empty:
        raise ValueError('No valid prediction in candidate group')
    return good.sort_values(['n_predictions','conditional_mae_days','sample_id'],
                            ascending=[False,True,True]).iloc[0]


def compare_best(baseline, extended):
    comparisons,paired_records = [],[]
    for model in ('M0','M1'):
        old = select_best(baseline['metrics'].query('model == @model and design == "grid"'))
        new = select_best(extended['metrics'].query('model == @model and design == "grid"'))
        record = {'model': model}
        for label,row in [('baseline',old),('extended',new)]:
            for col in ['sample_id','VPD_w','I_req','u_tr','n_predictions','n_overlaps',
                        'conditional_mae_days','mean_signed_error_days','mean_window_width_days']:
                record[f'{label}_{col}'] = row[col]
        record['same_prediction_coverage'] = old.n_predictions == new.n_predictions
        record['mae_reduction_days'] = (old.conditional_mae_days-new.conditional_mae_days
                                         if record['same_prediction_coverage'] else np.nan)
        comparisons.append(record)
        a = baseline['predictions'][baseline['predictions'].sample_id == old.sample_id]
        b = extended['predictions'][extended['predictions'].sample_id == new.sample_id]
        columns = ['input_key','status','predicted_start','predicted_end','overlap',
                   'interval_distance_days','signed_interval_error_days','window_width_days']
        pair = a[columns].merge(b[columns],on='input_key',suffixes=('_baseline','_extended'),validate='one_to_one')
        pair.insert(0,'model',model)
        pair['absolute_error_reduction_days'] = pair.interval_distance_days_baseline-pair.interval_distance_days_extended
        pair['comparison_status'] = np.where(pair.status_baseline != pair.status_extended,'prediction_status_changed',
                                            np.where(pair.status_extended=='predicted','paired_prediction','both_missing'))
        paired_records.append(pair)
    return pd.DataFrame(comparisons),pd.concat(paired_records,ignore_index=True)


def compare_at_fixed_other_parameters(baseline, extended):
    """Isolate I_req search expansion while holding VPD_w and u_tr fixed."""
    a = baseline['metrics'].query('design == "grid"')
    b = extended['metrics'].query('design == "grid"')
    records = []
    for (model,vpd,wind),old_group in a.groupby(['model','VPD_w','u_tr'],dropna=False):
        match = (b.model==model) & np.isclose(b.VPD_w,vpd)
        match &= b.u_tr.isna() if pd.isna(wind) else np.isclose(b.u_tr,wind)
        new_group = b[match]
        old,new = select_best(old_group),select_best(new_group)
        comparable = old.n_predictions == new.n_predictions == 6
        records.append({'model':model,'VPD_w':vpd,'u_tr':wind,
                        'baseline_I_req':old.I_req,'extended_I_req':new.I_req,
                        'baseline_n_predictions':old.n_predictions,'extended_n_predictions':new.n_predictions,
                        'baseline_mae_days':old.conditional_mae_days,'extended_mae_days':new.conditional_mae_days,
                        'both_complete':comparable,
                        'mae_reduction_days':old.conditional_mae_days-new.conditional_mae_days if comparable else np.nan})
    return pd.DataFrame(records)


def boundary_checks(run):
    records = []
    s = run['metrics'].query('design == "grid"')
    for model,block in s.groupby('model'):
        best=select_best(block)
        for cap in (90,120,150,180,200):
            eligible = block[block.I_req<=cap]
            winner=select_best(eligible)
            near = eligible[(eligible.n_predictions==winner.n_predictions)&
                            (eligible.conditional_mae_days <= winner.conditional_mae_days+1+1e-12)]
            records.append({'model':model,'upper_cap':cap,'n_grid':len(eligible),
                            'n_complete_combinations':int((eligible.n_predictions==6).sum()),
                            'best_n_predictions':winner.n_predictions,
                            'best_I_req':winner.I_req,'best_VPD_w':winner.VPD_w,'best_u_tr':winner.u_tr,
                            'best_mae_days':winner.conditional_mae_days,
                            'best_at_cap':bool(np.isclose(winner.I_req,cap)),
                            'n_within_one_day':len(near),
                            'near_I_min':near.I_req.min(),'near_I_max':near.I_req.max()})
    return pd.DataFrame(records)


def plot_comparison(baseline,extended,pairs,bounds,out):
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),layout='constrained')
    for j,model in enumerate(('M0','M1')):
        ax=axes[0,j]
        for label,run,color in [('Baseline',baseline,'#777777'),('Extended',extended,'#2166ac')]:
            block=run['profiles'].query('model == @model and parameter == "I_req"').sort_values('value')
            comparable=block.comparable_to_global_best.astype(bool)
            ax.plot(block.value,block.profile_mae_days.where(comparable),'o-',color=color,
                    markersize=4,label=label)
        ax.axvline(90,color='#777777',linestyle='--',linewidth=1)
        ax.set(title=f'{model}: I_req profile',xlabel='I_req (°C h)',ylabel='Profile interval MAE (days)')
        ax.legend()
        ax=axes[1,j]
        block=pairs[pairs.model==model]
        x=np.arange(len(block));width=.36
        ax.bar(x-width/2,block.signed_interval_error_days_baseline,width,label='Baseline grid best',color='#999999')
        ax.bar(x+width/2,block.signed_interval_error_days_extended,width,label='Extended grid best',color='#2166ac')
        ax.axhline(0,color='black',linewidth=.8)
        ax.set(xticks=x,xticklabels=block.input_key,title=f'{model}: paired site-year errors',
               ylabel='Signed interval error (days)')
        ax.tick_params(axis='x',rotation=35)
        ax.legend(fontsize=8)
    fig.suptitle('I_req upper-bound exploration: baseline 90 → 200 °C h\nFull-data grid diagnostics; negative error = early, positive = late')
    fig.savefig(out/'Ireq_expansion_comparison.png',dpi=180)
    fig.savefig(out/'Ireq_expansion_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    for ax,model in zip(axes,('M0','M1')):
        block=bounds[bounds.model==model]
        ax.plot(block.upper_cap,block.best_mae_days,'o-',color='#2166ac')
        for row in block.itertuples():
            ax.annotate(f'I_req={row.best_I_req:g}',(row.upper_cap,row.best_mae_days),
                        xytext=(0,8),textcoords='offset points',fontsize=8,ha='center')
        ax.set(title=model,xlabel='Allowed I_req upper bound (°C h)',ylabel='Best grid MAE (days)')
        ax.margins(y=.25)
    fig.suptitle('Nested upper bounds: is the best value still at the boundary?\nPost hoc diagnostics within one expanded grid, not independent validations')
    fig.savefig(out/'nested_upper_bounds.png',dpi=180);fig.savefig(out/'nested_upper_bounds.pdf');plt.close(fig)


def write_report(base_path,expanded_path,baseline,extended,check,comparison,pairs,fixed,bounds,out):
    lines=['# I_req 上界扩展至 200 的诊断试验','',
           '**目的：检查原上界 90 是否限制参数搜索，以及扩展后显症时间错配是否得到一致改善。**','',
           '## 方法','',
           '- 使用单独配置 `config/fsim_Ireq200_exploration.csv`，只扩展 I_req；原运行配置、历史默认值和上一轮结果保留。',
           '- 保留原 7 个 I_req 值（45、52.5、60、67.5、75、82.5、90），新增 100–200、间隔10，共18个取值。VPD_w和u_tr的网格保持原样。',
           '- M0：162个组合；M1：1458个组合。逐组合保留全部六个站点年份，包括未产生预测的情况。另跑两个默认值对照。',
           '- 先优先保证预测数量，再比较区间距离；本文涉及的误差下降仅在相同完整覆盖下计算。逐站点年份比较用于识别平均值掩盖的恶化。',
           '- 同时比较原/新全网格最优结果，以及保持VPD_w和u_tr不变、仅扩大I_req搜索空间的结果。',
           '- 从扩展网格提取90、120、150、180、200五个嵌套上界，用于观察最优值是否持续追随边界。这些不是独立验证。',
           '- 近优集合采用0、0.5、1、2天描述性容差，不是置信区间。分段网格间距不同，集合数量不能解释为概率或参数空间体积。',
           '- 范围扩展是用户指定的探索试验，并非新增生物学参数证据。','',
           '## 原结果复现','',
           f'- 原范围共有 {check["n_common_grid_records"]} 条网格站点年份记录，连同默认对照共 {check["n_common_records"]} 条；预测状态、窗口、误差及窗口数量全部一致。',
           '- 气象输入哈希一致，生物过程与季节控制器源码一致。配置加载器只新增显式网格支持。','',
           '## 全网格比较','',
           '| 版本 | 原范围最小MAE（天） | 扩展后最小MAE（天） | 降低（天） | 扩展后最优I_req | 原/新预测数 |',
           '|---|---:|---:|---:|---:|---|']
    for row in comparison.itertuples():
        lines.append(f'| {row.model} | {row.baseline_conditional_mae_days:.3f} | {row.extended_conditional_mae_days:.3f} | {row.mae_reduction_days:.3f} | {row.extended_I_req:g} | {row.baseline_n_predictions}/{row.extended_n_predictions} |')
    lines+=['','这些最小值使用全部六站点年份的观测进行选择，只描述搜索范围内的拟合潜力，不能作为LOSYO成绩，也不能称为已识别的生物学参数。','',
            '## 逐站点年份变化','',
            '| 版本 | 站点年份 | 原有符号误差（天） | 扩展后有符号误差（天） | 绝对误差降低（天） |',
            '|---|---|---:|---:|---:|']
    for r in pairs.itertuples():
        lines.append(f'| {r.model} | {r.input_key} | {r.signed_interval_error_days_baseline:.3f} | {r.signed_interval_error_days_extended:.3f} | {r.absolute_error_reduction_days:.3f} |')
    lines+=['','负有符号误差表示预测偏早，正值表示偏晚；绝对误差降低为正表示改善。','']
    for model,g in pairs.groupby('model'):
        nbetter=int((g.absolute_error_reduction_days>1e-9).sum())
        nworse=int((g.absolute_error_reduction_days< -1e-9).sum())
        lines.append(f'- {model}：改善 {nbetter} 个、恶化 {nworse} 个、误差基本不变 {len(g)-nbetter-nworse} 个站点年份。')
    lines+=['','## 预测完整性与固定其他参数的检查','']
    for model,g in extended['metrics'].query('design == "grid"').groupby('model'):
        old=g[g.I_req<=90];new=g[g.I_req>90]
        lines.append(f'- {model}：原范围 {int((old.n_predictions==6).sum())}/{len(old)} 个组合覆盖六站点年份；新增范围 {int((new.n_predictions==6).sum())}/{len(new)} 个组合覆盖六站点年份。这些比例由网格决定，不是预测成功概率。')
        f=fixed[(fixed.model==model)&fixed.both_complete]
        lines.append(f'- {model}：保持其余参数不变且原/新均完整覆盖的 {len(f)} 个条件组合中，{int((f.mae_reduction_days>1e-9).sum())} 个因放宽I_req搜索而降低误差；最大降低 {f.mae_reduction_days.max():.3f} 天，中位数降低 {f.mae_reduction_days.median():.3f} 天。扩大嵌套搜索集合后的最优误差本来就不会增加，是否有科学意义仍需看改善量和折外结果。')
    lines+=['','## 边界与近优集合','',
            '| 版本 | 允许上界 | 最优I_req | 最小MAE（天） | 最优值是否在上界 |',
            '|---|---:|---:|---:|---|']
    for r in bounds.itertuples():
        lines.append(f'| {r.model} | {r.upper_cap:g} | {r.best_I_req:g} | {r.best_mae_days:.3f} | {r.best_at_cap} |')
    for model,g in extended['near'].query('tolerance_days == 1').groupby('model'):
        values=sorted(g.I_req.unique())
        lines+=['',f'- {model}：1天近优集合保留 {len(g)} 个网格组合，I_req离散值为 '+ '、'.join(f'{v:g}' for v in values)+'。']
    refinement = out/'conditional_refinement/refinement_metadata.json'
    if refinement.exists():
        fine_meta = json.loads(refinement.read_text())
        fine = pd.read_csv(out/'conditional_refinement/predictions.csv')
        changes = []
        for (model, site), block in fine.groupby(['model','input_key']):
            block = block.sort_values('I_req')
            for i in range(1,len(block)):
                before, after = block.iloc[i-1], block.iloc[i]
                changes.append({'model':model,'input_key':site,
                                'I_req_before':before.I_req,'I_req_after':after.I_req,
                                'predicted_start_before':before.predicted_start,
                                'predicted_start_after':after.predicted_start,
                                'absolute_error_reduction_days':before.interval_distance_days-after.interval_distance_days})
        changes = pd.DataFrame(changes)
        changes.to_csv(out/'conditional_refinement/adjacent_site_year_changes.csv',index=False)
        lines+=['','## 上界附近的条件细化','',
                '- 固定各版本扩展网格最优组合的其余参数，把I_req=180–200细化为步长1；另有252条主分析记录和84条重复天气策略对照。',
                '- 这是在已选择参数条件下的分辨率检查，不是联合细化或独立验证。']
        for result in fine_meta['results']:
            lines.append(f'- {result["model"]}：条件最小MAE={result["minimum_mae_days"]:.6f}天，对应I_req={result["minimizing_I_req_values"]}。')
        last = changes[(changes.model=='M0') & (changes.I_req_after==200)]
        different = last[~np.isclose(last.absolute_error_reduction_days,0,rtol=0,atol=1e-9)]
        lines.append(f'- M0从199到200时，平均绝对误差降低 {last.absolute_error_reduction_days.mean():.6f} 天；变化集中在 {len(different)} 个站点年份。M1最优风速为0时与M0相同。')
        for row in different.itertuples():
            lines.append(f'- {row.input_key}：预测窗口起点由{row.predicted_start_before}变为{row.predicted_start_after}，绝对误差降低 {row.absolute_error_reduction_days:.6f} 天。')
        lines+=['- 上界附近存在离散跳变，边界上的单一最佳网格点不能直接解释为稳定的生物学参数估计。详见`conditional_refinement/refinement_report.md`及图。']
    lines+=['','## 解释边界','',
            '- 新范围是否包含更好解，与该参数能否被可靠识别是不同问题；应结合剖面形状、近优集合宽度及站点间一致性判断。',
            '- 若最优值仍位于200，只能说本轮尚未找到范围内部的最优区域；本试验不自动继续扩大上界。',
            '- 不能把范围归一化的网格敏感性数值直接与上一轮比较：I_req归一化宽度由45变为155，且取样分布改变。',
            '- 本轮网格是分段离散的，不能排除网格点之间存在更低误差。后续若做局部细化或正式校准，必须限制在训练折内，并记录范围选择受全数据探索影响。','',
            '## 验证','',
            f'- M1零风速门控与M0对照：{extended["metadata"]["zero_gate_equivalence"]}。',
            f'- 重复年末天气记录处理策略对照：{extended["metadata"].get("duplicate_policy_check",{})}。',
            '- 原始天气JSON未修改。运行源码快照、参数表快照、输入指纹保存在结果目录。','',
            '## 文件','',
            '- `Ireq_expansion_comparison.png/pdf`：I_req剖面及逐站点年份有符号误差。',
            '- `nested_upper_bounds.png/pdf`：嵌套上界检查。',
            '- `range_comparison.csv`、`best_set_site_year_comparison.csv`：原/新最优网格的配对比较。',
            '- `fixed_other_parameters_comparison.csv`：保持VPD_w和u_tr不变的比较。',
            '- `nested_upper_bound_diagnostics.csv`：逐上界的误差、覆盖与边界状态。',
            '- `analysis_report.md`及通用响应面图：本次完整网格分析。','']
    (out/'Ireq_expansion_report.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'result/fsim_parameter_analysis_v1')
    p.add_argument('--expanded',type=Path,default=ROOT/'result/fsim_Ireq200_exploration')
    args=p.parse_args()
    baseline,extended=read_run(args.baseline),read_run(args.expanded)
    check=check_common_predictions(baseline,extended)
    comparison,pairs=compare_best(baseline,extended)
    fixed=compare_at_fixed_other_parameters(baseline,extended)
    bounds=boundary_checks(extended)
    out=args.expanded
    comparison.to_csv(out/'range_comparison.csv',index=False)
    pairs.to_csv(out/'best_set_site_year_comparison.csv',index=False)
    fixed.to_csv(out/'fixed_other_parameters_comparison.csv',index=False)
    bounds.to_csv(out/'nested_upper_bound_diagnostics.csv',index=False)
    plot_comparison(baseline,extended,pairs,bounds,out)
    write_report(args.baseline,args.expanded,baseline,extended,check,comparison,pairs,fixed,bounds,out)
    check['comparison_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'baseline_reproduction_check.json').write_text(json.dumps(check,indent=2))
    shutil.copy2(__file__,out/'source_snapshot/src'/Path(__file__).name)
    print(comparison.to_string(index=False))
    print(out/'Ireq_expansion_report.md')


if __name__=='__main__': main()
