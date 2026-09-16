#!/usr/bin/env python3
"""Compare a LOSYO refinement extension against its unchanged earlier stages."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'result/fsim_m0_losyo_Ireq40_400')
    p.add_argument('--refined',type=Path,default=ROOT/'result/fsim_m0_losyo_Ireq40_400_refine3')
    args=p.parse_args();old=args.baseline.resolve();new=args.refined.resolve()
    a=json.loads((old/'run_metadata.json').read_text());b=json.loads((new/'run_metadata.json').read_text())
    if a['status']!='complete' or b['status']!='complete':raise ValueError('Both runs must be complete')
    s0=a['settings'];s1=b['settings']
    changed={k for k in set(s0)|set(s1) if s0.get(k)!=s1.get(k)}
    if not changed<={'refinement_rounds','max_candidates_per_fold','workers'}:raise ValueError(f'Other settings changed: {changed}')
    if a['bounds']!=b['bounds'] or a['input_sha256']!=b['input_sha256']:raise ValueError('Input or parameter ranges changed')
    for name in ['parameter_table.csv','site_year_manifest.csv']:
        if (old/name).read_bytes()!=(new/name).read_bytes():raise ValueError(f'{name} differs')
    # Runtime source hashes are keyed by absolute paths in these runs.
    for path,h in a['source_sha256'].items():
        if '/model/' in path or path.endswith(('calibrate_m0_losyo.py','analyze_fsim_parameters.py')):
            if b['source_sha256'].get(path)!=h:raise ValueError(f'Model/calibration implementation differs: {path}')
    before=pd.read_csv(old/'fold_summary.csv');after=pd.read_csv(new/'fold_summary.csv')
    if set(before.held_out)!=set(after.held_out) or len(before)!=6 or len(after)!=6:raise ValueError('Incomplete fold lists')
    preds0=pd.read_csv(old/'oof_predictions.csv').set_index('input_key')
    preds1=pd.read_csv(new/'oof_predictions.csv').set_index('input_key')
    rows=[]
    for fold in before.held_out:
        f0=before.set_index('held_out').loc[fold];f1=after.set_index('held_out').loc[fold]
        for stage in range(s0['refinement_rounds']+1):
            q0=pd.read_csv(old/'folds'/fold/f'loss_surface_stage_{stage}.csv')
            q1=pd.read_csv(new/'folds'/fold/f'loss_surface_stage_{stage}.csv')
            cols=['VPD_w','I_req','n_training_predictions','feasible','training_MAE_days','training_RMSE_days','first_stage']
            pd.testing.assert_frame_equal(q0[cols].sort_values(['VPD_w','I_req']).reset_index(drop=True),
                                          q1[cols].sort_values(['VPD_w','I_req']).reset_index(drop=True),
                                          check_exact=False,rtol=1e-12,atol=1e-12)
        improvement=f0.best_training_MAE_days-f1.best_training_MAE_days
        if improvement < -1e-9:raise ValueError('Nested refinement worsened minimum training loss')
        row={'held_out':fold,'old_candidates':int(f0.n_candidates),'new_candidates':int(f1.n_candidates),
             'old_training_MAE_days':f0.best_training_MAE_days,'new_training_MAE_days':f1.best_training_MAE_days,
             'training_improvement_days':improvement,'old_tied_best':int(f0.n_tied_best),'new_tied_best':int(f1.n_tied_best)}
        for label,f,pred,root in [('old',f0,preds0.loc[fold],old),('new',f1,preds1.loc[fold],new)]:
            row.update({f'{label}_{name}':f[name] for name in ['VPD_w','I_req']})
            row[f'{label}_oof_status']=pred.status
            row[f'{label}_oof_signed_error_days']=pred.signed_interval_error_days
            row[f'{label}_oof_absolute_error_days']=abs(pred.signed_interval_error_days)
            history=pd.read_csv(root/'folds'/fold/'refinement_history.csv')
            row[f'{label}_last_stage_improvement_days']=history.iloc[-1].improvement_days
            row[f'{label}_stop_reason']=f.stopping_reason
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(new/'refinement_comparison.csv',index=False)
    aggregate=[]
    summaries=[json.loads((root/'oof_summary.json').read_text()) for root in [old,new]]
    for key in summaries[0]:aggregate.append({'metric':key,'baseline':summaries[0][key],'refined':summaries[1][key]})
    pd.DataFrame(aggregate).to_csv(new/'oof_comparison.csv',index=False)
    lines=['# M0 LOSYO 加密对照','',
           f'原运行：`{old}`。加密运行：`{new}`。',
           '核对通过：模型实现、输入、范围和选参规则一致；新增加密之前的各折损失面逐点复现原运行。',
           '本轮仅增加网格分辨率；所有加密与选参仍仅使用五个训练站点，不根据留出表现选择参数。','',
           '| 留出站点 | 原训练MAE | 加密训练MAE | 训练改善（天） | 原留出有符号误差 | 加密留出有符号误差 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in frame.itertuples():lines.append(f'| {r.held_out} | {r.old_training_MAE_days:.3f} | {r.new_training_MAE_days:.3f} | {r.training_improvement_days:.3f} | {r.old_oof_signed_error_days:.3f} | {r.new_oof_signed_error_days:.3f} |')
    lines+=['','负误差表示预测偏早，正误差表示偏晚。训练损失下降不保证留出误差下降；并列参数选择也可能随网格分辨率改变。',
            '近优集合为已搜索参数集合，不是置信区间。局部加密不能排除粗网格之间尚未发现的狭窄低误差区域。','',
            '| OOF指标 | 原运行 | 加密运行 |','|---|---:|---:|']
    for row in aggregate:lines.append(f'| {row["metric"]} | {row["baseline"]} | {row["refined"]} |')
    (new/'refinement_comparison.md').write_text('\n'.join(lines)+'\n')
    audit={'baseline':str(old),'refined':str(new),'comparison_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'old_stages_reproduced':True,'comparison_inputs_sha256':{}}
    for root in [old,new]:
        for file in ['run_metadata.json','fold_summary.csv','oof_predictions.csv','oof_summary.json']:
            audit['comparison_inputs_sha256'][str(root/file)]=hashlib.sha256((root/file).read_bytes()).hexdigest()
    (new/'comparison_metadata.json').write_text(json.dumps(audit,indent=2))
    print(frame.to_string(index=False));print(pd.DataFrame(aggregate).to_string(index=False))


if __name__=='__main__':main()
