#!/usr/bin/env python3
"""Conditional I_req resolution check at each expanded-grid best combination.

Hold VPD_w and u_tr fixed; evaluate I_req=180,...,200 at unit increments.
This is not a joint fine-grid optimum or an independent validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from model.parameter_table import load_config
from src.analyze_fsim_parameters import load_cases, evaluate_group, summarize
from src.compare_fsim_Ireq_expansion import select_best
from src.report_fsim_parameter_analysis import plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir',type=Path,default=ROOT/'result/fsim_Ireq200_exploration')
    args=parser.parse_args();out=args.result_dir
    metadata=json.loads((out/'run_metadata.json').read_text())
    if metadata['status']!='complete': raise SystemExit('Complete expanded grid first')
    for name,digest in {**metadata['input_sha256'],**metadata['source_sha256']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Input or evaluated source changed: {name}')
    fine_dir=out/'conditional_refinement'
    fine_dir.mkdir(exist_ok=False)
    cfg=load_config(out/'parameter_table.csv')
    cases,errors,_,duplicates,_=load_cases(out/'site_year_manifest.csv')
    if errors: raise ValueError(errors)
    metrics=pd.read_csv(out/'sample_metrics.csv')
    design=[];settings=[]
    for model in ('M0','M1'):
        best=select_best(metrics[(metrics.model==model)&(metrics.design=='grid')])
        settings.append({'model':model,'VPD_w':best.VPD_w,'u_tr':None if pd.isna(best.u_tr) else best.u_tr,
                         'coarse_best_I_req':best.I_req,'coarse_best_mae_days':best.conditional_mae_days})
        for value in range(180,201):
            design.append({'sample_id':f'{model}_fine_{value}','model':model,'design':'conditional_refinement',
                           'VPD_w':best.VPD_w,'u_tr':best.u_tr,'I_req':value})
    frame=pd.DataFrame(design)
    frame.to_csv(fine_dir/'design.csv',index=False)
    tasks=[(case,block.to_dict('records'),cfg) for case in cases+duplicates
           for _,block in frame.groupby('model')]
    with ProcessPoolExecutor(max_workers=4) as pool:
        records=[record for batch in pool.map(evaluate_group,tasks) for record in batch]
    all_predictions=pd.DataFrame(records)
    all_predictions.to_csv(fine_dir/'all_predictions.csv',index=False)
    if (all_predictions.status=='runtime_error').any():
        raise RuntimeError('Runtime errors in conditional profile; inspect all_predictions.csv')
    predictions=all_predictions[all_predictions.weather_policy!='keep_first'].copy()
    alternate=all_predictions[all_predictions.weather_policy=='keep_first']
    summary=summarize(predictions)
    predictions.to_csv(fine_dir/'predictions.csv',index=False)
    summary.to_csv(fine_dir/'sample_metrics.csv',index=False)
    checks={}
    for label,other,keys in [
        ('duplicate_policy',alternate,['sample_id','input_key']),
        ('coarse_grid',pd.read_csv(out/'predictions.csv').query('design == "grid"'),
         ['model','VPD_w','u_tr','I_req','input_key'])]:
        paired=predictions.merge(other,on=keys,suffixes=('_fine','_other'),validate='one_to_one')
        same=np.ones(len(paired),dtype=bool)
        for col in ('status','predicted_start','predicted_end'):
            same &= (paired[col+'_fine'].fillna('MISSING')==paired[col+'_other'].fillna('MISSING')).to_numpy()
        if not len(paired) or not same.all(): raise ValueError(f'Resolution check failed: {label}')
        checks[label]={'n_pairs':len(paired),'all_equal':bool(same.all())}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
    results=[]
    for ax,model in zip(axes,('M0','M1')):
        block=summary[summary.model==model].sort_values('I_req')
        maxn=block.n_predictions.max()
        eligible=block[block.n_predictions==maxn]
        best=eligible.conditional_mae_days.min()
        optimal=eligible[np.isclose(eligible.conditional_mae_days,best,rtol=0,atol=1e-10)]
        near=eligible[eligible.conditional_mae_days<=best+1+1e-12]
        results.append({'model':model,'max_n_predictions':int(maxn),'minimum_mae_days':best,
                        'minimizing_I_req_values':optimal.I_req.tolist(),
                        'within_one_day_I_req_values':near.I_req.tolist()})
        ax.plot(block.I_req,block.all_site_year_mae_days,'o-',color='#2166ac',markersize=4)
        ax.set(title=model,xlabel='I_req (°C h)',ylabel='All-six interval MAE (days)',xticks=[180,185,190,195,200])
    fig.suptitle('Conditional boundary refinement: unit I_req increments\nVPD_w and u_tr held at the expanded-grid best values')
    fig.savefig(fine_dir/'conditional_Ireq_profile.png',dpi=180)
    fig.savefig(fine_dir/'conditional_Ireq_profile.pdf');plt.close(fig)
    record={'purpose':'post hoc conditional resolution check, not joint calibration',
            'I_req_values':list(range(180,201)),'fixed_settings':settings,
            'n_main_records':len(predictions),'n_duplicate_records':len(alternate),
            'results':results,'checks':checks,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (fine_dir/'refinement_metadata.json').write_text(json.dumps(record,indent=2))
    shutil.copy2(__file__,fine_dir/Path(__file__).name)
    lines=['# I_req 上界附近的条件细化检查','',
           '固定扩展网格各版本最优组合的VPD_w和u_tr，只把I_req=180–200细化为步长1。此为条件剖面，不是联合细网格最优解或独立验证。','']
    for fixed,result in zip(settings,results):
        lines += [f'- {result["model"]}：VPD_w={fixed["VPD_w"]:g}，u_tr={fixed["u_tr"]}；最小MAE={result["minimum_mae_days"]:.6f}天，预测数={result["max_n_predictions"]}/6。',
                  '- 条件最小误差对应I_req：'+ '、'.join(str(v) for v in result['minimizing_I_req_values'])+'。',
                  '- 距条件最小误差不超过1天的I_req：'+ '、'.join(str(v) for v in result['within_one_day_I_req_values'])+'。','']
    lines += [f'复现检查：{checks}。', '',
              '若多个相邻I_req值产生同样最小误差，粗网格上的单一最佳点不能被解释为参数已唯一识别。结果只适用于本次固定的其余参数及离散步长。']
    (fine_dir/'refinement_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
