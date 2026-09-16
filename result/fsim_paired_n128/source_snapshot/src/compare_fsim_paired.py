#!/usr/bin/env python3
"""Evaluate M0 at the exact shared parameters of an existing M1 Sobol design."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.sobol_fsim import init_worker, run_task
from src.analyze_fsim_parameters import load_cases
from model.parameter_table import load_config


def common_design(design):
    """Deduplicate only exact shared-coordinate matches; never round parameters."""
    m1=design[design.model.eq('M1')].copy()
    common=m1[['VPD_w','I_req']].drop_duplicates().reset_index(drop=True)
    common['common_id']=[f'paired_common_{i:06d}' for i in range(len(common))]
    mapping=m1.merge(common,on=['VPD_w','I_req'],how='left',validate='many_to_one')
    if len(mapping)!=len(m1) or mapping.common_id.isna().any():
        raise ValueError('Invalid matching design')
    return common,mapping


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'result/fsim_sobol_n128')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'result/fsim_paired_n128')
    parser.add_argument('--workers',type=int,choices=[2,4],default=4)
    args=parser.parse_args();source=args.input_dir.resolve();out=args.output_dir.resolve()
    parent=json.loads((source/'run_metadata.json').read_text())
    if parent['status']!='complete': raise ValueError('Requires a complete parent run')
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new empty output directory')
    required={**parent['input_sha256'],**{k:v for k,v in parent['source_sha256'].items()
              if k.startswith('model/') or k=='src/analyze_fsim_parameters.py'}}
    for rel,expected in required.items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Parent/current model or input differs: {rel}')
    if parent.get('duplicate_policy_check',{}).get('n_differences')!=0:
        raise ValueError('Parent duplicate-weather check is absent or differs')
    design=pd.read_csv(source/'sobol_design.csv',float_precision='round_trip')
    common,mapping=common_design(design)
    cases,errors,context,alternate,repairs=load_cases(source/'site_year_manifest.csv')
    if errors: raise ValueError(errors)
    all_cases=cases+alternate
    base=load_config(source/'parameter_table.csv')
    out.mkdir(parents=True,exist_ok=True)
    common.to_csv(out/'unique_common_design.csv',index=False)
    mapping.to_csv(out/'paired_design.csv',index=False)
    repairs.to_csv(out/'weather_duplicate_audit.csv',index=False)
    context.to_csv(out/'weather_wind_context.csv',index=False)
    samples=common.rename(columns={'common_id':'sample_id'}).copy()
    samples['model']='M0';samples['u_tr']=np.nan;samples['design']='matched_M1_shared_parameters'
    samples['seed']=0;samples['design_index']=np.arange(len(samples))
    tasks=[(case_id,samples.iloc[start:start+32].to_dict('records'))
           for case_id in range(len(all_cases)) for start in range(0,len(samples),32)]
    metadata={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
              'parent_run':str(source),'parent_settings':parent['settings'],'workers':args.workers,
              'matching':'exact VPD_w and I_req, same site-year and fixed configuration; only wind gate differs',
              'n_M1_rows':len(mapping),'n_unique_common_rows':len(common),
              'n_new_main_evaluations':len(common)*len(cases),
              'n_new_duplicate_evaluations':len(common)*len(alternate),
              'verified_parent_model_input_sha256':required,
              'parent_results_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.csv')},
              'source_sha256':{}}
    for path in [Path(__file__),ROOT/'src/sobol_fsim.py',ROOT/'src/analyze_fsim_parameters.py']+list((ROOT/'model').glob('*.py')):
        rel=path.relative_to(ROOT);dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);metadata['source_sha256'][str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'{len(mapping)} matched M1 design rows; {len(common)} unique M0 settings; '
          f'{metadata["n_new_main_evaluations"]} main + {metadata["n_new_duplicate_evaluations"]} duplicate evaluations',flush=True)
    records=[]
    try:
        with tqdm(total=len(common)*len(all_cases),desc='Matched M0 evaluations',unit='eval',dynamic_ncols=True) as progress:
            with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(all_cases,base)) as pool:
                futures=[pool.submit(run_task,task) for task in tasks]
                for future in as_completed(futures):
                    batch=future.result();records.extend(batch)
                    with (out/'checkpoint.jsonl').open('a') as f:
                        f.write(pd.DataFrame(batch).to_json(orient='records',lines=True))
                    progress.update(len(batch))
        all_pred=pd.DataFrame(records)
        all_pred.to_csv(out/'all_m0_predictions.csv',index=False)
        if not all_pred.status.isin(['predicted','no_prediction']).all():
            raise ValueError('Runtime errors in matched evaluation')
        main=all_pred[all_pred.weather_policy.ne('keep_first')].copy()
        alt=all_pred[all_pred.weather_policy.eq('keep_first')].copy()
        if len(main)!=metadata['n_new_main_evaluations'] or len(alt)!=metadata['n_new_duplicate_evaluations']:
            raise ValueError('Incomplete paired evaluation')
        check=main.merge(alt,on=['sample_id','input_key'],validate='one_to_one',suffixes=('_last','_first'))
        same=np.ones(len(check),dtype=bool)
        for col in ['status','predicted_start','predicted_end']:
            same &= check[col+'_last'].fillna('MISSING').eq(check[col+'_first'].fillna('MISSING')).to_numpy()
        check.loc[~same].to_csv(out/'duplicate_policy_differences.csv',index=False)
        if not same.all(): raise ValueError('Duplicate policy changes matched M0 predictions')
        main.sort_values(['sample_id','input_key']).to_csv(out/'m0_predictions.csv',index=False)
        metadata.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                        runtime_errors=0,duplicate_policy_differences=int((~same).sum()))
    except Exception as exc:
        metadata.update(status='failed',error=str(exc));raise
    finally:
        (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(out/'m0_predictions.csv',flush=True)


if __name__=='__main__':main()
