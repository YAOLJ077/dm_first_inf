import sys,json,tempfile,unittest,threading,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import calibrate_m0_grouped_tpe as runner
import pandas as pd

def process_probe(directory):
    # Force 16 distinct workers to rendezvous, proving concurrent process capacity.
    import model.first_infection as model
    pid=os.getpid();start=time.monotonic()
    (Path(directory)/str(pid)).write_text('ready')
    while len(list(Path(directory).iterdir()))<16:
        if time.monotonic()-start>20:raise RuntimeError('16 workers did not start')
        time.sleep(.01)
    return pid,str(model.__file__)


class Checks(unittest.TestCase):
    def setUp(self):
        self.s=json.loads((runner.ROOT/'config/fsim_m0_grouped_tpe_4p.json').read_text())
        self.base=runner.load_config(runner.ROOT/self.s['parameter_table'])
        self.manifest=pd.read_csv(runner.ROOT/self.s['manifest']);self.keys=self.manifest.input_key.tolist()
        self.anchor=dict(VPD_w=2.,I_req=200.,RH_sp=80.,MP_sp=3)

    def test_four_dimensions_and_budget(self):
        runner.validate_settings(self.s)
        self.assertEqual(self.s['n_trials'],1024)
        self.assertEqual(self.s['n_startup_trials'],128)
        self.assertEqual(self.s['n_tpe_trials'],896)
        study=runner.make_study(1,{**self.s,'n_startup_trials':3},self.anchor)
        values=[]
        for i in range(12):
            t,p=runner.ask_params(study,self.s['search_space'])
            self.assertEqual(set(t.params),set(runner.NAMES))
            self.assertIsInstance(p['MP_sp'],int);values.append(p['RH_sp'])
            runner.tell(study,t,dict(n_no_prediction=0,training_MAE_days=p['VPD_w'],optimizer_value=p['VPD_w']))
        self.assertTrue(any(v!=80 for v in values))

    def test_group_selection_resume_and_no_leakage(self):
        s={**self.s,'n_trials':8,'n_startup_trials':3,'n_tpe_trials':5,'checkpoints':[3,8]}
        for scheme in ['site','year']:
            groups=runner.make_groups(self.manifest,scheme)
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);calls=[]
                outer=self
                class Fake:
                    base=outer.base
                    def evaluate(inner,params,keys):
                        calls.append(keys)
                        # Tests have 1/2/3 cases; identify by completed selection,
                        # not length, since DA training also contains 3 cases.
                        is_test=any(keys==test and (out/'folds'/g/'selection.json').exists() for g,test in groups.items())
                        if is_test:
                            group=next(g for g,test in groups.items() if keys==test)
                            selection=json.loads((out/'folds'/group/'selection.json').read_text())
                            outer.assertFalse(set(keys)&set(selection['training_keys']))
                            outer.assertEqual(params,{n:selection[n] for n in runner.NAMES})
                        return [dict(input_key=k,status='predicted',signed_interval_error_days=999 if is_test else params['VPD_w']) for k in keys],[]
                anchors={g:self.anchor for g in groups}
                summary,count,_=runner.run_groups(s,self.keys,groups,anchors,out,Fake(),False)
                self.assertEqual(summary['n_folds'],3);self.assertEqual(summary['n_site_years'],6)
                self.assertEqual(summary['MAE_days'],999);self.assertEqual(count,12*8+6)
                calls.clear();runner.run_groups(s,self.keys,groups,anchors,out,Fake(),False)
                self.assertEqual(calls,[])

    def test_six_concurrent_folds_match_sequential_search(self):
        s={**self.s,'n_trials':8,'n_startup_trials':3,'n_tpe_trials':5,'checkpoints':[3,8]}
        designs={scheme:runner.make_groups(self.manifest,scheme) for scheme in ['site','year']}
        anchors={scheme:{g:self.anchor for g in groups} for scheme,groups in designs.items()}
        outer=self
        class Fake:
            base=outer.base
            def __init__(inner,concurrent=False):
                inner.stop=threading.Event();inner.local=threading.local()
                inner.barrier=threading.Barrier(6) if concurrent else None
                inner.calls=0;inner.lock=threading.Lock()
            def evaluate(inner,p,keys):
                if inner.barrier is not None and not getattr(inner.local,'started',False):
                    inner.local.started=True;inner.barrier.wait(timeout=10)
                with inner.lock:inner.calls+=1
                return [dict(input_key=k,status='predicted',signed_interval_error_days=p['VPD_w']+p['RH_sp']/100) for k in keys],[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parallel=Fake(True)
            actual=runner.run_parallel_folds(s,self.keys,designs,anchors,root/'parallel',parallel,False)
            sequential=Fake()
            for scheme,groups in designs.items():
                expected=runner.run_groups(s,self.keys,groups,anchors[scheme],root/'sequential'/scheme,sequential,False)
                self.assertEqual(actual[scheme],expected)
                for group in groups:
                    for file in (root/'parallel'/scheme/'folds'/group).glob('seed_*/trial_*.json'):
                        other=root/'sequential'/file.relative_to(root/'parallel')
                        self.assertEqual(json.loads(file.read_text()),json.loads(other.read_text()))
            # Replay in parallel must not call the model again.
            replay=Fake()
            self.assertEqual(actual,runner.run_parallel_folds(s,self.keys,designs,anchors,root/'parallel',replay,False))
            self.assertEqual(replay.calls,0)

    def test_sixteen_distinct_concurrent_processes(self):
        evaluator=runner.Evaluator([],[],self.base,16)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                results=list(evaluator.pool.map(process_probe,[tmp]*16))
                self.assertEqual(len({pid for pid,_ in results}),16)
                self.assertTrue(all(Path(path).is_relative_to(runner.REFERENCE) for _,path in results))
                print('Verified 16 simultaneous distinct worker PIDs',flush=True)
        finally:evaluator.close()

    def test_four_processes_variable_training_and_failures(self):
        weather=pd.DataFrame(dict(datetime=pd.date_range('2024-01-01',periods=48,freq='h').astype(str),air_temp=20.,relative_humidity=50.,precip_acc_period_adjusted=0.,wind_speed=2.))
        rb=dict(year=2024,BBCH_15='2024-01-01',weather_hourly=weather.to_dict('records'))
        cases=[dict(input_key=k,L='2024-01-01',U='2024-01-02',rb=rb) for k in self.keys]
        evaluator=runner.Evaluator(cases,[],self.base,16)
        try:
            records,_=evaluator.evaluate({**self.anchor,'RH_sp':91.2,'MP_sp':7},self.keys[:3])
            self.assertTrue(all(r['status']=='no_prediction' and r['MP_sp']==7 for r in records))
            m=runner.training_metrics(records,self.keys[:3]);self.assertEqual(m['n_no_prediction'],3)
            self.assertIsNone(m['training_MAE_days']);self.assertFalse(m['feasible'])
        finally:evaluator.close()

if __name__=='__main__':unittest.main()
