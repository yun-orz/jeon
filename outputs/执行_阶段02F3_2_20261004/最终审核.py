# -*- coding: utf-8 -*-
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
from optics.stage02c_source import tree_sha
run=ROOT/'results/stage02f3_2/run_20261004_231220'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'));a=json.loads((HERE/'独立审核.json').read_text(encoding='utf-8'));e=json.loads((HERE/'失败路径验收.json').read_text(encoding='utf-8'));n=json.loads((HERE/'无保存验收.json').read_text(encoding='utf-8'))
assert r['status']=='completed' and r['validation_passed'] and r['planned_jobs']==r['completed_jobs']==5 and not r['budget_exhausted']
assert a['passed'] and e['passed'] and n['passed'] and r['config']['alpha_factors']==[.01,.001,.0001,.00001]
assert r['train_pixel_count']==7373 and r['validation_pixel_count']==1843 and len(r['candidate_records'])==4
log=(HERE/'完整回归.log').read_text(encoding='utf-8');assert 'Ran 250 tests' in log and log.rstrip().endswith('OK')
manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'));assert len(evidence['sources'])==12
for s in evidence['sources']:assert tree_sha(Path(s['path']))==s['sha256']
assert len(list((run/'arrays').glob('*.npz')))==8 and len(list((run/'figures').glob('*.png')))==2
assert len(list((run/'metrics').glob('*_optimization.json')))==len(list((run/'metrics').glob('*_optimization.csv')))==5
for rec in r['candidate_records']+[r['full_refit']]:
    name=rec['identity'];trajectory=json.loads((run/'metrics'/(name+'_optimization.json')).read_text(encoding='utf-8'));h=trajectory['history']
    assert trajectory['alpha']==rec['alpha'] and trajectory['status']==rec['solver_status']=='converged' and len(h)==rec['iterations'] and h[-1]==rec['final_optimization'] and rec['certificate_passed'] and rec['numerical_validation_passed'] and rec['initialization']=='zeros'
    with (run/'metrics'/(name+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(h)
    for row,step in zip(rows,h):
        for k,v in step.items():
            if isinstance(v,bool):assert row[k]==str(v)
            elif v is None:assert row[k]==''
            else:assert float(row[k])==v
    if name.startswith('candidate'):
        with np.load(run/'arrays'/(name+'.npz'),allow_pickle=False) as z:assert not any('truth' in k or 'ideal' in k for k in z.files)
assert a['selected_factor']==r['selection']['selected_factor']==1e-5 and r['selection']['truth_used'] is False and r['selection']['ideal_measurement_used'] is False
assert r['evaluation']['baseline_execution']=='reused_verified' and r['evaluation']['baseline_new_iterations']==0
result=dict(passed=True,formal_run=str(run),new_jobs=5,candidates=4,refits=1,tests=250,test_seconds=274.725,formal_seconds=r['elapsed_s'],training_pixels=7373,validation_pixels=1843,selected_factor=r['selection']['selected_factor'],selected_at_grid_boundary=True,selected_alpha_train=r['candidate_records'][-1]['alpha'],selected_alpha_full=r['full_refit']['alpha'],final_cube_error=a['final_cube_error'],baseline_cube_error=a['baseline_cube_error'],final_error_optimization_interval=a['final_error_optimization_interval'],baseline_error_optimization_interval=a['baseline_error_optimization_interval'],frozen_source_runs=12,source_and_code_fingerprints_unchanged=True,artifacts=dict(npz=8,png=2,history_json=5,history_csv=5),independent_checks=a['checks'],failure_checks=e,no_save_checks=n,environment=r['environment'],scope=r['scope'],limitations=['仅一个场景一次随机拆分，验证预测准则不等于光谱cube误差','选中最弱网格边界，不证明网格外最优','恢复相对误差仍约0.68','真实GUI/PyCharm点击/相机标定/论文旋转方向对应/网络/双孔径未验证'])
with (HERE/'最终审核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(result,ensure_ascii=False,indent=2))
