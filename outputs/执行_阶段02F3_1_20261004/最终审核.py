# -*- coding: utf-8 -*-
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
from optics.stage02c_source import tree_sha
run=ROOT/'results/stage02f3_1/run_20261004_101012';first=ROOT/'results/stage02f3_1/run_20261004_100605'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'));a=json.loads((HERE/'独立审核_版式复核.json').read_text(encoding='utf-8'));e=json.loads((HERE/'复核/失败路径验收.json').read_text(encoding='utf-8'));n=json.loads((HERE/'无保存验收.json').read_text(encoding='utf-8'))
assert r['status']=='completed' and r['validation_passed'] and r['planned_jobs']==r['completed_jobs']==12 and not r['budget_exhausted']
assert r['numerical_validation_passed'] and r['all_controls_converged'] and r['certificate_validation_passed'] and a['passed'] and e['passed'] and n['passed']
assert r['noisy_verified_reused_count']==12 and r['noisy_new_iterations']==0 and a['all_baseline_dominant_under_optimization_bounds']
log=(HERE/'完整回归.log').read_text(encoding='utf-8');assert 'Ran 241 tests' in log and log.rstrip().endswith('OK')
manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'));assert len(evidence['sources'])==11
for s in evidence['sources']:assert tree_sha(Path(s['path']))==s['sha256']
for folder,pattern in [('arrays','*.npz'),('figures','*.png'),('metrics','*_optimization.json'),('metrics','*_optimization.csv'),('metrics','*_decomposition.json')]:assert len(list((run/folder).glob(pattern)))==12
# 版式调整前后的全部原始数组逐值一致，旧目录仍保留。
for p in (run/'arrays').glob('*.npz'):
    with np.load(p,allow_pickle=False) as z,np.load(first/'arrays'/p.name,allow_pickle=False) as old:assert set(z.files)==set(old.files) and all(np.array_equal(z[k],old[k]) for k in z.files)
with (run/'metrics/decomposition.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==12 and {v['identity'] for v in rows}=={v['identity'] for v in r['records']}
for row in rows:
    rec=next(v for v in r['records'] if v['identity']==row['identity']);d=rec['decomposition']
    assert row['status']=='converged' and float(row['alpha'])==rec['alpha']
    for key,name in [('total_relative','total'),('baseline_relative','baseline'),('noise_perturbation_relative','noise_perturbation')]:assert float(row[key])==d['norms_relative'][name]
    assert float(row['cross_relative_squared'])==d['cross_term_relative_squared']
    for prefix,key in [('baseline','baseline_interval'),('noise','noise_perturbation_interval')]:
        for bound in ['lower','upper']:assert float(row[prefix+'_'+bound])==d[key][bound+'_relative']
baseline=[v['decomposition']['norms_relative']['baseline'] for v in r['records']];noise=[v['decomposition']['norms_relative']['noise_perturbation'] for v in r['records']]
result=dict(passed=True,formal_run=str(run),retained_first_run=str(first),jobs=12,verified_reused_noisy_solutions=12,noisy_new_iterations=0,tests=241,test_seconds=205.106,formal_seconds=r['elapsed_s'],baseline_relative_range=[min(baseline),max(baseline)],noise_perturbation_relative_range=[min(noise),max(noise)],all_baseline_dominant_under_optimization_bounds=True,layout_only_arrays_unchanged=True,frozen_source_runs=11,source_and_code_fingerprints_unchanged=True,artifacts=dict(npz=12,png=12,history_json=12,history_csv=12,decomposition_json=12),independent_checks=a['checks'],failure_checks=e,no_save_checks=n,environment=r['environment'],scope=r['scope'],limitations=['单次实现与固定W/alpha下的理想控制诊断，不是纯岭偏差或统计置信','下一阶段留出alpha选择尚未执行','真实GUI/PyCharm点击/相机标定未验证','论文PSF旋转方向对应/原网络/双孔径未完成'])
with (HERE/'最终审核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(result,ensure_ascii=False,indent=2))
