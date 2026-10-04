# -*- coding: utf-8 -*-
import csv,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
from optics.stage02c_source import tree_sha
run=ROOT/'results/stage02f2/run_20261004_095323'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'));a=json.loads((HERE/'独立审核.json').read_text(encoding='utf-8'));e=json.loads((HERE/'失败路径验收.json').read_text(encoding='utf-8'));n=json.loads((HERE/'无保存验收.json').read_text(encoding='utf-8'))
assert r['status']=='completed' and r['validation_passed'] and r['planned_jobs']==r['completed_jobs']==12 and not r['budget_exhausted']
assert r['numerical_validation_passed'] and r['all_reconstructions_converged'] and r['certificate_validation_passed'] and a['passed'] and e['passed'] and n['passed']
log=(HERE/'完整回归.log').read_text(encoding='utf-8');assert 'Ran 232 tests' in log and log.rstrip().endswith('OK')
manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'));assert len(evidence['sources'])==10
for s in evidence['sources']:assert tree_sha(Path(s['path']))==s['sha256']
assert len(list((run/'arrays').glob('*.npz')))==len(list((run/'figures').glob('*.png')))==12
assert len(list((run/'metrics').glob('*_optimization.json')))==len(list((run/'metrics').glob('*_optimization.csv')))==12
with (run/'metrics/evaluation.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==12 and {a['identity'] for a in rows}=={a['identity'] for a in r['records']}
for row in rows:
    rec=next(a for a in r['records'] if a['identity']==row['identity'])
    assert row['status']==rec['solver_status'] and float(row['alpha'])==rec['alpha'] and float(row['cube_error'])==rec['evaluation']['cube_relative_l2'] and row['certificate_passed']=='True'
errors=[a['evaluation']['cube_relative_l2'] for a in r['records']]
result=dict(passed=True,formal_run=str(run),jobs=12,tests=232,test_seconds=203.320,formal_seconds=r['elapsed_s'],cube_error_range=[min(errors),max(errors)],max_certificate_relative=max(a['certificate']['distance_upper_bound_relative'] for a in r['records']),frozen_source_runs=10,source_and_code_fingerprints_unchanged=True,artifacts=dict(npz=12,png=12,history_json=12,history_csv=12),independent_checks=a['checks'],failure_entry_checks=e,no_save_check=n,environment=r['environment'],scope=r['scope'],limitations=['恢复质量仍差，数值通过不等于真值准确','固定权重和alpha规则仅为基础近似基线','理想控制和更多增益/种子未运行','真实GUI/PyCharm点击和相机标定未验证','论文PSF旋转方向对应、原论文网络、双孔径未完成'])
with (HERE/'最终审核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(result,ensure_ascii=False,indent=2))
