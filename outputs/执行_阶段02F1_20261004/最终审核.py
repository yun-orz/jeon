# -*- coding: utf-8 -*-
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'jeon2019_optics';HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from optics.stage02c_source import tree_sha
run=ROOT/'results/stage02f1/run_20261004_093359'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'));a=json.loads((HERE/'独立审核.json').read_text(encoding='utf-8'));e=json.loads((HERE/'入口验收.json').read_text(encoding='utf-8'))
assert r['status']=='completed' and r['validation_passed'] and r['completed_jobs']==r['planned_jobs']==18
assert all(v['passed'] for v in r['controls'].values()) and a['passed'] and e['passed']
log=(HERE/'完整回归.log').read_text(encoding='utf-8');assert 'Ran 223 tests' in log and log.rstrip().endswith('OK')
manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'));assert len(evidence['sources'])==9
for s in evidence['sources']:assert tree_sha(Path(s['path']))==s['sha256']
assert len(list((run/'arrays').glob('*.npz')))==18 and len(list((run/'figures').glob('*.png')))==18
with (run/'metrics/statistics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==108
for row in rows:
    rec=next(x for x in r['records'] if x['identity']==row['identity']);assert float(row['z'])==rec['statistics']['checks'][row['check']]['z'] and row['passed']=='True'
result=dict(passed=True,formal_run=str(run),jobs=18,statistic_repeats_per_job=64,tests=223,test_seconds=416.706,formal_seconds=r['elapsed_s'],max_abs_z=a['max_abs_z'],negative_readout_pixels=a['negative_readout_pixels'],frozen_source_runs=9,source_and_code_fingerprints_unchanged=True,artifacts=dict(npz=18,png=18,statistics_csv_rows=108),entry_checks=e,environment=r['environment'],scope=r['scope'],not_verified=['真实PyCharm点击与GUI交互','真实相机标定','噪声重建','论文PSF旋转方向对应','原论文网络','双孔径'])
(HERE/'最终审核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
