# -*- coding: utf-8 -*-
"""入口分模式与有限圆盘边界反例；不删除或覆盖历史结果。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent/'jeon2019_optics'
sys.path.insert(0,str(PROJECT))
import main_stage02b as M
from optics.coordinates import make_grid
from optics.psf_analysis import find_first_radius_reaching
sys.stdout.reconfigure(encoding='utf-8')
go=make_grid(301,1e-6)
X,Y=go.meshgrid()
# 合法非负测试强度全部位于方窗角落，完整圆盘内能量严格为零。
I=(np.hypot(X,Y)>155e-6).astype(float)
pin=float(I.sum()*go.cell_area)
corner=find_first_radius_reaching(I,go,pin,.8)
run=PROJECT/'results/stage02b1/run_20261001_224235'
met=json.loads((run/'metrics/psf_metrics.json').read_text('utf-8'))
m=met['records']['jeon@540nm']
cmd=[sys.executable,'-B',str(PROJECT/'main_stage02b.py')]
env=dict(os.environ,PYTHONIOENCODING='utf-8')
results={}
for label,args in [('height',['--only','height','--run-dir',str(ROOT/('partial_height_'+uuid.uuid4().hex[:8]))]),
                   ('analyze',['--only','analyze','--from-run',str(run),'--no-save'])]:
    r=subprocess.run(cmd+args,cwd=ROOT.parent.parent,env=env,capture_output=True,text=True,encoding='utf-8')
    (ROOT/(label+'_entry.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    results[label]={'exit_code':r.returncode,'argv':cmd+args}
    if label=='height':
        p=Path(args[-1])/'completion.json'
        results[label]['completion']=json.loads(p.read_text('utf-8')) if p.exists() else None

def snapshot():
    return {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in PROJECT.rglob('*') if p.is_file()}
before=snapshot()
r=subprocess.run(cmd+['--no-save'],cwd=ROOT.parent.parent,env=env,capture_output=True,text=True,encoding='utf-8')
after=snapshot()
(ROOT/'no_save_entry.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
results['no_save']={'exit_code':r.returncode,'file_snapshot_equal':before==after,
                    'changed':[k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)]}
cfg=M.load_and_validate_config(PROJECT/'config_stage02b.json')
results['required_all_has_analysis']='analysis_completed' in M.required_checks_02b('all',True)
results['corner_radius']=corner
results['jeon540_Eabs_at_Rlimit_reported']=m['R80']['Eabs_at_Rlimit']
results['jeon540_Eabs_150_actual']=next(e['Eabs'] for e in m['abs_encircled'] if round(e['R_um'])==150)
(ROOT/'mode_probes.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('角落反例R80:',corner['radius_um'],'标记',corner['status'],'正确应null')
print('540nm Eabs_at_Rlimit报告/实际:',results['jeon540_Eabs_at_Rlimit_reported'],results['jeon540_Eabs_150_actual'])
print('height退出/状态:',results['height']['exit_code'],results['height']['completion']['status'])
print('analyze退出:',results['analyze']['exit_code'])
print('no-save:',results['no_save'])
