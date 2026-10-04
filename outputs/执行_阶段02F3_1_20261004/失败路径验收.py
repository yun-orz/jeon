# -*- coding: utf-8 -*-
import copy,json,os,sys,subprocess,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
import main_stage02f3_1 as m
c=json.loads((ROOT/'config_stage02f3_1.json').read_text(encoding='utf-8'));small=copy.deepcopy(c);small['devices']=['continuous'];small['scenes']=['coincident_points'];small['methods']=['unweighted']
result={}
def cli(name,config):
    p=HERE/(name+'_config.json')
    with p.open('x',encoding='utf-8') as f:json.dump(config,f)
    r=subprocess.run([sys.executable,'-B',str(ROOT/'main_stage02f3_1.py'),'--config',str(p)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    with (HERE/(name+'.log')).open('xb') as f:f.write(r.stdout)
    return r.returncode
bad=copy.deepcopy(small);bad['source_run']='results/nonexistent_F3_source';result['bad_source_exit']=cli('错误来源',bad);assert result['bad_source_exit']!=0
budget=copy.deepcopy(small);budget['budget_seconds']=1e-9;result['budget_exit']=cli('预算截断',budget);assert result['budget_exit']==2
# 保留各篡改副本，仅修改副本单个参数，不动正式run。
for name,key in [('冻结alpha篡改','alpha'),('冻结权重篡改','weights')]:
    dest=HERE/(name+'副本');shutil.copytree(ROOT/c['source_run'],dest)
    p=next((dest/'arrays').glob('*.npz'))
    with np.load(p,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    values[key]=values[key]*1.01
    np.savez_compressed(p,**values)
    bad=copy.deepcopy(small);bad['source_run']=str(dest);result[key+'_tamper_exit']=cli(name,bad);assert result[key+'_tamper_exit']!=0
import matplotlib.pyplot as plt
with patch.object(plt,'show') as show:
    report=m.run(small,no_save=True,show_plots=True);assert report['validation_passed'];assert show.call_count==1
assert not plt.get_fignums();result['mock_show_once_closed']=True
original_allocate=m.allocate_unique_run_dir;created=[]
def allocate(stage):
    p=original_allocate(stage);created.append(p);return p
with patch.object(m,'allocate_unique_run_dir',side_effect=allocate),patch.object(plt,'show',side_effect=RuntimeError('注入显示故障')):
    try:m.run(small,show_plots=True)
    except RuntimeError:pass
    else:raise AssertionError('显示故障未传播')
assert (created[-1]/'failed.json').exists() and not (created[-1]/'metrics/validation.json').exists() and not plt.get_fignums();result['show_failure_no_completed_closed']=True
original=m.write_json
def broken(path,value):
    if path.name=='validation.json':raise OSError('注入报告写入故障')
    return original(path,value)
with patch.object(m,'allocate_unique_run_dir',side_effect=allocate),patch.object(m,'write_json',side_effect=broken):
    try:m.run(small)
    except OSError:pass
    else:raise AssertionError('报告故障未传播')
assert (created[-1]/'failed.json').exists() and not (created[-1]/'metrics/validation.json').exists() and not plt.get_fignums();result['report_failure_no_completed_closed']=True
# 在控制求解接口注入一轮迭代限制；不是默认配置或真实CLI改参。
original_solve=m.solve_fixed_target
def limited(*args,**kwargs):
    mutable=list(args);solver=copy.deepcopy(mutable[8]);solver['max_iterations']=1;mutable[8]=solver
    return original_solve(*mutable,**kwargs)
with patch.object(m,'solve_fixed_target',side_effect=limited):report=m.run(small,no_save=True)
assert report['status']=='incomplete' and not report['validation_passed'] and report['records'][0]['solver_status']=='iteration_limit';result['injected_iteration_limit_not_completed']=True
result.update(passed=True,real_gui_verified=False,failed_run_paths=[str(p) for p in created])
with (HERE/'失败路径验收.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
