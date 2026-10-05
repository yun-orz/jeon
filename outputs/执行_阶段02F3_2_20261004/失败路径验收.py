# -*- coding: utf-8 -*-
import copy,json,os,sys,subprocess,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
import main_stage02f3_2 as m
c=json.loads((ROOT/'config_stage02f3_2.json').read_text(encoding='utf-8'));small=copy.deepcopy(c);small['alpha_factors']=[.01]
result={}
def cli(name,config):
    p=HERE/(name+'_config.json')
    with p.open('x',encoding='utf-8') as f:json.dump(config,f)
    before=set((ROOT/'results/stage02f3_2').glob('run_*'))
    r=subprocess.run([sys.executable,'-B',str(ROOT/'main_stage02f3_2.py'),'--config',str(p)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    with (HERE/(name+'.log')).open('xb') as f:f.write(r.stdout)
    after=set((ROOT/'results/stage02f3_2').glob('run_*'));created=after-before
    # 并行正式实验可能也新建目录，只取本CLI的config身份。
    actual=[]
    for path in created:
        cfg=path/'config_effective.json'
        if cfg.exists() and json.loads(cfg.read_text(encoding='utf-8'))['alpha_factors']==config['alpha_factors']:actual.append(path)
    return r.returncode,actual
bad=copy.deepcopy(small);bad['source_run']='results/nonexistent_F32_source';result['bad_source_exit'],_=cli('错误来源',bad);assert result['bad_source_exit']!=0
budget=copy.deepcopy(c);budget['budget_seconds']=1e-9;result['budget_exit'],paths=cli('预算截断',budget);assert result['budget_exit']==2
for path in paths:
    report=json.loads((path/'metrics/validation.json').read_text(encoding='utf-8'));assert report['status']=='incomplete' and 'selection' not in report and not (path/'metrics/selection.json').exists()
limit=copy.deepcopy(c);limit['reconstruction']['solver']['max_iterations']=1;result['iteration_limit_exit'],paths=cli('候选未收敛',limit);assert result['iteration_limit_exit']==2
for path in paths:
    report=json.loads((path/'metrics/validation.json').read_text(encoding='utf-8'));assert report['status']=='incomplete' and 'selection' not in report and not (path/'metrics/selection.json').exists()
# 只改变理想控制副本；正式来源完整保留。
dest=HERE/'理想测量篡改来源副本';shutil.copytree(ROOT/c['source_run'],dest)
p=next((dest/'arrays').glob('*.npz'))
with np.load(p,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
v['ideal_measurement_e'][0,0]+=1;np.savez_compressed(p,**v)
bad=copy.deepcopy(small);bad['source_run']=str(dest);result['tampered_ideal_source_exit'],_=cli('理想控制篡改拒绝',bad);assert result['tampered_ideal_source_exit']!=0
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
    if path.name=='validation.json':raise OSError('注入完成报告故障')
    return original(path,value)
with patch.object(m,'allocate_unique_run_dir',side_effect=allocate),patch.object(m,'write_json',side_effect=broken):
    try:m.run(small)
    except OSError:pass
    else:raise AssertionError('报告故障未传播')
assert (created[-1]/'failed.json').exists() and not (created[-1]/'metrics/validation.json').exists() and not plt.get_fignums();result['report_failure_no_completed_closed']=True
original_fit=m.fit_candidate
def certificate_failure(*args,**kwargs):
    fit=original_fit(*args,**kwargs);fit['certificate_passed']=False;return fit
with patch.object(m,'fit_candidate',side_effect=certificate_failure),patch.object(m,'allocate_unique_run_dir',side_effect=allocate):report=m.run(small)
assert report['status']=='incomplete' and not report['validation_passed'] and 'selection' not in report and not (created[-1]/'metrics/selection.json').exists();result['injected_certificate_failure_no_selection']=True
result.update(passed=True,real_gui_verified=False,retained_fault_runs=[str(p) for p in created])
with (HERE/'失败路径验收.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
