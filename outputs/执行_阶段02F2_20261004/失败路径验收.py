# -*- coding: utf-8 -*-
import copy,json,os,sys,subprocess,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
import main_stage02f2 as m
c=json.loads((ROOT/'config_stage02f2.json').read_text(encoding='utf-8'));small=copy.deepcopy(c);small['devices']=['continuous'];small['scenes']=['coincident_points'];small['methods']=['unweighted']
result={}
def cli(name,config):
    p=HERE/(name+'_config.json');p.write_text(json.dumps(config),encoding='utf-8')
    r=subprocess.run([sys.executable,'-B',str(ROOT/'main_stage02f2.py'),'--config',str(p)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    (HERE/(name+'.log')).write_bytes(r.stdout);return r.returncode
bad=copy.deepcopy(small);bad['source_run']='results/nonexistent_F2_source';result['bad_source_exit']=cli('错误来源',bad);assert result['bad_source_exit']!=0
budget=copy.deepcopy(small);budget['budget_seconds']=1e-9;result['budget_exit']=cli('预算截断',budget);assert result['budget_exit']==2
limit=copy.deepcopy(small);limit['reconstruction']['solver']['max_iterations']=1;result['iteration_limit_exit']=cli('未收敛',limit);assert result['iteration_limit_exit']==2
# 只修改专用来源副本的原始计数；完整保留副本，不修改正式数据。
copy_path=HERE/'篡改来源副本';shutil.copytree(ROOT/c['source_run'],copy_path)
p=next((copy_path/'arrays').glob('*.npz'))
with np.load(p,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
values['shot_counts'][0,0]+=1;np.savez_compressed(p,**values)
tamper=copy.deepcopy(small);tamper['source_run']=str(copy_path);result['tampered_counts_exit']=cli('篡改计数拒绝',tamper);assert result['tampered_counts_exit']!=0
import matplotlib.pyplot as plt
with patch.object(plt,'show') as show:
    report=m.run(small,no_save=True,show_plots=True);assert report['validation_passed'];assert show.call_count==1
assert not plt.get_fignums();result['mock_show_once_closed']=True
# 显示故障测试使用保存模式，以实际核对没有完成标记。
original_allocate=m.allocate_unique_run_dir;created=[]
def allocate(stage):
    p=original_allocate(stage);created.append(p);return p
with patch.object(m,'allocate_unique_run_dir',side_effect=allocate),patch.object(plt,'show',side_effect=RuntimeError('注入显示失败')):
    try:m.run(small,show_plots=True)
    except RuntimeError:pass
    else:raise AssertionError('显示失败没有传播')
assert not plt.get_fignums() and (created[-1]/'failed.json').exists() and not (created[-1]/'metrics/validation.json').exists();result['show_failure_no_completed_closed']=True
original=m.write_json
def broken(path,value):
    if path.name=='validation.json':raise OSError('注入报告失败')
    return original(path,value)
with patch.object(m,'allocate_unique_run_dir',side_effect=allocate),patch.object(m,'write_json',side_effect=broken):
    try:m.run(small)
    except OSError:pass
    else:raise AssertionError('报告失败没有传播')
assert not plt.get_fignums() and (created[-1]/'failed.json').exists() and not (created[-1]/'metrics/validation.json').exists();result['report_failure_no_completed_closed']=True
result.update(passed=True,real_gui_verified=False,failed_run_paths=[str(p) for p in created])
(HERE/'失败路径验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
