# -*- coding: utf-8 -*-
import copy,json,os,sys,subprocess,tempfile,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
import main_stage02f1 as m
from optics.stage02c_source import tree_sha
c=json.loads((ROOT/'config_stage02f1.json').read_text(encoding='utf-8'))
result={}
def cli(name,config,extra=None):
    name='复核_'+name
    p=HERE/(name+'_config.json');p.write_text(json.dumps(config),encoding='utf-8')
    r=subprocess.run([sys.executable,'-B',str(ROOT/'main_stage02f1.py'),'--config',str(p)]+(extra or []),cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    (HERE/(name+'.log')).write_bytes(r.stdout);return r.returncode
bad=copy.deepcopy(c);bad['source_run']='results/nonexistent_02f1_source'
result['bad_source_exit']=cli('错误来源',bad);assert result['bad_source_exit']!=0
budget=copy.deepcopy(c);budget['budget_seconds']=1e-9
result['budget_exit']=cli('预算截断',budget);assert result['budget_exit']==2
small=copy.deepcopy(c);small['devices']=['continuous'];small['scenes']=['coincident_points'];small['detector']['gains_e_per_relative_power']=[1000.]
# 正式来源仍实际递归核验；仅缩小新实验任务数量。
before=tree_sha(ROOT)
result['alternate_cwd_no_save_exit']=cli('异地无保存',small,['--no-save']);assert result['alternate_cwd_no_save_exit']==0
assert tree_sha(ROOT)==before;result['no_save_tree_unchanged']=True;result['hashed_files']=len(before)
# 图形显示只是mock检查调用/关闭/失败路径，不作为真实GUI验证。
import matplotlib.pyplot as plt
with patch.object(plt,'show') as show:
    report=m.run(small,no_save=True,show_plots=True);assert report['validation_passed'];assert show.call_count==1
assert not plt.get_fignums();result['mock_show_once_closed']=True
with patch.object(plt,'show',side_effect=RuntimeError('注入显示失败')):
    try:m.run(small,no_save=True,show_plots=True)
    except RuntimeError:pass
    else:raise AssertionError('显示失败没有传播')
assert not plt.get_fignums();result['mock_show_failure_closed']=True
# 给报告写入注入故障，保留失败目录，验证完成标记不会提前写。
original=m.write_json
created=[]
original_alloc=m.allocate_unique_run_dir
def allocate(stage):
    path=original_alloc(stage);created.append(path);return path
def broken(path,value):
    if path.name=='validation.json':raise OSError('注入报告写入失败')
    return original(path,value)
with patch.object(m,'allocate_unique_run_dir',side_effect=allocate),patch.object(m,'write_json',side_effect=broken):
    try:m.run(small)
    except OSError:pass
    else:raise AssertionError('报告故障没有传播')
assert (created[0]/'failed.json').is_file() and not (created[0]/'metrics/validation.json').exists();assert not plt.get_fignums()
result['report_failure_no_completed_marker']=True;result['report_failure_run']=str(created[0]);result['real_gui_verified']=False;result['passed']=True
(HERE/'入口验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
