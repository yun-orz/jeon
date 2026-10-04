# -*- coding: utf-8 -*-
"""02E-2入口、保存/显示故障、预算和无保存审核，保留证据。"""
from pathlib import Path
import os,sys,json,subprocess,hashlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
PY=sys.executable;env=dict(os.environ,PYTHONIOENCODING='utf-8')
def child(label,args=None,code=None):
    cmd=[PY,'-B',str(ROOT/'main_stage02e3.py')]+args if args is not None else [PY,'-B','-c',code]
    r=subprocess.run(cmd,cwd=EV,env=env,capture_output=True,text=True,encoding='utf-8')
    (EV/(label+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');print(label,'exit',r.returncode,flush=True);return r
original=json.loads((ROOT/'config_stage02e3.json').read_text(encoding='utf-8'))
def small():
    c=json.loads(json.dumps(original));c['solver']['max_iterations']=1;return c
def config_path(c,label):
    p=EV/(label+'.json');p.write_text(json.dumps(c,ensure_ascii=False),encoding='utf-8');return str(p)
base=ROOT/'results/stage02e3'
before=set(base.iterdir());c=small();c['source_run']='results/不存在的02E3来源'
r=child('缺失来源CLI',args=['--config',config_path(c,'缺失来源配置')]);assert r.returncode!=0
new=set(base.iterdir())-before;assert len(new)==1
p=next(iter(new));assert (p/'failed.json').is_file() and not (p/'metrics/validation.json').exists()
r=child('未收敛CLI',args=['--config',config_path(small(),'未收敛配置')]);assert r.returncode==2
summary=json.loads(r.stdout.strip().splitlines()[-1]);v=json.loads((Path(summary['run_dir'])/'metrics/validation.json').read_text(encoding='utf-8'))
assert v['completed_jobs']==6 and not v['stability_validation_passed'] and v['validation_passed']
c=small();c['budget_seconds']=.001
r=child('预算截断CLI',args=['--config',config_path(c,'预算截断配置')]);assert r.returncode==2
summary=json.loads(r.stdout.strip().splitlines()[-1]);v=json.loads((Path(summary['run_dir'])/'metrics/validation.json').read_text(encoding='utf-8'))
assert v['completed_jobs']==1 and v['planned_jobs']==6 and v['budget_exhausted']
# 保存/显示失败均不能留下最终完成标记。
for fault in ['报告保存故障','显示故障']:
    code=f"""import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,{str(ROOT)!r})
import main_stage02e3 as m
from optics.stage02_runtime import configure_plotting
c=json.loads({json.dumps(small(),ensure_ascii=False)!r})
original=Path.write_text;before=set((m.ROOT/'results/stage02e3').iterdir())
plt,backend=configure_plotting(False)
def fail(p,*a,**kw):
    if p.name=='report_stage02e3.md':raise OSError('验收注入报告失败')
    return original(p,*a,**kw)
if {fault!r}=='报告保存故障':
    with patch.object(Path,'write_text',fail):
        try:m.run(c)
        except OSError:pass
        else:raise AssertionError('写入故障未传播')
else:
    with patch.object(m,'configure_plotting',return_value=(plt,backend)),patch.object(plt,'show',side_effect=RuntimeError('验收注入显示失败')):
        try:m.run(c,show_plots=True)
        except RuntimeError:pass
        else:raise AssertionError('显示故障未传播')
new=set((m.ROOT/'results/stage02e3').iterdir())-before;assert len(new)==1
p=next(iter(new));assert (p/'failed.json').is_file() and not (p/'metrics/validation.json').exists()
assert not plt.get_fignums()
"""
    r=child(fault,code=code);assert r.returncode==0
def snapshot():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()}
a=snapshot();print('默认无保存严格复核开始',flush=True)
r=child('默认无保存CLI',args=['--no-save']);assert r.returncode in [0,2]
no_save_exit=r.returncode;summary=json.loads(r.stdout.strip().splitlines()[-1]);assert summary['run_dir'] is None and summary['status']=='completed'
no_save_stability=summary['stability_validation_passed']
code=f"""import sys,json
from unittest.mock import patch
sys.path.insert(0,{str(ROOT)!r})
import main_stage02e3 as m
from optics.stage02_runtime import configure_plotting
plt,backend=configure_plotting(False)
c=json.loads({json.dumps(small(),ensure_ascii=False)!r})
with patch.object(m,'configure_plotting',return_value=(plt,backend)),patch.object(plt,'show') as show:
    r=m.run(c,no_save=True,show_plots=True)
    assert r['completed_jobs']==6 and not r['optimization_validation_passed']
    show.assert_called_once_with()
assert not plt.get_fignums()
"""
r=child('模拟显示无保存',code=code);assert r.returncode==0
b=snapshot();assert a==b
report=dict(missing_source_failed=True,iteration_limit_exit=2,budget_partial_exit=2,report_write_failure=True,show_failure=True,
    default_no_save_exit=no_save_exit,default_no_save_stability_passed=no_save_stability,no_save_files_unchanged=True,
    files_checked=len(a),mock_show_called_once=True,figures_closed=True,real_PyCharm_and_GUI_verified=False)
(EV/'入口交付验收.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False),flush=True)
