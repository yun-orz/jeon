# -*- coding: utf-8 -*-
"""检查真实CLI状态、保存故障、预算截断和无保存/模拟显示。"""
from pathlib import Path
import sys,os,json,subprocess,hashlib
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
PY=sys.executable;env=dict(os.environ,PYTHONIOENCODING='utf-8')
def child(label,args=None,code=None):
    command=[PY,'-B',str(ROOT/'main_stage02e1.py')]+args if args is not None else [PY,'-B','-c',code]
    r=subprocess.run(command,cwd=EV,env=env,capture_output=True,text=True,encoding='utf-8')
    (EV/(label+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');print(label,'exit',r.returncode,flush=True);return r
def cfg_file(c,label):
    p=EV/(label+'.json');p.write_text(json.dumps(c,ensure_ascii=False),encoding='utf-8');return str(p)
def small():
    c=json.loads((ROOT/'config_stage02e1.json').read_text(encoding='utf-8'))
    c['alpha_factors']=[1.];c['stricter']['enabled']=False;c['stricter']['alpha_factors']=[1.]
    return c
before=set((ROOT/'results/stage02e1').iterdir());c=small();c['source_run']='results/不存在的02E1来源'
r=child('缺失来源CLI',args=['--config',cfg_file(c,'缺失来源配置')]);assert r.returncode!=0
new=set((ROOT/'results/stage02e1').iterdir())-before;assert len(new)==1
p=next(iter(new));assert (p/'failed.json').is_file() and not (p/'metrics/validation.json').exists()
c=small();c['solver']['max_iterations']=1
r=child('未收敛CLI',args=['--config',cfg_file(c,'未收敛配置')]);assert r.returncode==2
s=json.loads(r.stdout.strip().splitlines()[-1]);v=json.loads((Path(s['run_dir'])/'metrics/validation.json').read_text(encoding='utf-8'))
assert v['status']=='completed' and v['completed_jobs']==3 and not v['all_reconstructions_converged']
# 预算在实验之间截断，已完成数据保留，不能当作完整扫描。
c=small();c['budget_seconds']=.001
r=child('预算截断CLI',args=['--config',cfg_file(c,'预算截断配置')]);assert r.returncode==2
s=json.loads(r.stdout.strip().splitlines()[-1]);v=json.loads((Path(s['run_dir'])/'metrics/validation.json').read_text(encoding='utf-8'))
assert v['budget_exhausted'] and v['completed_jobs']==1 and v['planned_jobs']==3
c=small();c['solver']['max_iterations']=1
code=f"""import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,{str(ROOT)!r})
import main_stage02e1 as m
c=json.loads({json.dumps(c,ensure_ascii=False)!r})
original=Path.write_text;before=set((m.ROOT/'results/stage02e1').iterdir())
def fail(p,*a,**kw):
    if p.name=='report_stage02e1.md':raise OSError('验收注入报告写入故障')
    return original(p,*a,**kw)
with patch.object(Path,'write_text',fail):
    try:m.run(c)
    except OSError:pass
    else:raise AssertionError('报告失败未传播')
new=set((m.ROOT/'results/stage02e1').iterdir())-before;assert len(new)==1
p=next(iter(new));assert (p/'failed.json').is_file() and not (p/'metrics/validation.json').exists()
"""
r=child('报告保存故障',code=code);assert r.returncode==0
# 无保存阶段开始后不修改项目文件。
def snapshot():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()}
a=snapshot();print('默认无保存14组开始',flush=True)
r=child('默认无保存CLI',args=['--no-save']);assert r.returncode in [0,2]
default_exit=r.returncode
s=json.loads(r.stdout.strip().splitlines()[-1]);assert s['status']=='completed' and s['run_dir'] is None
default_optimization=s['optimization_validation_passed']
c=small()
code=f"""import sys,json
from unittest.mock import patch
sys.path.insert(0,{str(ROOT)!r})
import main_stage02e1 as m
from optics.stage02_runtime import configure_plotting
plt,backend=configure_plotting(False)
c=json.loads({json.dumps(c,ensure_ascii=False)!r})
with patch.object(m,'configure_plotting',return_value=(plt,backend)),patch.object(plt,'show') as show:
    r=m.run(c,no_save=True,show_plots=True)
    assert r['optimization_validation_passed'] and r['completed_jobs']==3
    show.assert_called_once_with()
assert not plt.get_fignums()
"""
r=child('模拟显示无保存',code=code);assert r.returncode==0
b=snapshot();assert a==b
report=dict(missing_source_failure=True,iteration_limit_exit=2,budget_partial_exit=2,report_write_failure=True,
    default_no_save_exit=default_exit,default_no_save_optimization_passed=default_optimization,
    no_save_files_unchanged=True,files_checked=len(a),mock_show_called_once=True,all_figures_closed=True,real_GUI_verified=False)
(EV/'入口交付验收.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False),flush=True)
