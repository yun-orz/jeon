# -*- coding: utf-8 -*-
"""阶段02D-3入口验收，证据保留在项目目录外，禁止删除旧文件。"""
from pathlib import Path
import sys,os,json,subprocess,hashlib,time
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics')
EV=ROOT.parent/'执行_阶段02D3_20261002'
PY=r'D:/dev/python/python3.10.4/python.exe'
env=dict(os.environ,PYTHONIOENCODING='utf-8')
def child(code,label,args=None):
    command=[PY,'-B',str(ROOT/'main_stage02d3.py')]+args if args is not None else [PY,'-B','-c',code]
    r=subprocess.run(command,cwd=str(EV),env=env,capture_output=True,text=True,encoding='utf-8')
    (EV/(label+'.log')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
    print(label,'exit',r.returncode,flush=True)
    return r
cfg=json.loads((ROOT/'config_stage02d3.json').read_text(encoding='utf-8'))
# 实际CLI缺失来源应留失败证据。
before=set((ROOT/'results/stage02d3').iterdir())
bad=dict(cfg,source_run='results/不存在的来源02D3')
(EV/'缺失来源配置.json').write_text(json.dumps(bad,ensure_ascii=False),encoding='utf-8')
r=child('', '缺失来源CLI', ['--config',str(EV/'缺失来源配置.json')])
new=set((ROOT/'results/stage02d3').iterdir())-before
assert r.returncode!=0 and len(new)==1
missing=next(iter(new))
assert (missing/'failed.json').is_file() and not (missing/'metrics/validation.json').exists()
# 真实未收敛CLI必须退出2，不能当作优化通过。
cfg['solver']['max_iterations']=1
(EV/'未收敛配置.json').write_text(json.dumps(cfg,ensure_ascii=False),encoding='utf-8')
r=child('', '未收敛CLI', ['--config',str(EV/'未收敛配置.json')])
assert r.returncode==2
summary=json.loads(r.stdout.strip().splitlines()[-1]); short=Path(summary['run_dir'])
v=json.loads((short/'metrics/validation.json').read_text(encoding='utf-8'))
assert v['status']=='completed' and not v['optimization_validation_passed'] and not v['all_reconstructions_converged']
# 在最后报告写入时模拟保存失败，不写完成标记。
code=f"""import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,{str(ROOT)!r})
import main_stage02d3 as m
c=json.loads((m.ROOT/'config_stage02d3.json').read_text(encoding='utf-8'));c['solver']['max_iterations']=1
original=Path.write_text
before=set((m.ROOT/'results/stage02d3').iterdir())
def fail(p,*a,**kw):
    if p.name=='report_stage02d3.md':raise OSError('验收注入报告保存失败')
    return original(p,*a,**kw)
with patch.object(Path,'write_text',fail):
    try:m.run(c)
    except OSError:pass
    else:raise AssertionError('保存故障未传播')
new=set((m.ROOT/'results/stage02d3').iterdir())-before
assert len(new)==1
p=next(iter(new));assert (p/'failed.json').is_file() and not (p/'metrics/validation.json').exists()
print(p)
"""
r=child(code,'报告保存故障');assert r.returncode==0
# 验证无保存运行不会改变项目内文件内容或集合。
def snapshot():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()}
a=snapshot();print('开始默认无保存运行',flush=True)
r=child('', '默认无保存CLI', ['--no-save']);assert r.returncode==0
s=json.loads(r.stdout.strip().splitlines()[-1]);assert s['optimization_validation_passed'] and s['run_dir'] is None
print('开始模拟手动显示分支',flush=True)
code=f"""import sys,json
from unittest.mock import patch
sys.path.insert(0,{str(ROOT)!r})
import main_stage02d3 as m
from optics.stage02_runtime import configure_plotting
plt,backend=configure_plotting(False)
c=json.loads((m.ROOT/'config_stage02d3.json').read_text(encoding='utf-8'))
with patch.object(m,'configure_plotting',return_value=(plt,backend)),patch.object(plt,'show') as show:
    report=m.run(c,no_save=True,show_plots=True)
    assert report['optimization_validation_passed']
    show.assert_called_once_with()
assert not plt.get_fignums()
"""
r=child(code,'模拟显示无保存');assert r.returncode==0
b=snapshot();assert a==b,'无保存模式改变了项目文件'
report=dict(cli_missing_source_failed=True,missing_source_run=str(missing),iteration_limit_exit=2,iteration_limit_run=str(short),report_write_failure_verified=True,no_save_exit=0,no_save_files_unchanged=True,files_checked=len(a),mock_show_called_once=True,figures_closed=True,real_GUI_verified=False)
(EV/'入口交付验收.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
