# -*- coding: utf-8 -*-
"""复审02A-R：只新增证据，反例不覆盖旧run或真实报告。"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import logging
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'outputs/jeon2019_optics'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))
sys.stdout.reconfigure(encoding='utf-8')
import main_stage02 as entry


def command(name,args):
    cmd=[sys.executable,'-B']+args
    t=time.perf_counter()
    done=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as f:f.write(done.stdout+'\n'+done.stderr)
    return {'command':cmd,'cwd':str(ROOT),'exit_code':done.returncode,'elapsed_s':time.perf_counter()-t}


def main():
    evidence={}
    # 生产文件及历史报告不允许在审核中改变。
    protected=list((PROJECT/'optics').glob('*.py'))+list((PROJECT/'tests').glob('*.py'))
    protected+=list((PROJECT/'docs').glob('*.md'))+[PROJECT/'main_stage02.py',PROJECT/'README.md']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    old=PROJECT/'results/stage02a/run_20261001_203729_02'
    new=PROJECT/'results/stage02a/run_20261001_213450_02'
    rows=[]
    for name in ['doe_continuous.npz','psf_420nm.npz','psf_540nm.npz','psf_660nm.npz','fresnel_control.npz']:
        a=np.load(old/'arrays'/name);b=np.load(new/'arrays'/name)
        keys=[k for k in ['delta_h_m','mask','u2_complex','intensity_raw','x_out_m','y_out_m'] if k in a and k in b]
        rows.append({'file':name,'shared_fields':keys,'all_shared_arrays_exactly_equal':all(np.array_equal(a[k],b[k]) for k in keys)})
    evidence['optical_arrays_vs_independently_reviewed_previous_run']=rows
    manifest=json.loads((new/'source_manifest.json').read_text(encoding='utf-8'))
    evidence['formal_run_manifest_matches_current_source']=all(hashlib.sha256(Path(a['absolute_path']).read_bytes()).hexdigest()==a['sha256'] for a in manifest['files'])
    evidence['tests']=command('完整70项测试',['-m','unittest','discover','-s',str(PROJECT/'tests'),'-t',str(PROJECT),'-v'])
    evidence['default_nosave']=command('项目外完整无保存',[str(PROJECT/'main_stage02.py'),'--no-save'])
    evidence['relative_config_cli']=command('相对配置反例',[str(PROJECT/'main_stage02.py'),'--config','config_stage02a.json','--only','height','--no-save'])
    cfg=entry.load_and_validate_config(PROJECT/'config_stage02a.json')
    # 上轮6个非法配置现应全部拒绝。
    bad=[]
    for p in sorted((ROOT/'outputs/审核_阶段02A_20261001').glob('反例配置_*.json')):
        try:entry.load_and_validate_config(p);rejected=False
        except Exception:rejected=True
        bad.append({'path':str(p),'rejected':rejected})
    evidence['previous_six_invalid_configs']=bad
    for name,field in [('zero',0j),('nan',complex(float('nan'),0))]:
        with patch.object(entry,'fresnel_kernel_separable',side_effect=lambda **k:np.full((len(k['y_out']),len(k['x_out'])),field,dtype=complex)):
            try:entry.run_stage02a(copy.deepcopy(cfg),None,only_mode='preview');rejected=False;error=''
            except Exception as exc:rejected=True;error=str(exc)
        evidence[name+'_field_rejected']={'rejected':rejected,'error':error}
    original=entry.fresnel_kernel_separable
    def scaled(**kwargs):return 2*original(**kwargs)
    with patch.object(entry,'fresnel_kernel_separable',side_effect=scaled):
        res=entry.run_stage02a(copy.deepcopy(cfg),None,only_mode='control')
    ctrl=res['self_check']['control_550nm']
    evidence['control_energy_excess_probe']={'status':res['completion_state']['status'],'control_self_check':ctrl['status'],'eta_window':ctrl['Pwindow']/ctrl['Pin']}
    calls=[]
    def track(**kwargs):calls.append(kwargs['wavelength']);return original(**kwargs)
    narrow=copy.deepcopy(cfg);narrow['optical']['preview_wavelengths_m']=[540e-9,540.05e-9,660e-9]
    with patch.object(entry,'fresnel_kernel_separable',side_effect=track):
        result=entry.run_stage02a(narrow,None,only_mode='preview')
    evidence['nearby_wavelength_probe']={'requested_m':narrow['optical']['preview_wavelengths_m'],'actually_propagated_m':calls,'status':result['completion_state']['status']}
    show_cfg=copy.deepcopy(cfg);show_cfg['runtime']['show_plots']=True
    show_path=OUT/'仅配置显示开启.json'
    with show_path.open('x',encoding='utf-8') as f:json.dump(show_cfg,f,ensure_ascii=False,indent=2)
    evidence['show_config_only']=command('配置显示反例',[str(PROJECT/'main_stage02.py'),'--config',str(show_path),'--only','height','--no-save'])
    # 检查非空目录是否被拒绝，用mock阻止所有计算和日志写入，保留标志文件。
    nonempty=OUT/'非空目录保护反例';nonempty.mkdir(exist_ok=False)
    (nonempty/'保留标志.txt').write_text('不可覆盖的既有目录标志',encoding='utf-8')
    args=argparse.Namespace(config=str(PROJECT/'config_stage02a.json'),only='height',no_save=False,show_plots=False,run_dir=str(nonempty))
    entered=[]
    def stopped(**k):entered.append(True);raise RuntimeError('审核mock中止，未写入计算结果')
    with patch.object(entry,'parse_args',return_value=args),patch.object(entry,'setup_logger',return_value=logging.getLogger('review_no_file')),patch.object(entry,'run_stage02a',side_effect=stopped):
        try:entry.main()
        except SystemExit:pass
    evidence['nonempty_run_dir_probe']={'calculation_entered_without_rejecting_nonempty_directory':bool(entered),'existing_marker_preserved':(nonempty/'保留标志.txt').read_text(encoding='utf-8')=='不可覆盖的既有目录标志','real_overwrite_attempted':False}
    evidence['production_files_and_docs_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in before.items())
    with (OUT/'复审独立数值.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,ensure_ascii=False,indent=2)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
