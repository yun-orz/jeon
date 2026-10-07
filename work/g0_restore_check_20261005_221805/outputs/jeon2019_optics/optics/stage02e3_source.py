# -*- coding: utf-8 -*-
"""固定共同α的来源检查和明确缓存复用条件，不修改旧来源模块。"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from .imaging import SpectralImager
from .reconstruction import evaluate_cube
from .ridge_certificate import ridge_certificate
from .stage02c_source import tree_sha
from .stage02e2_source import load_comparison_source
from .stage02_runtime import resolve_project_path

def reuse_compatible(item,config,alpha):
    r=item['record'];relative=r['certificate']['distance_upper_bound_relative']
    return bool(config['reuse_continuous'] and item['device']=='continuous' and float(item['values']['alpha'])==alpha
        and r['solver_config']==config['solver'] and r['solver_status']=='converged' and r['initialization']=='zeros'
        and relative is not None and relative<=config['certificate_relative_threshold'])

def load_common_alpha_source(source,root):
    source,root=Path(source).resolve(),Path(root).resolve();before=tree_sha(source)
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if report.get('status')!='completed' or report.get('validation_passed') is not True or report.get('budget_exhausted') or (source/'failed.json').exists():raise ValueError('E3源未通过实现交付或预算截断')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02e3.py','optics/stage02e2_source.py','optics/ridge_certificate.py'}<=set(manifest):raise ValueError('E3源码指纹缺失')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('E3源码不相容')
    c=report['config'];items,evidence=load_comparison_source(resolve_project_path(c['source_run'],root),[resolve_project_path(p,root) for p in c['reference_runs']],root)
    old=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    if old['sources']!=evidence['sources']:raise ValueError('E3递归数据来源改变')
    records={r['identity']:r for r in report['records']}
    expected={item['identity']+'_comparison' for item in items if item['device'] in c['devices'] and item['scene'] in c['scenes']}
    if len(records)!=len(report['records']) or set(records)!=expected or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected or report['completed_jobs']!=report['planned_jobs']:raise ValueError('E3身份集合错误')
    selected=[]
    for item in items:
        ident=item['identity']+'_comparison'
        if ident not in records:continue
        r=records[ident]
        with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        if any(k not in v or not np.array_equal(v[k],value) for k,value in item['values'].items() if k not in ['reconstruction','prediction','residual']):raise ValueError('E3固定物理数据/α改变')
        if r['alpha']!=float(v['alpha']) or r['device']!=item['device'] or r['scene']!=item['scene'] or r['baseline_alpha']!=item['record']['baseline_alpha']:raise ValueError('E3器件/实际α身份错误')
        op=SpectralImager(v['kernels'],list(v['truth'].shape[1:]),float(v['pitch_m']),v['response'],v['crop_indices'].tolist())
        x=v['reconstruction'];prediction=op.forward(x);residual=prediction-v['measurement'];alpha=float(v['alpha'])
        if np.any(x<0) or not np.all(np.isfinite(x)) or not np.array_equal(v['prediction'],prediction) or not np.array_equal(v['residual'],residual):raise ValueError('E3恢复/拟合错误')
        trajectory=json.loads((source/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'));h=trajectory['history'];last=h[-1]
        g=op.adjoint(residual)+alpha*x;L=last['L'];mapping=L*(x-np.maximum(0,x-g/L))
        pg=float(np.linalg.norm(mapping.ravel())/max(np.linalg.norm(op.adjoint(v['measurement']).ravel()),1e-300))
        objective=.5*float(np.sum(residual**2))+.5*alpha*float(np.sum(x*x))
        conv=pg<=c['solver']['gradient_tolerance'] and last['objective_relative_change']<=c['solver']['objective_tolerance']
        cert=ridge_certificate(op,v['measurement'],x,alpha);passed=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=c['certificate_relative_threshold']
        if not (trajectory['alpha']==alpha and len(h)==r['iterations'] and last==r['final_optimization'] and r['solver_config']==c['solver']
                and trajectory['status']==r['solver_status']==('converged' if conv else 'iteration_limit') and (conv or len(h)==c['solver']['max_iterations'])
                and np.isclose(objective,last['objective'],rtol=1e-13,atol=0) and np.isclose(pg,last['projected_gradient_relative'],rtol=1e-10,atol=1e-14)
                and cert==r['certificate'] and passed==r['certificate_passed'] and r['initialization']=='zeros'):raise ValueError('E3目标/状态/轨迹/证书错误')
        if not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:])):raise ValueError('E3轨迹非单调')
        if evaluate_cube(v['truth'],x,c['evaluation']['spectrum_norm_threshold'])!=r['evaluation']:raise ValueError('E3恢复评价不一致')
        with (source/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:
            if len(list(csv.DictReader(f)))!=len(h):raise ValueError('E3 CSV轮数错误')
        selected.append(dict(identity=ident,device=item['device'],scene=item['scene'],values=v,record=r,L_initial=item['L_initial'],history=h,source_run=str(source)))
    allconv=all(r['solver_status']=='converged' for r in records.values());allcert=all(r['certificate_passed'] for r in records.values())
    if not (report['all_reconstructions_converged']==report['optimization_validation_passed']==allconv and report['certificate_validation_passed']==allcert and report['stability_validation_passed']==(allconv and allcert)):raise ValueError('E3总验收标志错误')
    cont=[item for item in selected if item['device']=='continuous']
    if len(cont)!=3 or {i['scene'] for i in cont}!={'coincident_points','separated_points','lines_and_square'}:raise ValueError('需要三组连续器件参考')
    common_alpha=float(cont[0]['record']['baseline_alpha'])*.01
    if any(float(i['values']['alpha'])!=common_alpha or float(i['record']['baseline_alpha'])*.01!=common_alpha for i in cont):raise ValueError('连续参考共同α不一致')
    if tree_sha(source)!=before:raise RuntimeError('读取期间E3来源改变')
    return selected,dict(sources=[dict(path=str(source),sha256=before)]+evidence['sources'],source_config=c,common_alpha=common_alpha,
        comparison_strategy='共同实际α固定为continuous D3基线×0.01；continuous条件相同可复用，nearest_depth重新计算')
