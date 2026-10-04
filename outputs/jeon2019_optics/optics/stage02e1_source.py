# -*- coding: utf-8 -*-
"""只读载入含未收敛诊断的02E-1；逐组验证，不伪造总优化标志。"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from .imaging import SpectralImager
from .ridge_certificate import ridge_certificate
from .stage02c_source import tree_sha
from .stage02d3_source import load_accelerated_source
from .stage02_runtime import resolve_project_path

def load_sensitivity_source(source,root):
    source,root=Path(source).resolve(),Path(root).resolve()
    before=tree_sha(source)
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if report.get('status')!='completed' or report.get('validation_passed') is not True or (source/'failed.json').exists():
        raise ValueError('参数扫描源未通过实现验收或含失败标记')
    if report.get('budget_exhausted') or report['completed_jobs']!=report['planned_jobs']:raise ValueError('参数扫描源不完整')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02e1.py','optics/ridge_certificate.py','optics/stage02d3_source.py'}<=set(manifest):raise ValueError('参数扫描源源码指纹缺失')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('参数扫描源源码不相容：'+name)
    parent=resolve_project_path(report['config']['source_run'],root)
    items,evidence=load_accelerated_source(parent,root)
    old=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    if old['sources']!=evidence['sources']:raise ValueError('扫描源递归数据改变')
    c=report['config'];jobs=[(item,factor,False) for item in items for factor in c['alpha_factors']]
    if c['stricter']['enabled']:jobs += [(item,factor,True) for item in items if item['scene']==c['stricter']['scene'] for factor in c['stricter']['alpha_factors']]
    records={r['identity']:r for r in report['records']}
    expected={item['identity']+'_alpha_'+format(factor,'.8g').replace('.','p')+('_strict' if strict else '') for item,factor,strict in jobs}
    if len(records)!=len(report['records']) or set(records)!=expected or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected:raise ValueError('扫描源实验身份集合错误')
    selected=[]
    for item,factor,strict in jobs:
        ident=item['identity']+'_alpha_'+format(factor,'.8g').replace('.','p')+('_strict' if strict else '')
        rec=records[ident];base=item['values']
        with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        if any(k not in values or not np.array_equal(values[k],value) for k,value in base.items() if k not in ['alpha','reconstruction','prediction','residual']):raise ValueError('扫描源固定物理数据改变')
        alpha=float(base['alpha'])*factor
        if not (float(values['alpha'])==alpha==rec['alpha'] and float(values['alpha_factor'])==factor==rec['alpha_factor'] and rec['baseline_alpha']==float(base['alpha']) and rec['stricter']==strict and rec['scene']==item['scene']):raise ValueError('扫描源α或身份改变')
        op=SpectralImager(values['kernels'],list(values['truth'].shape[1:]),float(values['pitch_m']),values['response'],values['crop_indices'].tolist())
        x=values['reconstruction'];prediction=op.forward(x);residual=prediction-values['measurement']
        if np.any(x<0) or not np.all(np.isfinite(x)) or not np.array_equal(values['prediction'],prediction) or not np.array_equal(values['residual'],residual):raise ValueError('扫描源恢复/拟合错误')
        trajectory=json.loads((source/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'));h=trajectory['history'];last=h[-1]
        solver=dict(c['solver'])
        if strict:
            for key in ['gradient_tolerance','objective_tolerance']:solver[key]=c['stricter'][key]
        gradient=op.adjoint(residual)+alpha*x;L=last['L'];mapping=L*(x-np.maximum(0,x-gradient/L))
        pg=float(np.linalg.norm(mapping.ravel())/max(np.linalg.norm(op.adjoint(values['measurement']).ravel()),1e-300))
        objective=.5*float(np.sum(residual**2))+.5*alpha*float(np.sum(x*x))
        converged=pg<=solver['gradient_tolerance'] and last['objective_relative_change']<=solver['objective_tolerance']
        if not (len(h)==rec['iterations'] and last==rec['final_optimization'] and trajectory['alpha']==alpha and trajectory['status']==rec['solver_status'] and rec['solver_config']==solver
                and np.isclose(objective,last['objective'],rtol=1e-13,atol=0) and np.isclose(pg,last['projected_gradient_relative'],rtol=1e-10,atol=1e-14)
                and rec['solver_status']==('converged' if converged else 'iteration_limit') and (converged or len(h)==solver['max_iterations'])):raise ValueError('扫描源状态/目标/轨迹不一致')
        if not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:])):raise ValueError('扫描源轨迹非单调')
        cert=ridge_certificate(op,values['measurement'],x,alpha)
        if cert!=rec['certificate']:raise ValueError('扫描源证书不一致')
        with (source/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:
            if len(list(csv.DictReader(f)))!=len(h):raise ValueError('扫描源CSV轮数错误')
        if not strict: selected.append(dict(identity=ident,scene=item['scene'],factor=float(factor),values=values,record=rec,L_initial=item['L_initial']))
    all_converged=all(r['solver_status']=='converged' for r in records.values())
    if report['all_reconstructions_converged']!=all_converged or report['optimization_validation_passed']!=all_converged:raise ValueError('扫描源总收敛标志错误')
    if tree_sha(source)!=before:raise RuntimeError('读取期间参数扫描来源改变')
    return selected,dict(sources=[dict(path=str(source),sha256=before)]+evidence['sources'],source_config=c,
        baseline_all_converged=all_converged,baseline_optimization_validation_passed=report['optimization_validation_passed'])
