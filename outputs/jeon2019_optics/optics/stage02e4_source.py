# -*- coding: utf-8 -*-
"""为电子噪声实验检查02E-4的来源；不使用其重建值生成测量。"""
import hashlib,json
from pathlib import Path
import numpy as np
from .stage02e3_source import load_common_alpha_source,reuse_compatible
from .stage02_runtime import resolve_project_path
from .stage02c_source import tree_sha
from .imaging import SpectralImager
from .ridge_certificate import ridge_certificate

def load_noise_source(source,root):
    source,root=Path(source).resolve(),Path(root).resolve();before=tree_sha(source)
    r=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if r.get('status')!='completed' or r.get('stability_validation_passed') is not True or r.get('budget_exhausted') or (source/'failed.json').exists():raise ValueError('E4来源未完成稳定性验证')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02e4.py','optics/stage02e3_source.py'}<=set(manifest):raise ValueError('E4源码指纹缺失')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('E4源码不相容')
    c=r['config'];items,evidence=load_common_alpha_source(resolve_project_path(c['source_run'],root),root)
    if json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['sources']!=evidence['sources']:raise ValueError('E4递归来源改变')
    records={a['identity']:a for a in r['records']};expected={i['identity']+'_common_alpha' for i in items}
    if set(records)!=expected or len(records)!=len(r['records']) or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected or r['completed_jobs']!=r['planned_jobs'] or len(records)!=6:raise ValueError('E4身份集合错误')
    selected=[]
    for item in items:
        ident=item['identity']+'_common_alpha';rec=records[ident]
        with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        excluded={'alpha','alpha_factor','reconstruction','prediction','residual'}
        if set(v)!=set(item['values']) or any(not np.array_equal(v[k],a) for k,a in item['values'].items() if k not in excluded):raise ValueError('E4固定物理数据改变')
        alpha=evidence['common_alpha'];op=SpectralImager(v['kernels'],list(v['truth'].shape[1:]),float(v['pitch_m']),v['response'],v['crop_indices'].tolist())
        x=v['reconstruction'];pred=op.forward(x)
        if not np.all(np.isfinite(x)) or np.any(x<0) or not np.array_equal(pred,v['prediction']) or not np.array_equal(pred-v['measurement'],v['residual']):raise ValueError('E4拟合不一致')
        cert=ridge_certificate(op,v['measurement'],x,alpha)
        if float(v['alpha'])!=alpha or rec['alpha']!=alpha or float(v['alpha_factor'])!=rec['alpha_factor'] or cert!=rec['certificate'] or not rec['certificate_passed'] or cert['distance_upper_bound_relative']>c['certificate_relative_threshold']:raise ValueError('E4共同alpha/证书不一致')
        trajectory=json.loads((source/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))
        last=trajectory['history'][-1];gradient=op.adjoint(v['residual'])+alpha*x
        mapping=last['L']*(x-np.maximum(0,x-gradient/last['L']))
        pg=float(np.linalg.norm(mapping)/max(np.linalg.norm(op.adjoint(v['measurement'])),1e-300))
        objective=.5*float(np.sum(v['residual']**2))+.5*alpha*float(np.sum(x*x))
        if not (pg<=c['solver']['gradient_tolerance'] and last['objective_relative_change']<=c['solver']['objective_tolerance'] and np.isclose(pg,last['projected_gradient_relative'],rtol=1e-10,atol=1e-14) and np.isclose(objective,last['objective'],rtol=1e-13,atol=0)):raise ValueError('E4停止条件不一致')
        if trajectory['history'][-1]!=rec['final_optimization'] or len(trajectory['history'])!=rec['iterations'] or rec['solver_status']!='converged' or rec['scene']!=item['scene'] or rec['device']!=item['device']:raise ValueError('E4轨迹/身份不一致')
        if rec['execution']=='reused_verified' and (not reuse_compatible(item,c,alpha) or any(not np.array_equal(v[k],a) for k,a in item['values'].items()) or trajectory['history']!=item['history'] or rec['new_iterations']!=0):raise ValueError('E4缓存不相容')
        selected.append(dict(identity=ident,device=item['device'],scene=item['scene'],values=v))
    if tree_sha(source)!=before:raise RuntimeError('读取期间E4来源改变')
    return selected,dict(sources=[dict(path=str(source),sha256=before)]+evidence['sources'])
