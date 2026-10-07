# -*- coding: utf-8 -*-
"""只读复核02D-3收敛来源及递归光学/成像/PG证据。"""
import hashlib,json
from pathlib import Path
import numpy as np
from .imaging import SpectralImager
from .stage02c_source import tree_sha
from .stage02d2_source import load_pg_source
from .stage02_runtime import resolve_project_path

def load_accelerated_source(source,root):
    source,root=Path(source).resolve(),Path(root).resolve()
    before=tree_sha(source)
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if not all(report.get(k) is True for k in ['validation_passed','optimization_validation_passed','all_reconstructions_converged']) or report.get('status')!='completed':
        raise ValueError('加速源未通过收敛和交付验收')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02d3.py','optics/reconstruction_accelerated.py','optics/stage02d2_source.py'}<=set(manifest):
        raise ValueError('加速源源码清单不完整')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError('加速源源码不相容：'+name)
    pg=resolve_project_path(report['config']['source_run'],root)
    items,evidence=load_pg_source(pg,root)
    old=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    for key in ['sha256','imaging_sha256','optical_sha256']:
        if old[key]!=evidence[key]:raise ValueError('加速源的递归数据来源改变')
    records={r['identity']:r for r in report['records']}
    expected={item['device']+'_'+item['scene']+'_'+mode for item in items for mode in item['baselines']}
    if len(records)!=len(report['records']) or set(records)!=expected or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected:
        raise ValueError('加速源身份集合错误')
    selected=[]
    for item in items:
        v=item['values']
        for mode,b in item['baselines'].items():
            ident=item['device']+'_'+item['scene']+'_'+mode;r=records[ident]
            op=SpectralImager(v['kernels'],list(v['cube'].shape[1:]),float(v['pitch_m']),v['response'],v['crop_indices'].tolist() if mode=='crop' else None)
            with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                values={k:z[k].copy() for k in z.files}
            fixed=dict(truth=v['cube'],measurement=v['measurement_'+mode],kernels=v['kernels'],response=v['response'],
                wavelengths_m=v['wavelengths_m'],pitch_m=v['pitch_m'],fixed_height_fingerprint=v['fixed_height_fingerprint'],
                crop_indices=np.asarray(op.crop),alpha=np.asarray(b['alpha']),**op.coordinates())
            if any(k not in values or not np.array_equal(values[k],value) for k,value in fixed.items()):
                raise ValueError('加速源物理数据/α改变：'+ident)
            x=values['reconstruction'];prediction=op.forward(x);residual=prediction-values['measurement']
            if np.any(x<0) or not np.all(np.isfinite(x)) or not np.array_equal(values['prediction'],prediction) or not np.array_equal(values['residual'],residual):
                raise ValueError('加速源前向一致性错误')
            trajectory=json.loads((source/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))
            last=trajectory['history'][-1]
            objective=.5*float(np.sum(residual**2))+.5*b['alpha']*float(np.sum(x*x))
            if not (r['solver_status']==trajectory['status']=='converged' and r['alpha']==trajectory['alpha']==b['alpha']
                    and last==r['final_optimization'] and len(trajectory['history'])==r['iterations']
                    and np.isclose(objective,last['objective'],rtol=1e-13,atol=0)
                    and last['projected_gradient_relative']<=report['config']['solver']['gradient_tolerance']
                    and last['objective_relative_change']<=report['config']['solver']['objective_tolerance']):
                raise ValueError('加速源目标/轨迹/收敛状态错误')
            if item['device']=='continuous' and mode=='crop':
                selected.append(dict(identity=ident,scene=item['scene'],values=values,record=r,L_initial=b['L_initial']))
    if tree_sha(source)!=before:raise RuntimeError('读取期间加速来源改变')
    sources=[dict(path=str(source),sha256=before)]
    sources += [dict(path=evidence[k],sha256=evidence[h]) for k,h in [('source_run','sha256'),('imaging_source_run','imaging_sha256'),('optical_source_run','optical_sha256')]]
    return selected,dict(sources=sources,source_config=report['config'],source_manifest=manifest,scope=report['scope'])
