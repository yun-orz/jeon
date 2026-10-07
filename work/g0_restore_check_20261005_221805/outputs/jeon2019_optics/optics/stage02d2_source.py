# -*- coding: utf-8 -*-
"""只读载入未收敛PG基线，冻结每个实验的实际α与测量。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .imaging import SpectralImager
from .stage02d1_source import load_imaging_source
from .stage02c_source import tree_sha
from .stage02_runtime import resolve_project_path


def load_pg_source(source,root):
    source,root=Path(source),Path(root)
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if report.get('status')!='completed' or report.get('validation_passed') is not True:
        raise ValueError('PG源run未通过实现/交付检查')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02d2.py','optics/reconstruction.py','optics/stage02d1_source.py'}<=set(manifest):
        raise ValueError('PG源代码指纹缺失')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError('PG源码不兼容：'+name)
    before=tree_sha(source)
    imaging=resolve_project_path(report['config']['source_run'],root)
    items,evidence=load_imaging_source(imaging,root)
    old=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    if old['sha256']!=evidence['sha256'] or old['optical_sha256']!=evidence['optical_sha256']:
        raise ValueError('PG的成像/光学来源已改变')
    records={r['identity']:r for r in report['records']}
    if len(records)!=len(report['records']):raise ValueError('PG实验身份重复')
    expected=set()
    for item in items:
        v=item['values'];device=item['device'];label=item['scene'];baselines={}
        modes=['crop']+(['full'] if label==report['config']['full_diagnostic_scene'] else [])
        for mode in modes:
            ident=device+'_'+label+'_'+mode;expected.add(ident)
            if ident not in records:raise ValueError('PG实验身份缺失')
            rec=records[ident]
            crop=v['crop_indices'].tolist() if mode=='crop' else None
            op=SpectralImager(v['kernels'],list(v['cube'].shape[1:]),float(v['pitch_m']),v['response'],crop)
            with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                alpha=float(z['alpha'])
                if not (np.isfinite(alpha) and alpha>0 and alpha==rec['alpha']
                        and np.array_equal(z['truth'],v['cube']) and np.array_equal(z['kernels'],v['kernels'])
                        and np.array_equal(z['measurement'],v['measurement_'+mode])
                        and np.array_equal(z['response'],v['response']) and np.array_equal(z['wavelengths_m'],v['wavelengths_m'])
                        and float(z['pitch_m'])==float(v['pitch_m']) and np.array_equal(z['crop_indices'],op.crop)
                        and str(z['fixed_height_fingerprint'])==str(v['fixed_height_fingerprint'])
                        and all(np.array_equal(z[k],value) for k,value in op.coordinates().items())):
                    raise ValueError('PG核/测量/实际α/坐标不一致：'+ident)
                x=z['reconstruction'];prediction=op.forward(x)
                if np.any(x<0) or not np.array_equal(z['prediction'],prediction) or not np.array_equal(z['residual'],prediction-z['measurement']):
                    raise ValueError('PG恢复及前向拟合不一致')
                trajectory=json.loads((source/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))
                last=trajectory['history'][-1]
                objective=.5*float(np.sum(z['residual']**2))+.5*alpha*float(np.sum(x*x))
                if not (np.isclose(objective,last['objective'],rtol=1e-13,atol=0)
                        and trajectory['status']==rec['solver_status'] and trajectory['alpha']==alpha
                        and last==rec['final_optimization'] and len(trajectory['history'])==rec['iterations']):
                    raise ValueError('PG轨迹/目标/状态不一致')
                baselines[mode]=dict(record=rec,alpha=alpha,L_initial=float(last['L']))
        item['baselines']=baselines
    if set(records)!=expected or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected:
        raise ValueError('PG实验身份集合不匹配')
    if tree_sha(source)!=before:raise RuntimeError('载入期间PG源run被修改')
    return items,dict(source_run=str(source.resolve()),sha256=before,config=report['config'],
        imaging_source_run=str(imaging),imaging_sha256=evidence['sha256'],
        optical_source_run=evidence['optical_source_run'],optical_sha256=evidence['optical_sha256'],
        source_unchanged=True,baseline_all_converged=report['all_reconstructions_converged'])
