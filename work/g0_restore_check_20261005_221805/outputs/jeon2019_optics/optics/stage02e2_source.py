# -*- coding: utf-8 -*-
"""双器件对照的只读来源复核；允许参考批次证书未全通过。"""
import hashlib,json
from pathlib import Path
import numpy as np
from .imaging import SpectralImager
from .ridge_certificate import ridge_certificate
from .stage02c_source import tree_sha
from .stage02d3_source import load_accelerated_source
from .stage02e1_source import load_sensitivity_source
from .stage02_runtime import resolve_project_path

def load_comparison_source(source,references,root):
    source,root=Path(source).resolve(),Path(root).resolve()
    continuous,evidence=load_accelerated_source(source,root)
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    records={r['identity']:r for r in report['records']}
    all_items=[]
    # 上面的冻结加载器已验证全部8份D3数组；这里不更改旧加载器的筛选范围。
    for device in ['continuous','nearest_depth']:
        for scene in ['coincident_points','separated_points','lines_and_square']:
            ident=device+'_'+scene+'_crop';rec=records[ident]
            with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
            baseline_alpha=float(values['alpha']);values['alpha']=np.asarray(baseline_alpha*.01);values['alpha_factor']=np.asarray(.01)
            all_items.append(dict(identity=ident+'_alpha_0p01',device=device,scene=scene,factor=.01,values=values,
                record=dict(rec,baseline_alpha=baseline_alpha),L_initial=rec['baseline_PG']['final_optimization']['L'],references=[]))
    sources={s['path']:s for s in evidence['sources']};cached={};reference_details=[]
    if len(references)!=2 or len(set(str(Path(p).resolve()) for p in references))!=2:raise ValueError('需要两份不同的E2参考批次')
    for reference in references:
        reference=Path(reference).resolve();before=tree_sha(reference)
        rr=json.loads((reference/'metrics/validation.json').read_text(encoding='utf-8'))
        if rr.get('status')!='completed' or rr.get('validation_passed') is not True or rr.get('budget_exhausted') or (reference/'failed.json').exists():raise ValueError('E2参考未通过实现交付验收')
        manifest=json.loads((reference/'source_manifest.json').read_text(encoding='utf-8'))
        if not {'main_stage02e2.py','optics/stage02e1_source.py','optics/ridge_certificate.py'}<=set(manifest):raise ValueError('E2参考源码清单缺失')
        for name,sha in manifest.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('E2参考源码不相容')
        parent=resolve_project_path(rr['config']['source_run'],root)
        if str(parent) not in cached:cached[str(parent)]=load_sensitivity_source(parent,root)
        parents,pe=cached[str(parent)]
        old=json.loads((reference/'source_evidence.json').read_text(encoding='utf-8'))
        if old['sources']!=pe['sources']:raise ValueError('E2参考递归来源改变')
        for s in pe['sources']:
            if s['path'] in sources and sources[s['path']]['sha256']!=s['sha256']:raise ValueError('参考来源链矛盾')
            sources[s['path']]=s
        rows={r['identity']:r for r in rr['records']}
        expected={'continuous_'+scene+'_crop_alpha_0p01_refined' for scene in rr['config']['scenes']}
        if rr['completed_jobs']!=rr['planned_jobs'] or len(rows)!=len(rr['records']) or set(rows)!=expected or {p.stem for p in (reference/'arrays').glob('*.npz')}!=expected:raise ValueError('E2参考身份集合错误')
        for ident,rec in rows.items():
            parent_item=next(p for p in parents if p['scene']==rec['scene'] and p['factor']==.01)
            with np.load(reference/'arrays'/(ident+'.npz'),allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
            if any(k not in values or not np.array_equal(values[k],v) for k,v in parent_item['values'].items() if k not in ['reconstruction','prediction','residual']):raise ValueError('E2参考固定数据/实际α改变')
            op=SpectralImager(values['kernels'],list(values['truth'].shape[1:]),float(values['pitch_m']),values['response'],values['crop_indices'].tolist())
            x=values['reconstruction'];alpha=float(values['alpha']);prediction=op.forward(x);residual=prediction-values['measurement']
            if np.any(x<0) or not np.all(np.isfinite(x)) or not np.array_equal(values['prediction'],prediction) or not np.array_equal(values['residual'],residual) or alpha!=rec['alpha']:raise ValueError('E2参考恢复/拟合错误')
            trajectory=json.loads((reference/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'));h=trajectory['history'];last=h[-1]
            solver=rr['config']['solver'];gradient=op.adjoint(residual)+alpha*x;L=last['L']
            pg=float(np.linalg.norm((L*(x-np.maximum(0,x-gradient/L))).ravel())/max(np.linalg.norm(op.adjoint(values['measurement']).ravel()),1e-300))
            objective=.5*float(np.sum(residual**2))+.5*alpha*float(np.sum(x*x))
            conv=pg<=solver['gradient_tolerance'] and last['objective_relative_change']<=solver['objective_tolerance']
            cert=ridge_certificate(op,values['measurement'],x,alpha);passed=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=rr['config']['certificate_relative_threshold']
            if not (len(h)==rec['iterations'] and last==rec['final_optimization'] and trajectory['status']==rec['solver_status']==('converged' if conv else 'iteration_limit')
                    and (conv or len(h)==solver['max_iterations']) and trajectory['alpha']==alpha and rec['solver_config']==solver
                    and np.isclose(objective,last['objective'],rtol=1e-13,atol=0) and np.isclose(pg,last['projected_gradient_relative'],rtol=1e-10,atol=1e-14)
                    and cert==rec['certificate'] and rec['certificate_passed']==passed):raise ValueError('E2参考状态/证书/轨迹不一致')
            item=next(i for i in all_items if i['device']=='continuous' and i['scene']==rec['scene'])
            item['references'].append(dict(run=str(reference),record=rec,values=values))
        allconv=all(r['solver_status']=='converged' for r in rows.values());allcert=all(r['certificate_passed'] for r in rows.values())
        if not (rr['all_reconstructions_converged']==rr['optimization_validation_passed']==allconv and rr['certificate_validation_passed']==allcert and rr['stability_validation_passed']==(allconv and allcert)):raise ValueError('E2参考总标志错误')
        if tree_sha(reference)!=before:raise RuntimeError('读取期间E2参考改变')
        sources[str(reference)]=dict(path=str(reference),sha256=before)
        reference_details.append(dict(run=str(reference),optimization_validation_passed=allconv,certificate_validation_passed=allcert,stability_validation_passed=allconv and allcert))
    for scene in ['coincident_points','separated_points','lines_and_square']:
        pair=[i for i in all_items if i['scene']==scene]
        for key in ['truth','wavelengths_m','pitch_m','crop_indices','scene_x_m','scene_y_m','output_x_m','output_y_m']:
            if not np.array_equal(pair[0]['values'][key],pair[1]['values'][key]):raise ValueError('器件对照的场景/坐标不一致')
    for s in sources.values():
        if tree_sha(Path(s['path']))!=s['sha256']:raise RuntimeError('载入期间对照来源改变')
    return all_items,dict(sources=list(sources.values()),source_config=report['config'],reference_details=reference_details,
        comparison_strategy='各器件02D-3实际α×0.01，实际α不同；相同相对正则化尺度，非纯量化因素对照')
