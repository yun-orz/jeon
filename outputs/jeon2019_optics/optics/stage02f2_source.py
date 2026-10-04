# -*- coding: utf-8 -*-
"""重算02F-2目标、权重、停止条件与证书后才复用含噪恢复。"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from .stage02c_source import tree_sha
from .stage02_runtime import resolve_project_path
from .stage02f1_source import load_electron_source
from .electron_reconstruction import make_electron_operator,WeightedOperator,measured_weights
from .reconstruction import estimate_data_lipschitz,evaluate_cube,squared_norm
from .ridge_certificate import ridge_certificate

def load_fixed_control_source(source,root):
    source,root=Path(source).resolve(),Path(root).resolve();before=tree_sha(source)
    r=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if r.get('status')!='completed' or r.get('validation_passed') is not True or r.get('budget_exhausted') or (source/'failed.json').exists():raise ValueError('F2来源未完整通过')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02f2.py','optics/electron_reconstruction.py','optics/stage02f1_source.py'}<=set(manifest):raise ValueError('F2源码指纹缺失')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('F2冻结源码不相容')
    c=r['config'];items,evidence=load_electron_source(resolve_project_path(c['source_run'],root),root)
    if json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['sources']!=evidence['sources']:raise ValueError('F2递归来源改变')
    records={a['identity']:a for a in r['records']};parents={a['identity']:a for a in items if float(a['values']['gain_e_per_relative_power'])==c['gain_e_per_relative_power'] and a['device'] in c['devices'] and a['scene'] in c['scenes']}
    expected={i+'_'+method for i in parents for method in c['methods']}
    if len(expected)!=12 or set(records)!=expected or len(records)!=len(r['records']) or r['completed_jobs']!=r['planned_jobs'] or r['completed_jobs']!=12 or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected:raise ValueError('需要F2正式完整12项')
    selected=[]
    for ident,rec in records.items():
        item=parents[rec['source_identity']];old=item['values']
        with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        for key in ['kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','quantum_efficiency','reference_wavelength_m','gain_e_per_relative_power','read_sigma_e','background_e','output_x_m','output_y_m','scene_x_m','scene_y_m','seed']:
            if key not in v or not np.array_equal(v[key],old[key]):raise ValueError('F2固定器件/单位参数改变')
        if not np.array_equal(v['measurement_e'],old['noisy_measurement_e']) or not np.array_equal(v['evaluation_truth'],old['truth']) or str(v['measurement_unit'])!='electron' or str(v['method'])!=rec['method'] or rec['device']!=item['device'] or rec['scene']!=item['scene']:raise ValueError('F2观测/评价真值/身份改变')
        base=make_electron_operator(v['kernels'],list(v['evaluation_truth'].shape[1:]),float(v['pitch_m']),v['response'],v['wavelengths_m'],v['quantum_efficiency'],float(v['reference_wavelength_m']),float(v['gain_e_per_relative_power']),v['crop_indices'].tolist())
        if rec['method']=='unweighted':weights=np.ones(base.output_shape);variance=np.asarray([])
        elif rec['method']=='fixed_weighted':weights,variance=measured_weights(v['measurement_e'],float(v['read_sigma_e']),c['reconstruction']['variance_floor_e2'])
        else:raise ValueError('未知F2方法')
        if not np.array_equal(weights,v['weights']) or not np.array_equal(variance,v['variance_estimate_e2']) or not np.array_equal(base.response,v['effective_electron_response']):raise ValueError('F2固定权重/响应改变')
        op=WeightedOperator(base,weights);target=weights*(v['measurement_e']-float(v['background_e']));x=v['reconstruction'];alpha=float(v['alpha'])
        estimate=estimate_data_lipschitz(op,c['reconstruction']['power'])
        if estimate!=rec['spectral_estimate'] or float(v['L_data_estimate'])!=estimate['value'] or alpha!=rec['alpha'] or alpha!=c['reconstruction']['alpha_factor']*estimate['value'] or rec['alpha_factor']!=c['reconstruction']['alpha_factor']:raise ValueError('F2谱范数/alpha规则改变')
        prediction=base.forward(x)+float(v['background_e']);residual=prediction-v['measurement_e']
        if np.any(x<0) or not np.all(np.isfinite(x)) or not np.array_equal(target,v['weighted_target']) or not np.array_equal(prediction,v['prediction_e']) or not np.array_equal(residual,v['residual_e']):raise ValueError('F2预测/非负恢复错误')
        trajectory=json.loads((source/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'));h=trajectory['history'];last=h[-1]
        gradient=op.adjoint(op.forward(x)-target)+alpha*x;L=last['L'];mapping=L*(x-np.maximum(0,x-gradient/L))
        pg=np.sqrt(squared_norm(mapping))/max(np.sqrt(squared_norm(op.adjoint(target))),1e-300)
        objective=.5*squared_norm(op.forward(x)-target)+.5*alpha*squared_norm(x)
        cert=ridge_certificate(op,target,x,alpha)
        passed=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=c['certificate_relative_threshold']
        change=abs(h[-2]['objective']-objective)/max(abs(h[-2]['objective']),1e-300) if len(h)>1 else abs(.5*squared_norm(target)-objective)/max(.5*squared_norm(target),1e-300)
        if not (trajectory['alpha']==alpha and trajectory['status']==rec['solver_status']=='converged' and len(h)==rec['iterations'] and last==rec['final_optimization'] and rec['initialization']=='zeros'
            and np.isclose(objective,last['objective'],rtol=1e-12,atol=0) and np.isclose(pg,last['projected_gradient_relative'],rtol=1e-10,atol=1e-14)
            and np.isclose(change,last['objective_relative_change'],rtol=1e-6,atol=1e-14) and pg<=c['reconstruction']['solver']['gradient_tolerance'] and last['objective_relative_change']<=c['reconstruction']['solver']['objective_tolerance']
            and cert==rec['certificate'] and passed and rec['certificate_passed'] and rec['numerical_validation_passed']):raise ValueError('F2目标/停止/证书不相容')
        if not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:])):raise ValueError('F2轨迹非单调')
        if evaluate_cube(v['evaluation_truth'],x,c['evaluation']['spectrum_norm_threshold'])!=rec['evaluation']:raise ValueError('F2真值评价不相容')
        with (source/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=len(h):raise ValueError('F2 CSV轨迹长度错误')
        for row,step in zip(rows,h):
            if set(row)!=set(step):raise ValueError('F2 CSV字段错误')
            for key,value in step.items():
                if isinstance(value,bool):good=row[key]==str(value)
                elif value is None:good=row[key]==''
                else:good=float(row[key])==value
                if not good:raise ValueError('F2 CSV轨迹数值错误')
        selected.append(dict(identity=ident,device=rec['device'],scene=rec['scene'],method=rec['method'],values=v,record=rec,history=h,mean_signal_raw_e=old['mean_signal_raw_e'],source_run=str(source)))
    if not all(r.get(key) is True for key in ['numerical_validation_passed','all_reconstructions_converged','certificate_validation_passed']):raise ValueError('F2总体标记不相容')
    with (source/'metrics/evaluation.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=12 or {a['identity'] for a in rows}!=expected:raise ValueError('F2评价CSV身份错误')
    for row in rows:
        rec=records[row['identity']]
        if float(row['alpha'])!=rec['alpha'] or float(row['cube_error'])!=rec['evaluation']['cube_relative_l2'] or row['status']!='converged' or row['certificate_passed']!='True':raise ValueError('F2评价CSV错误')
    if tree_sha(source)!=before:raise RuntimeError('读取期间F2来源改变')
    return selected,dict(sources=[dict(path=str(source),sha256=before)]+evidence['sources'],source_config=c)
