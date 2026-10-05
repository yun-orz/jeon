# -*- coding: utf-8 -*-
"""核验冻结控制与分解，随后把理想信息隔离在参数选择之外。"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from .stage02c_source import tree_sha
from .stage02_runtime import resolve_project_path
from .stage02f2_source import load_fixed_control_source
from .electron_reconstruction import make_electron_operator,WeightedOperator
from .fixed_target_control import decompose_errors
from .reconstruction import squared_norm,evaluate_cube
from .ridge_certificate import ridge_certificate

def load_holdout_source(source,root):
    source,root=Path(source).resolve(),Path(root).resolve();before=tree_sha(source)
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if report.get('status')!='completed' or report.get('validation_passed') is not True or report.get('budget_exhausted') or (source/'failed.json').exists():raise ValueError('F3.1来源未完整通过')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02f3_1.py','optics/fixed_target_control.py','optics/stage02f2_source.py'}<=set(manifest):raise ValueError('F3.1源码指纹缺失')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('F3.1冻结源码不相容')
    c=report['config'];parents,evidence=load_fixed_control_source(resolve_project_path(c['source_run'],root),root)
    old_evidence=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    if old_evidence!=evidence:raise ValueError('F3.1递归来源改变')
    pc=evidence['source_config'];records={r['identity']:r for r in report['records']};expected={i['identity']+'_ideal_control' for i in parents}
    if len(expected)!=12 or set(records)!=expected or len(records)!=len(report['records']) or report['planned_jobs']!=report['completed_jobs'] or report['completed_jobs']!=12 or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected:raise ValueError('需要正式12控制来源')
    items=[]
    for parent in parents:
        ident=parent['identity']+'_ideal_control';r=records[ident];old=parent['values']
        with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
        for key in ['evaluation_truth','kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','quantum_efficiency','reference_wavelength_m','gain_e_per_relative_power','read_sigma_e','background_e','output_x_m','output_y_m','scene_x_m','scene_y_m','seed','weights','variance_estimate_e2','alpha','L_data_estimate','effective_electron_response','method','measurement_unit']:
            if not np.array_equal(v[key],old[key]):raise ValueError('F3.1冻结参数或单位改变')
        for key,oldkey in [('noisy_measurement_e','measurement_e'),('noisy_reconstruction','reconstruction'),('noisy_prediction_e','prediction_e'),('noisy_residual_e','residual_e')]:
            if not np.array_equal(v[key],old[oldkey]):raise ValueError('F3.1旧观测/恢复改变')
        if r['source_identity']!=parent['identity'] or str(v['source_identity'])!=parent['identity'] or r['device']!=parent['device'] or r['scene']!=parent['scene'] or r['method']!=parent['method'] or r['alpha']!=float(v['alpha']) or r['L_data_estimate']!=float(v['L_data_estimate']):raise ValueError('F3.1身份改变')
        base=make_electron_operator(v['kernels'],list(v['evaluation_truth'].shape[1:]),float(v['pitch_m']),v['response'],v['wavelengths_m'],v['quantum_efficiency'],float(v['reference_wavelength_m']),float(v['gain_e_per_relative_power']),v['crop_indices'].tolist())
        ideal=base.forward(v['evaluation_truth'])+float(v['background_e']);x=v['ideal_reconstruction'];op=WeightedOperator(base,v['weights']);target=v['weights']*(ideal-float(v['background_e']));alpha=float(v['alpha'])
        if not np.array_equal(v['ideal_measurement_e'],ideal) or not np.array_equal(v['ideal_weighted_target'],target) or np.any(x<0) or not np.all(np.isfinite(x)) or not np.array_equal(v['ideal_prediction_e'],base.forward(x)+float(v['background_e'])) or not np.array_equal(v['ideal_residual_e'],v['ideal_prediction_e']-ideal):raise ValueError('F3.1理想目标/预测错误')
        cert=ridge_certificate(op,target,x,alpha);noisy_cert=parent['record']['certificate']
        trajectory=json.loads((source/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'));h=trajectory['history'];last=h[-1];L=last['L']
        residual=op.forward(x)-target;gradient=op.adjoint(residual)+alpha*x;pg=np.sqrt(squared_norm(L*(x-np.maximum(0,x-gradient/L))))/max(np.sqrt(squared_norm(op.adjoint(target))),1e-300);objective=.5*squared_norm(residual)+.5*alpha*squared_norm(x)
        if not (cert==r['ideal_certificate'] and noisy_cert==r['noisy_certificate'] and r['certificate_passed'] and cert['distance_upper_bound_relative']<=pc['certificate_relative_threshold'] and trajectory['alpha']==alpha and trajectory['status']==r['solver_status']=='converged' and len(h)==r['iterations'] and last==r['final_optimization'] and r['solver_config']==pc['reconstruction']['solver'] and r['initialization']=='zeros' and np.isclose(pg,last['projected_gradient_relative'],rtol=1e-10,atol=1e-14) and np.isclose(objective,last['objective'],rtol=1e-12,atol=0) and pg<=r['solver_config']['gradient_tolerance'] and last['objective_relative_change']<=r['solver_config']['objective_tolerance']):raise ValueError('F3.1控制优化/证书错误')
        if not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:])):raise ValueError('F3.1目标非单调')
        errors,d=decompose_errors(v['evaluation_truth'],v['noisy_reconstruction'],x,noisy_cert['distance_upper_bound'],cert['distance_upper_bound'])
        if any(not np.array_equal(v[k],a) for k,a in errors.items()) or d!=r['decomposition'] or d!=json.loads((source/'metrics'/(ident+'_decomposition.json')).read_text(encoding='utf-8')):raise ValueError('F3.1分解/区间改变')
        if r['noisy_execution']!='reused_verified' or r['noisy_new_iterations']!=0 or r['noisy_elapsed_this_run_s']!=0 or r['noisy_evaluation']!=parent['record']['evaluation'] or evaluate_cube(v['evaluation_truth'],x,pc['evaluation']['spectrum_norm_threshold'])!=r['ideal_evaluation']:raise ValueError('F3.1复用/评价错误')
        with (source/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if len(rows)!=len(h):raise ValueError('F3.1 CSV长度错误')
        for row,step in zip(rows,h):
            for k,value in step.items():
                good=row[k]==str(value) if isinstance(value,bool) else row[k]=='' if value is None else float(row[k])==value
                if not good:raise ValueError('F3.1 CSV轨迹改变')
        items.append(dict(identity=ident,device=parent['device'],scene=parent['scene'],method=parent['method'],values=v,parent=parent))
    if not all(report.get(k) is True for k in ['numerical_validation_passed','all_controls_converged','certificate_validation_passed']) or report['noisy_verified_reused_count']!=12 or report['noisy_new_iterations']!=0:raise ValueError('F3.1总验收标记错误')
    with (source/'metrics/decomposition.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=12 or {v['identity'] for v in rows}!=expected:raise ValueError('F3.1分解CSV身份错误')
    for row in rows:
        r=records[row['identity']];d=r['decomposition']
        if float(row['alpha'])!=r['alpha'] or row['status']!='converged':raise ValueError('F3.1分解CSV状态错误')
        for col,k in [('total_relative','total'),('baseline_relative','baseline'),('noise_perturbation_relative','noise_perturbation')]:
            if float(row[col])!=d['norms_relative'][k]:raise ValueError('F3.1分解CSV数值错误')
        if float(row['cross_relative_squared'])!=d['cross_term_relative_squared']:raise ValueError('F3.1交叉项错误')
        for prefix,key in [('baseline','baseline_interval'),('noise','noise_perturbation_interval')]:
            for bound in ['lower','upper']:
                if float(row[prefix+'_'+bound])!=d[key][bound+'_relative']:raise ValueError('F3.1区间CSV错误')
    if tree_sha(source)!=before:raise RuntimeError('读取期间F3.1来源改变')
    return items,dict(sources=[dict(path=str(source),sha256=before)]+evidence['sources'],evaluation_threshold=pc['evaluation']['spectrum_norm_threshold'])
