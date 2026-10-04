# -*- coding: utf-8 -*-
"""非负岭正则化投影梯度；求解器只接收算子和测量，不接收真值。"""
import time
import numpy as np
from .imaging import real_finite


def validate_solver(c):
    for name in ['alpha_relative','gradient_tolerance','objective_tolerance','power_tolerance','initial_L_factor']:
        if not np.isfinite(c[name]) or c[name]<=0: raise ValueError(name+'须有限且为正')
    for name in ['max_iterations','power_iterations','max_backtracks']:
        if isinstance(c[name],bool) or not isinstance(c[name],int) or c[name]<1: raise ValueError(name+'须为正整数')
    if not np.isfinite(c['backtracking_factor']) or c['backtracking_factor']<=1: raise ValueError('回溯倍数须大于1')
    if isinstance(c['seed'],bool) or not isinstance(c['seed'],int) or c['seed']<0: raise ValueError('种子须为非负整数')


def squared_norm(x):
    # 普通数组求和避免小数组反复启用多线程BLAS；保持双精度。
    return float(np.sum(np.asarray(x)**2))


def estimate_data_lipschitz(op,config):
    """幂迭代Rayleigh商是数值估计，不声称严格上界；回溯负责验步长。"""
    rng=np.random.default_rng(config['seed'])
    x=rng.standard_normal(op.cube_shape); x/=np.sqrt(squared_norm(x))
    history=[]; previous=None; converged=False
    for i in range(config['power_iterations']):
        y=op.forward(x); rayleigh=squared_norm(y)
        history.append(rayleigh)
        z=op.adjoint(y); norm=np.sqrt(squared_norm(z))
        if norm==0: return dict(value=0.,history=history,status='zero_operator')
        x=z/norm
        if previous is not None and abs(rayleigh-previous)<=config['power_tolerance']*max(rayleigh,1e-300):
            converged=True;break
        previous=rayleigh
    return dict(value=float(history[-1]),history=history,status='converged' if converged else 'iteration_limit')


def solve_nonnegative_ridge(op,measurement,config,spectral_estimate=None):
    validate_solver(config)
    y=real_finite(measurement,'重建测量')
    if y.shape!=op.output_shape: raise ValueError('重建测量shape与算子不匹配')
    start=time.perf_counter()
    estimate=estimate_data_lipschitz(op,config) if spectral_estimate is None else spectral_estimate
    data_L=float(estimate['value'])
    if not np.isfinite(data_L) or data_L<0: raise ValueError('谱范数估计非法')
    # 对零响应算子明确选用正的正则标度，唯一解为零。
    scale=data_L if data_L>0 else 1.
    alpha=config['alpha_relative']*scale
    L=config['initial_L_factor']*(data_L+alpha)
    x=np.zeros(op.cube_shape)
    residual=-y.copy(); data=.5*squared_norm(residual); objective=data
    aty=op.adjoint(y); denominator=max(np.sqrt(squared_norm(aty)),1e-300)
    ynorm=np.sqrt(squared_norm(y)); history=[]
    status='iteration_limit'
    for iteration in range(1,config['max_iterations']+1):
        gradient=op.adjoint(residual)+alpha*x
        for attempt in range(config['max_backtracks']):
            trial=np.maximum(0.,x-gradient/L)
            delta=trial-x
            trial_residual=op.forward(trial)-y
            trial_data=.5*squared_norm(trial_residual)
            trial_reg=.5*alpha*squared_norm(trial)
            trial_objective=trial_data+trial_reg
            bound=objective+float(np.sum(gradient*delta))+.5*L*squared_norm(delta)
            rounding=32*np.finfo(float).eps*max(abs(objective),abs(trial_objective),1e-300)
            if np.isfinite(trial_objective) and trial_objective<=bound+rounding:break
            L*=config['backtracking_factor']
        else: raise RuntimeError('回溯达到上限，拒绝返回伪收敛结果')
        if trial_objective>objective+rounding: raise RuntimeError('投影梯度目标异常上升')
        change=abs(objective-trial_objective)/max(abs(objective),1e-300)
        x,residual,data,objective=trial,trial_residual,trial_data,trial_objective
        new_gradient=op.adjoint(residual)+alpha*x
        mapping=L*(x-np.maximum(0.,x-new_gradient/L))
        pg_relative=np.sqrt(squared_norm(mapping))/denominator
        resnorm=np.sqrt(squared_norm(residual))
        history.append(dict(iteration=iteration,data_term=data,regularization_term=trial_reg,
            objective=objective,residual_norm=resnorm,residual_relative=resnorm/ynorm if ynorm>0 else None,
            L=L,backtracks=attempt,projected_gradient_relative=float(pg_relative),
            objective_relative_change=change,elapsed_s=time.perf_counter()-start))
        if pg_relative<=config['gradient_tolerance'] and change<=config['objective_tolerance']:
            status='converged';break
    return dict(reconstruction=x,status=status,iterations=len(history),alpha=alpha,L_data_estimate=data_L,
                spectral_estimate=estimate,history=history,elapsed_s=time.perf_counter()-start,
                projected_gradient_normalizer=denominator,initialization='zeros')


def evaluate_cube(truth,reconstruction,threshold):
    """仅在求解结束后评价真值；零光谱SAM不编造为零度。"""
    t,r=real_finite(truth,'真值'),real_finite(reconstruction,'恢复')
    if t.shape!=r.shape or t.ndim!=3 or np.any(t<0) or np.any(r<0): raise ValueError('评价须同形非负cube')
    if not np.isfinite(threshold) or threshold<=0: raise ValueError('光谱阈值须为正')
    error=r-t
    tn=np.sqrt(np.sum(t*t,axis=0));rn=np.sqrt(np.sum(r*r,axis=0))
    active=tn>threshold; valid=active&(rn>threshold)
    cos=np.sum(t*r,axis=0)[valid]/(tn[valid]*rn[valid])
    sam=np.degrees(np.arccos(np.clip(cos,-1.,1.)))
    bands=[]
    for b in range(len(t)):
        norm=np.sqrt(squared_norm(t[b])); other=(t[b]<=threshold)
        other_band_only=other&(np.sum(t,axis=0)>threshold)
        bands.append(dict(relative_l2=np.sqrt(squared_norm(error[b]))/norm if norm>0 else None,
            rmse=float(np.sqrt(np.mean(error[b]**2))),truth_power=float(t[b].sum()),
            recovered_power=float(r[b].sum()),power_on_truth_zero_pixels=float(r[b][other].sum()),
            power_at_other_band_only_truth_pixels=float(r[b][other_band_only].sum())))
    norm=np.sqrt(squared_norm(t))
    return dict(cube_relative_l2=np.sqrt(squared_norm(error))/norm if norm>0 else None,
        cube_rmse=float(np.sqrt(np.mean(error**2))),bands=bands,
        sam_mean_deg=float(sam.mean()) if len(sam) else None,sam_median_deg=float(np.median(sam)) if len(sam) else None,
        sam_valid_count=int(valid.sum()),sam_undefined_reconstruction_zero_count=int((active&~valid).sum()),
        sam_truth_zero_excluded_count=int((~active).sum()),spectrum_norm_threshold=threshold)
