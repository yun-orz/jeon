# -*- coding: utf-8 -*-
"""冻结权重/alpha的理想观测控制与单次实现的确定性误差分解。"""
import numpy as np
from .imaging import real_finite
from .electron_reconstruction import WeightedOperator
from .reconstruction import squared_norm
from .reconstruction_accelerated import solve_accelerated_ridge
from .ridge_certificate import ridge_certificate

def check_frozen_parameters(weights,alpha,data_L,reference):
    w=real_finite(weights,'控制权重')
    if not np.array_equal(w,reference['weights']) or float(alpha)!=float(reference['alpha']) or float(data_L)!=float(reference['L_data_estimate']):raise ValueError('冻结W/alpha/L改变，拒绝理想控制')
    for value in [alpha,data_L]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('冻结alpha/L须正有限')

def solve_fixed_target(base,measurement_e,background_e,weights,alpha,data_L,initial_L_factor,solver_config,reference):
    check_frozen_parameters(weights,alpha,data_L,reference)
    z=real_finite(measurement_e,'理想电子观测');b=real_finite(background_e,'背景')
    if z.shape!=base.output_shape or b.ndim not in [0,2] or (b.ndim==2 and b.shape!=z.shape) or np.any(b<0):raise ValueError('控制观测或背景不合法')
    if isinstance(initial_L_factor,bool) or not np.isfinite(initial_L_factor) or initial_L_factor<=0:raise ValueError('初始L因子须正有限')
    op=WeightedOperator(base,weights);target=op.weights*(z-b)
    # 不估计新权重、谱范数或alpha；真值不进入求解接口。
    result=solve_accelerated_ridge(op,target,alpha,solver_config,initial_L_factor*(data_L+alpha))
    return dict(result=result,operator=op,target=target,certificate=ridge_certificate(op,target,result['reconstruction'],alpha))

def decompose_errors(truth,noisy,ideal,delta_noisy,delta_ideal):
    t=real_finite(truth,'评价真值');xn=real_finite(noisy,'含噪恢复');x0=real_finite(ideal,'理想控制恢复')
    if t.shape!=xn.shape or t.shape!=x0.shape or t.ndim!=3 or np.any(t<0) or np.any(xn<0) or np.any(x0<0):raise ValueError('分解须同形非负cube')
    for d in [delta_noisy,delta_ideal]:
        if isinstance(d,bool) or not np.isfinite(d) or d<0:raise ValueError('证书绝对距离须非负有限')
    tn=np.sqrt(squared_norm(t))
    if tn<=0:raise ValueError('本批分解要求非零真值')
    total=xn-t;baseline=x0-t;noise=xn-x0
    norms={k:np.sqrt(squared_norm(a)) for k,a in [('total',total),('baseline',baseline),('noise_perturbation',noise)]}
    cross=2*float(np.sum(baseline*noise));closure=squared_norm(total)-squared_norm(baseline)-squared_norm(noise)-cross
    def interval(n,d):return dict(lower=max(0.,n-d),upper=n+d,lower_relative=max(0.,n-d)/tn,upper_relative=(n+d)/tn,uncertainty_absolute=d)
    metrics=dict(truth_norm=tn,norms_absolute=norms,norms_relative={k:v/tn for k,v in norms.items()},cross_term=cross,cross_term_relative_squared=cross/(tn*tn),
        vector_identity_relative=np.sqrt(squared_norm(total-baseline-noise))/max(norms['total'],1e-300),squared_identity_relative=abs(closure)/max(squared_norm(total),1e-300),
        baseline_interval=interval(norms['baseline'],delta_ideal),noise_perturbation_interval=interval(norms['noise_perturbation'],delta_noisy+delta_ideal),
        interpretation='固定观测W与alpha下的单次确定性分解；基线包括编码/约束/正则化及含噪权重影响，不是纯岭偏差；区间是优化误差界而非统计置信。')
    return dict(error_total=total,error_baseline=baseline,error_noise_perturbation=noise),metrics
