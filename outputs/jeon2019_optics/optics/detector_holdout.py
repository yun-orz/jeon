# -*- coding: utf-8 -*-
"""探测器采样/散射伴随与只使用含噪测量的留出选参。"""
import numpy as np
from .imaging import real_finite
from .electron_noise import random_stream
from .reconstruction import squared_norm,estimate_data_lipschitz
from .reconstruction_accelerated import solve_accelerated_ridge
from .ridge_certificate import ridge_certificate

class DetectorSubsetOperator:
    def __init__(self,base,indices):
        a=np.asarray(indices)
        if a.ndim!=1 or len(a)<1 or not np.issubdtype(a.dtype,np.integer) or len(np.unique(a))!=len(a) or np.any(a<0) or np.any(a>=np.prod(base.output_shape)):raise ValueError('像元索引须唯一、整数、非空且不越界')
        self.base=base;self.indices=a.astype(np.int64).copy();self.indices.setflags(write=False)
        self.cube_shape=base.cube_shape;self.output_shape=(1,len(a))
    def gather(self,detector):
        a=real_finite(detector,'探测器数组')
        if a.shape!=self.base.output_shape:raise ValueError('探测器shape错误')
        return a.ravel()[self.indices].reshape(self.output_shape)
    def forward(self,x):return self.gather(self.base.forward(x))
    def adjoint(self,y):
        a=real_finite(y,'采样测量')
        if a.shape!=self.output_shape:raise ValueError('采样测量shape错误')
        full=np.zeros(self.base.output_shape);full.ravel()[self.indices]=a.ravel()
        return self.base.adjoint(full)

def split_detector(shape,seed,identity,validation_fraction):
    if len(shape)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1 for v in shape):raise ValueError('探测器shape须两个正整数')
    if isinstance(validation_fraction,bool) or not np.isfinite(validation_fraction) or not 0<validation_fraction<1:raise ValueError('留出比例须在(0,1)')
    N=int(np.prod(shape));n=int(np.floor(validation_fraction*N))
    if n<1 or n>=N:raise ValueError('拆分须保留非空训练和验证集合')
    order=random_stream(seed,identity+'_'+str(list(shape)),'detector_split').permutation(N)
    return order[n:],order[:n]

def prepare_problem(base,measurement_e,background_e,split_config):
    z=real_finite(measurement_e,'含噪观测');b=real_finite(background_e,'背景')
    if z.shape!=base.output_shape or b.ndim not in [0,2] or (b.ndim==2 and b.shape!=z.shape) or np.any(b<0):raise ValueError('观测/背景shape或符号错误')
    ti,vi=split_detector(base.output_shape,split_config['seed'],split_config['identity'],split_config['validation_fraction'])
    train=DetectorSubsetOperator(base,ti);validation=DetectorSubsetOperator(base,vi)
    return dict(base=base,train=train,validation=validation,target_train=train.gather(z-b),target_validation=validation.gather(z-b),measurement=z.copy(),background=np.broadcast_to(b,z.shape).copy())

def fit_candidate(problem,factor,config,spectral_estimate):
    if isinstance(factor,bool) or not np.isfinite(factor) or factor<=0:raise ValueError('alpha因子须正有限')
    L=float(spectral_estimate['value'])
    if not np.isfinite(L) or L<=0:raise ValueError('训练L须正有限')
    alpha=factor*L;op=problem['train'];target=problem['target_train']
    result=solve_accelerated_ridge(op,target,alpha,config['solver'],config['initial_L_factor']*(L+alpha))
    x=result['reconstruction'];cert=ridge_certificate(op,target,x,alpha)
    prediction_train=op.forward(x);prediction_validation=problem['validation'].forward(x)
    residual_validation=prediction_validation-problem['target_validation']
    score=squared_norm(residual_validation)/residual_validation.size
    passed=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=config['certificate_relative_threshold']
    return dict(result=result,alpha=alpha,factor=factor,certificate=cert,certificate_passed=bool(passed),validation_mse_e2=float(score),prediction_train_signal_e=prediction_train,residual_train_e=prediction_train-target,prediction_validation_signal_e=prediction_validation,residual_validation_e=residual_validation)

def choose_candidate(records,factors):
    # 所有预声明候选完整通过，才允许选择；严格并列使用预声明顺序。
    if len(records)!=len(factors) or any(r['alpha_factor']!=f for r,f in zip(records,factors)):raise ValueError('候选身份/数量不完整')
    if any(r['solver_status']!='converged' or not r['certificate_passed'] or not np.isfinite(r['validation_mse_e2']) for r in records):raise ValueError('候选未全部收敛认证，拒绝选参')
    return int(np.argmin([r['validation_mse_e2'] for r in records]))

def refit_selected(problem,factor,config):
    estimate=estimate_data_lipschitz(problem['base'],config['power']);L=estimate['value'];alpha=factor*L
    result=solve_accelerated_ridge(problem['base'],problem['measurement']-problem['background'],alpha,config['solver'],config['initial_L_factor']*(L+alpha))
    cert=ridge_certificate(problem['base'],problem['measurement']-problem['background'],result['reconstruction'],alpha)
    return dict(result=result,alpha=alpha,spectral_estimate=estimate,certificate=cert)
