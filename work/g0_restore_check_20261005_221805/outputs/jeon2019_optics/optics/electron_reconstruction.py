# -*- coding: utf-8 -*-
"""电子数与固定对角权重算子；求解接口不接收真值。"""
import numpy as np
from .imaging import SpectralImager,real_finite
from .electron_noise import electron_response
from .reconstruction import estimate_data_lipschitz
from .reconstruction_accelerated import solve_accelerated_ridge
from .ridge_certificate import ridge_certificate

class WeightedOperator:
    def __init__(self,base,weights):
        self.base=base;self.weights=real_finite(weights,'固定权重').copy()
        if self.weights.shape!=base.output_shape or np.any(self.weights<=0):raise ValueError('权重须与探测器同形且为正')
        self.weights.setflags(write=False);self.cube_shape=base.cube_shape;self.output_shape=base.output_shape
    def forward(self,x):return self.weights*self.base.forward(x)
    def adjoint(self,y):
        y=real_finite(y,'加权测量')
        if y.shape!=self.output_shape:raise ValueError('加权测量shape错误')
        return self.base.adjoint(self.weights*y)

def make_electron_operator(kernels,scene_shape,pitch_m,response,wavelengths_m,qe,reference_wavelength_m,gain,crop):
    if isinstance(gain,bool) or not np.isfinite(gain) or gain<=0:raise ValueError('电子增益须正有限')
    factors,_=electron_response(wavelengths_m,qe,reference_wavelength_m)
    return SpectralImager(kernels,scene_shape,pitch_m,np.asarray(response)*gain*factors,crop)

def measured_weights(measurement_e,read_sigma_e,variance_floor_e2):
    z=real_finite(measurement_e,'含噪电子观测')
    if isinstance(read_sigma_e,bool) or not np.isfinite(read_sigma_e) or read_sigma_e<0:raise ValueError('读出标准差非法')
    if isinstance(variance_floor_e2,bool) or not np.isfinite(variance_floor_e2) or variance_floor_e2<=0:raise ValueError('方差下限须正有限')
    # 只截断方差估计的计数部分；目标观测的负读出值必须保留。
    variance=np.maximum(np.maximum(z,0.)+read_sigma_e**2,variance_floor_e2)
    return 1/np.sqrt(variance),variance

def reconstruct_electrons(base,measurement_e,background_e,read_sigma_e,method,config):
    z=real_finite(measurement_e,'电子观测');b=real_finite(background_e,'背景')
    if z.shape!=base.output_shape or b.ndim not in [0,2] or (b.ndim==2 and b.shape!=z.shape) or np.any(b<0):raise ValueError('观测/背景shape或符号错误')
    if method=='unweighted':weights=np.ones(z.shape);variance=None
    elif method=='fixed_weighted':weights,variance=measured_weights(z,read_sigma_e,config['variance_floor_e2'])
    else:raise ValueError('未知重建方法')
    op=WeightedOperator(base,weights);target=weights*(z-b)
    estimate=estimate_data_lipschitz(op,config['power']);data_L=estimate['value']
    if not np.isfinite(data_L) or data_L<=0:raise ValueError('电子算子谱范数估计必须为正')
    alpha=config['alpha_factor']*data_L
    result=solve_accelerated_ridge(op,target,alpha,config['solver'],config['initial_L_factor']*(data_L+alpha))
    certificate=ridge_certificate(op,target,result['reconstruction'],alpha)
    return dict(result=result,operator=op,target=target,weights=weights,variance_estimate_e2=variance,spectral_estimate=estimate,alpha=alpha,certificate=certificate)
