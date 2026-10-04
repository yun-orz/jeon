# -*- coding: utf-8 -*-
"""由强凸距离界推导器件cube误差差的区间；只用于评价，不用于迭代。"""
import numpy as np

def error_difference_interval(error_cont,error_near,distance_cont,distance_near,truth_norm):
    values=[error_cont,error_near,distance_cont,distance_near,truth_norm]
    if any(isinstance(v,bool) or not np.isfinite(v) for v in values) or min(values[:4])<0 or truth_norm<=0:raise ValueError('误差与距离界须非负有限、真值范数须正有限')
    delta=float(error_near-error_cont);radius=float((distance_cont+distance_near)/truth_norm)
    return dict(nearest_minus_continuous_cube_error=delta,optimization_uncertainty_radius=radius,
        lower=delta-radius,upper=delta+radius,ordering_supported_by_theoretical_bounds=bool(delta-radius>0 or delta+radius<0),
        caveat='固定共同α、当前理想离散模型的数值理论界；未作区间舍入包络，不等于真实相机性能')
