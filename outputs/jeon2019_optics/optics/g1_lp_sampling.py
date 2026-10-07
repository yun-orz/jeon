# -*- coding: utf-8 -*-
"""LightPipes核单元平均解释；固定采样DOE，不更改器件高度。"""
import numpy as np
from scipy.special import fresnel


def cell_matrix(output_axis,input_axis,spacing,wavelength,distance):
    """对每个方向在位移±spacing积分，再除2；对应宽2dx的核平均权重。"""
    if any(not np.isfinite(v) or v<=0 for v in [spacing,wavelength,distance]):
        raise ValueError('步长、波长与传播距离须有限正值')
    out=np.asarray(output_axis,dtype=float);inp=np.asarray(input_axis,dtype=float)
    if out.ndim!=1 or inp.ndim!=1 or not np.isfinite(out).all() or not np.isfinite(inp).all():
        raise ValueError('坐标须有限一维数组')
    shift=out[:,None]-inp[None,:];factor=np.sqrt(2/(wavelength*distance))
    sa,ca=fresnel(factor*(shift-spacing));sb,cb=fresnel(factor*(shift+spacing))
    return np.sqrt(wavelength*distance/2)*((cb-ca)+1j*(sb-sa))/2


def cell_averaged_field(u1,input_axis,output_axis,spacing,wavelength,distance):
    """核积分平均与点求和使用同一输入复场；全局相位另由调用方处理。"""
    if np.shape(u1)!=(len(input_axis),len(input_axis)):
        raise ValueError('输入复场尺寸与坐标不匹配')
    matrix=cell_matrix(output_axis,input_axis,spacing,wavelength,distance)
    return matrix@u1@matrix.T/(1j*wavelength*distance)
