# -*- coding: utf-8 -*-
"""低内存周期Fresnel频域算子；数学模型对应LightPipes Forvard。"""
import numpy as np
from scipy.fft import fft2,ifft2


def periodic_fresnel(field,spacing,wavelength,distance,block_rows=64):
    """同一格点传播；只分块生成传递函数，不改物理域或波前。"""
    value=np.asarray(field)
    if value.ndim!=2 or value.shape[0]!=value.shape[1] or value.shape[0]%2 or not np.isfinite(value).all():
        raise ValueError('输入须有限偶数方阵')
    if any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in [spacing,wavelength,distance]):
        raise ValueError('步长、波长和距离须有限正值')
    if type(block_rows) is not int or block_rows<1:raise ValueError('块行数须正整数')
    # 复制调用方的数组；FFT允许重用本函数自己的缓冲，输入不被修改。
    spectrum=fft2(np.array(value,dtype=np.complex128,copy=True),overwrite_x=True,workers=1)
    freq=np.fft.fftfreq(len(value),d=spacing);squared=freq**2
    for y in range(0,len(value),block_rows):
        transfer=np.exp(-1j*np.pi*wavelength*distance*(squared[y:y+block_rows,None]+squared[None,:]))
        spectrum[y:y+block_rows]*=transfer
    output=ifft2(spectrum,overwrite_x=True,workers=1)
    output*=np.exp(1j*(2*np.pi/wavelength)*distance)
    return output


def power_on_grid(field,spacing,block_rows=64):
    """分块累计强度，避免再建立全域实数临时图。"""
    return float(sum(float(np.sum(np.abs(field[y:y+block_rows])**2)) for y in range(0,len(field),block_rows))*spacing**2)
