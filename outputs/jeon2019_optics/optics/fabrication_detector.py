# -*- coding: utf-8 -*-
"""固定刻蚀深度量化与理想方像元面积求积；不修改连续DOE设计。"""
from dataclasses import dataclass
import numpy as np
from .coordinates import Grid2D, Axis1D
from .doe import DOEHeightProfile


def quantize_relative_height(height, mask, step_m, levels, method):
    """以深度d=−Δh量化，拒绝越界；最近半级向较深侧舍入。"""
    h, m = np.asarray(height, dtype=np.float64), np.asarray(mask)
    if h.shape!=m.shape or not np.all(np.isfinite(h)) or not np.all((m==0)|(m==1)):
        raise ValueError("高度必须有限且与二值孔径掩膜同形")
    if not np.isfinite(step_m) or step_m<=0 or isinstance(levels,bool) or not isinstance(levels,int) or not 2<=levels<=32767:
        raise ValueError("级间距必须为正，级数必须为至少2的整数")
    inside=m==1
    if not inside.any():raise ValueError("有效孔径为空")
    scaled=-h[inside]/step_m
    # 仅消除舍入边界的机器浮点差；不截断超出制造范围的器件。
    eps=1e-12
    if np.any(scaled < -eps) or np.any(scaled > levels-1+eps):
        raise ValueError("连续高度超出可用刻蚀深度范围")
    if method=="nearest_depth":codes_inside=np.floor(scaled+.5+eps).astype(np.int16)
    elif method=="floor_depth":codes_inside=np.floor(scaled+eps).astype(np.int16)
    else:raise ValueError("未知深度量化方法："+str(method))
    codes=np.zeros(h.shape,dtype=np.int16);codes[inside]=codes_inside
    quantized=np.zeros_like(h);quantized[inside]=-codes_inside*step_m
    return quantized,codes


def quantized_profile(source, step_m, levels, method):
    h,codes=quantize_relative_height(source.delta_h,source.mask,step_m,levels,method)
    params=dict(source.params,quantization_method=method,depth_step_m=float(step_m),
                available_levels=levels,continuous_source_fingerprint=source.compute_fingerprint())
    return DOEHeightProfile(source.grid,h,source.mask,source.lambda_design,
                           "jeon2019_quantized_"+method,params),codes


@dataclass(frozen=True)
class MidpointAxis:
    """对称中点轴；偶数点位于光轴两侧，不能套用旧零点索引轴。"""
    n: int
    d: float
    label: str = "像元内求积"

    @property
    def coords(self):
        return (np.arange(self.n,dtype=np.float64)-(self.n-1)/2)*self.d

    @property
    def half_width(self):
        return (self.n-1)*self.d/2


def detector_grids(pixels,pitch_m,q):
    """固定像元中心及边界；求积点覆盖相同物理面积。"""
    if isinstance(pixels,bool) or not isinstance(pixels,int) or pixels<3 or pixels%2!=1:
        raise ValueError("光轴中心像元要求至少3个奇数像元")
    if isinstance(q,bool) or not isinstance(q,int) or q<1 or not np.isfinite(pitch_m) or pitch_m<=0:
        raise ValueError("求积密度必须为正整数，像元间距必须为有限正数")
    camera=Grid2D(Axis1D(pixels,pitch_m,"像元x"),Axis1D(pixels,pitch_m,"像元y"))
    nodes=Grid2D(MidpointAxis(pixels*q,pitch_m/q,"求积x"),MidpointAxis(pixels*q,pitch_m/q,"求积y"))
    edges=(np.arange(pixels+1,dtype=np.float64)-pixels/2)*pitch_m
    return camera,nodes,edges


def integrate_square_pixels(intensity,pixels,pitch_m,q):
    """由真实衍射节点强度作二维中点面积积分，不以中心点替代。"""
    I=np.asarray(intensity)
    detector_grids(pixels,pitch_m,q)
    if I.shape!=(pixels*q,pixels*q) or not np.all(np.isfinite(I)) or np.any(I<0):
        raise ValueError("求积强度必须非负、有限且形状匹配")
    power=I.reshape(pixels,q,pixels,q).sum(axis=(1,3))*(pitch_m/q)**2
    return power,power/pitch_m**2
