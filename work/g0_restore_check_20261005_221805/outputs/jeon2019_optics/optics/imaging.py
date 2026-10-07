# -*- coding: utf-8 -*-
"""非相干波段功率的线性卷积与严格伴随；不模拟复场干涉。"""
import numpy as np
from scipy.signal import fftconvolve


def real_finite(value, label):
    a = np.asarray(value)
    if np.iscomplexobj(a): raise ValueError(label+'必须为实数功率/实数测试向量')
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)): raise ValueError(label+'包含NaN或Inf')
    return a


def positive_shape(value):
    if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in value):
        raise ValueError('场景shape必须为两个正整数')
    return tuple(value)


class SpectralImager:
    """y=C Σ_b R_b (X_b*K_b)，A*含裁剪零回填和翻转核valid卷积。"""
    def __init__(self, kernels, scene_shape, pitch_m, response, crop=None):
        self.kernels = real_finite(kernels, 'PSF核').copy()
        if self.kernels.ndim != 3 or any(v % 2 != 1 for v in self.kernels.shape[1:]):
            raise ValueError('核须为[波段,奇数行,奇数列]')
        if self.kernels.shape[0] < 1 or np.any(self.kernels < 0) or np.any(self.kernels.sum(axis=(1,2)) <= 0):
            raise ValueError('每个非负PSF核必须有非零通量')
        self.scene_shape = positive_shape(scene_shape)
        self.pitch_m = float(pitch_m)
        if not np.isfinite(self.pitch_m) or self.pitch_m <= 0: raise ValueError('像平面格点间距须有限且为正')
        self.response = real_finite(response, '辐射响应').copy()
        if self.response.shape != (len(self.kernels),) or np.any(self.response < 0):
            raise ValueError('响应必须是与波段对应的非负数组')
        self.full_shape = tuple(n+k-1 for n,k in zip(self.scene_shape,self.kernels.shape[1:]))
        if crop is None: crop = [0,self.full_shape[0],0,self.full_shape[1]]
        if len(crop) != 4 or any(isinstance(v,bool) or not isinstance(v,int) for v in crop):
            raise ValueError('裁剪为整数[y0,y1,x0,x1]，终点不包含')
        y0,y1,x0,x1 = crop
        if not (0<=y0<y1<=self.full_shape[0] and 0<=x0<x1<=self.full_shape[1]):
            raise ValueError('裁剪越界或为空')
        self.crop = tuple(crop)
        self.slices = (slice(y0,y1),slice(x0,x1))
        self.output_shape = (y1-y0,x1-x0)
        self.kernels.setflags(write=False); self.response.setflags(write=False)

    @property
    def cube_shape(self): return (len(self.kernels),)+self.scene_shape

    def per_band_full(self, cube):
        # 算子允许有正负的实数测试向量；物理场景非负性由入口另行检查。
        x = real_finite(cube, '场景')
        if x.shape != self.cube_shape: raise ValueError('场景波段或空间shape不匹配')
        return np.stack([self.response[b]*fftconvolve(x[b],k,mode='full')
                         for b,k in enumerate(self.kernels)])

    def forward(self, cube): return self.per_band_full(cube).sum(axis=0)[self.slices]

    def adjoint(self, measurement):
        y = real_finite(measurement, '测量')
        if y.shape != self.output_shape: raise ValueError('测量shape不匹配')
        extended = np.zeros(self.full_shape)
        extended[self.slices] = y
        return np.stack([self.response[b]*fftconvolve(extended,k[::-1,::-1],mode='valid')
                         for b,k in enumerate(self.kernels)])

    def coordinates(self):
        """格点中心：偶数场景关于光轴对称，full轴为两输入轴起点之和。"""
        sy,sx = [(np.arange(n)-(n-1)/2)*self.pitch_m for n in self.scene_shape]
        ky,kx = [(np.arange(n)-(n-1)/2)*self.pitch_m for n in self.kernels.shape[1:]]
        fy = sy[0]+ky[0]+np.arange(self.full_shape[0])*self.pitch_m
        fx = sx[0]+kx[0]+np.arange(self.full_shape[1])*self.pitch_m
        return dict(scene_x_m=sx,scene_y_m=sy,kernel_x_m=kx,kernel_y_m=ky,
                    full_x_m=fx,full_y_m=fy,output_x_m=fx[self.slices[1]],output_y_m=fy[self.slices[0]])


def direct_shift_sum(cube, kernels, response):
    """逐点平移叠加的独立参考，用于检查FFT没有循环绕回或翻转。"""
    x = real_finite(cube, '参考场景')
    k = real_finite(kernels, '参考核')
    out = np.zeros((x.shape[1]+k.shape[1]-1,x.shape[2]+k.shape[2]-1))
    for b in range(x.shape[0]):
        for y,z in np.argwhere(x[b] != 0):
            out[y:y+k.shape[1],z:z+k.shape[2]] += response[b]*x[b,y,z]*k[b]
    return out


def operator_checks(imager, seed, target):
    """带符号内积及独立方向差分，不以重建图像外观判断算子正确性。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(imager.cube_shape)
    y = rng.standard_normal(imager.output_shape)
    lhs = float(np.vdot(imager.forward(x),y).real)
    rhs = float(np.vdot(x,imager.adjoint(y)).real)
    inner_error = abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
    direction = rng.standard_normal(imager.cube_shape)
    direction /= np.linalg.norm(direction)
    residual = imager.forward(x)-target
    analytic = float(np.vdot(imager.adjoint(residual),direction).real)
    epsilon = 1e-3*max(1.,np.linalg.norm(x))
    # 二次目标中心差分无截断高阶误差，稍大epsilon避免减法损失。
    def objective(z): return .5*np.linalg.norm(imager.forward(z)-target)**2
    fd = (objective(x+epsilon*direction)-objective(x-epsilon*direction))/(2*epsilon)
    return dict(inner_product_relative_error=inner_error,lhs=lhs,rhs=rhs,
                gradient_relative_error=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12),
                gradient_analytic=analytic,gradient_finite_difference=float(fd),epsilon=epsilon)
