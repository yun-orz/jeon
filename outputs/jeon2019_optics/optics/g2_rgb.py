# -*- coding: utf-8 -*-
"""G2：非相干强度卷积、显式光谱单位与零延拓同域伴随。"""
import numpy as np
from scipy.signal import fftconvolve


def synthetic_response(wavelengths_m, config):
    """自定义测试用QE，不是Canon标定，也不是作者发布的曲线。"""
    if config['kind']!='synthetic_gaussian':raise ValueError('本阶段只实现明确标记的合成响应')
    center=np.asarray(config['centers_nm'],dtype=float)
    sigma=np.asarray(config['sigma_nm'],dtype=float)
    peak=np.asarray(config['peak_qe'],dtype=float)
    if any(v.shape!=(3,) or not np.all(np.isfinite(v)) for v in [center,sigma,peak]) or np.any(sigma<=0) or np.any(peak<0) or np.any(peak>1):
        raise ValueError('RGB合成响应参数须为三个有限值，sigma>0且QE在[0,1]')
    wave=np.asarray(wavelengths_m,dtype=float)
    if wave.ndim!=1 or not np.all(np.isfinite(wave)) or np.any(wave<=0) or np.any(np.diff(wave)<=0):raise ValueError('波长须有限正值且严格递增')
    return peak[:,None]*np.exp(-.5*((wave[None,:]*1e9-center[:,None])/sigma[:,None])**2)


class RGBForward:
    """Φ=响应混合×空间卷积，ΦT为欧氏内积下的转置。

    核中心采用奇数尺寸中点。零延拓的full卷积中心裁剪，等价same；
    奇数核保证翻转核same为同域转置，图像尺寸允许奇偶。
    线性运算允许带符号向量，以便内积检查和后续优化；物理输入另行检查。
    """
    def __init__(self,kernels,response,input_unit='photons_per_bin',bin_width_nm=10.):
        self.kernels=np.array(kernels,dtype=np.float64,copy=True)
        self.response=np.array(response,dtype=np.float64,copy=True)
        k=self.kernels;r=self.response
        if k.ndim!=3 or k.shape[0]<1 or min(k.shape[1:])<1 or any(n%2!=1 for n in k.shape[1:]):raise ValueError('核须为[波段,奇数高,奇数宽]')
        if r.shape!=(3,k.shape[0]):raise ValueError('RGB响应须为[3,波段数]')
        if not np.all(np.isfinite(k)) or np.any(k<0) or not np.all(np.isfinite(r)) or np.any(r<0):raise ValueError('核与响应须有限且非负')
        if np.any(k.sum(axis=(1,2))<=0):raise ValueError('每个PSF须非零')
        if isinstance(bin_width_nm,bool) or not np.isfinite(bin_width_nm) or bin_width_nm<=0:raise ValueError('波段宽度须有限正值')
        if input_unit not in ['photons_per_bin','photons_per_nm']:raise ValueError('输入单位仅允许波段光子数或每nm光子谱密度')
        self.input_unit=input_unit;self.bin_width_nm=float(bin_width_nm)
        self.spectral_factor=1. if input_unit=='photons_per_bin' else self.bin_width_nm
        self.kernels.setflags(write=False);self.response.setflags(write=False)

    def check_cube(self,cube,physical=False):
        value=np.asarray(cube,dtype=np.float64)
        if value.ndim!=3 or value.shape[-1]!=len(self.kernels) or min(value.shape[:2])<1 or not np.all(np.isfinite(value)):
            raise ValueError('立方体须为有限[H,W,波段]数组')
        if physical and np.any(value<0):raise ValueError('物理光子输入不能为负')
        return value

    def convolve_bands(self,cube):
        value=self.check_cube(cube)
        return np.stack([fftconvolve(value[:,:,b],kernel,mode='same') for b,kernel in enumerate(self.kernels)],axis=-1)

    def forward(self,cube):
        blurred=self.convolve_bands(cube)
        return np.einsum('hwb,cb->hwc',blurred,self.response)*self.spectral_factor

    def adjoint(self,rgb):
        value=np.asarray(rgb,dtype=np.float64)
        if value.ndim!=3 or value.shape[-1]!=3 or min(value.shape[:2])<1 or not np.all(np.isfinite(value)):raise ValueError('伴随输入须为有限[H,W,3]数组')
        mixed=np.einsum('hwc,cb->hwb',value,self.response)*self.spectral_factor
        return np.stack([fftconvolve(mixed[:,:,b],kernel[::-1,::-1],mode='same') for b,kernel in enumerate(self.kernels)],axis=-1)

    def flux_ledger(self,cube,rgb):
        value=self.check_cube(cube,physical=True)
        ideal=self.response@(value.sum(axis=(0,1))*self.kernels.sum(axis=(1,2)))*self.spectral_factor
        actual=np.asarray(rgb).sum(axis=(0,1))
        return dict(full_convolution_electrons=ideal.tolist(),finite_image_electrons=actual.tolist(),
                    crop_loss_electrons=(ideal-actual).tolist(),kernel_capture_fraction=self.kernels.sum(axis=(1,2)).tolist())


def demo_scene(height,width,wavelengths_m,amplitude):
    """构造验算场景，绝非真实HSI训练集或颜色标定。"""
    yy,xx=np.indices((height,width));wave=wavelengths_m*1e9
    cube=np.zeros((height,width,len(wave)),dtype=float)
    features=[((xx-width*.28)**2+(yy-height*.30)**2<(min(height,width)*.13)**2,460.,18.),
              ((xx>width*.52)&(xx<width*.78)&(yy>height*.18)&(yy<height*.42),540.,25.),
              ((xx>width*.18)&(xx<width*.45)&(yy>height*.58)&(yy<height*.82),610.,20.)]
    for mask,center,sigma in features:cube+=mask[:,:,None]*np.exp(-.5*((wave-center)/sigma)**2)*amplitude
    cube[height//2,width//2,12]+=amplitude*8
    cube[0,0,0]+=amplitude*8
    return cube


def operator_diagnostics(operator,seed=0):
    """随机内积、中心/边缘点源及单位转换检查。"""
    rng=np.random.default_rng(seed);rows=[]
    for h,w in [(12,13),(13,12),(31,31)]:
        x=rng.normal(size=(h,w,len(operator.kernels)));y=rng.normal(size=(h,w,3))
        ax=operator.forward(x);aty=operator.adjoint(y)
        left=float(np.sum(ax*y));right=float(np.sum(x*aty))
        scale=max(np.linalg.norm(ax)*np.linalg.norm(y),np.linalg.norm(x)*np.linalg.norm(aty),1e-30)
        rows.append(dict(shape=[h,w],left=left,right=right,relative_error=float(abs(left-right)/scale)))
    kh,kw=operator.kernels.shape[1:];shape=(kh+4,kw+4,len(operator.kernels))
    point=np.zeros(shape);point[shape[0]//2,shape[1]//2,:]=1.
    y=operator.forward(point);ledger=operator.flux_ledger(point,y)
    expected=np.asarray(ledger['full_convolution_electrons']);actual=np.asarray(ledger['finite_image_electrons'])
    center_error=float(np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),1e-30))
    point[:]=0.;point[0,0,:]=1.;edge=operator.flux_ledger(point,operator.forward(point))
    density=RGBForward(operator.kernels,operator.response,'photons_per_nm',operator.bin_width_nm)
    integrated=RGBForward(operator.kernels,operator.response,'photons_per_bin',operator.bin_width_nm)
    x=rng.random((8,9,len(operator.kernels)))
    unit_error=float(np.max(np.abs(density.forward(x)-integrated.forward(x*operator.bin_width_nm))))
    return dict(adjoint=rows,center_point_relative_error=center_error,center_point=ledger,edge_point=edge,
                unit_conversion_max_error=unit_error,zero_input_max=float(np.max(np.abs(operator.forward(np.zeros_like(x))))))
