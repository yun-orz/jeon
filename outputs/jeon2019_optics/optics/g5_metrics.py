# -*- coding: utf-8 -*-
"""G5指标：固定尺度PSNR、逐波段局部SSIM与逐像元光谱角。"""
import numpy as np
from scipy.signal import convolve2d


def pair(target, prediction):
    """指标使用float64；拒绝形状不符、非有限输入和负的真实目标。"""
    a=np.asarray(target,dtype=np.float64);b=np.asarray(prediction,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=3 or min(a.shape)<1:
        raise ValueError('需要相同形状的H×W×B数据')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any():
        raise ValueError('真实目标必须非负，输入必须有限')
    return a,b


def ssim_band(a,b,data_range):
    """Wang 2004式(13)：11×11高斯窗sigma=1.5，人口矩，valid域，无下采样。

    根据论文公式独立实现；参考作者算法说明：
    https://www.cns.nyu.edu/~lcv/ssim/ 。不复制作者MATLAB源码。
    """
    if min(a.shape)<11:raise ValueError('SSIM空间尺寸须至少11')
    x=np.arange(-5,6,dtype=float);g=np.exp(-x*x/(2*1.5**2));g/=g.sum()
    w=np.outer(g,g)
    avg=lambda v:convolve2d(v,w,mode='valid')
    ma,mb=avg(a),avg(b)
    va=avg(a*a)-ma*ma;vb=avg(b*b)-mb*mb;cov=avg(a*b)-ma*mb
    c1=(.01*data_range)**2;c2=(.03*data_range)**2
    return ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))


def assess(target,prediction,data_range=1.,zero_norm_threshold=1e-12):
    """不裁剪、不逐图归一化。SAM零谱无定义，输出NaN图及明确计数。"""
    a,b=pair(target,prediction)
    if isinstance(data_range,bool) or not np.isfinite(data_range) or data_range<=0:
        raise ValueError('PSNR/SSIM参考幅度必须有限正数')
    if isinstance(zero_norm_threshold,bool) or not np.isfinite(zero_norm_threshold) or zero_norm_threshold<0:
        raise ValueError('零谱阈值必须有限非负')
    squared=(a-b)**2;mse=float(squared.mean());band_mse=squared.mean(axis=(0,1))
    # 完全一致时PSNR为正无穷，用null和单独状态表达，保证严格JSON兼容。
    psnr=lambda v:float(10*np.log10(data_range**2/v)) if v>0 else None
    maps=np.stack([ssim_band(a[:,:,k],b[:,:,k],data_range) for k in range(a.shape[2])],axis=-1)
    na=np.linalg.norm(a,axis=-1);nb=np.linalg.norm(b,axis=-1)
    valid=(na>zero_norm_threshold)&(nb>zero_norm_threshold)
    angle=np.full(na.shape,np.nan)
    cosine=np.sum(a[valid]*b[valid],axis=-1)/(na[valid]*nb[valid])
    angle[valid]=np.arccos(np.clip(cosine,-1.,1.))
    record=dict(mse=mse,psnr_cube_db=psnr(mse),psnr_cube_is_positive_infinity=mse==0,
        psnr_per_band_db=[psnr(float(v)) for v in band_mse],mse_per_band=band_mse.tolist(),
        ssim_mean=float(maps.mean()),ssim_per_band=maps.mean(axis=(0,1)).tolist(),
        sam_mean_rad=float(angle[valid].mean()) if valid.any() else None,
        sam_mean_deg=float(np.rad2deg(angle[valid].mean())) if valid.any() else None,
        sam_valid_pixels=int(valid.sum()),sam_undefined_pixels=int((~valid).sum()),
        target_zero_spectra=int((na<=zero_norm_threshold).sum()),prediction_zero_spectra=int((nb<=zero_norm_threshold).sum()),
        target_min=float(a.min()),target_max=float(a.max()),prediction_min=float(b.min()),prediction_max=float(b.max()),
        prediction_negative_values=int((b<0).sum()),target_above_reference_values=int((a>data_range).sum()),
        prediction_above_reference_values=int((b>data_range).sum()),data_range=data_range,
        shape=list(a.shape),ssim_valid_shape=list(maps.shape),prediction_clipped=False)
    return record,dict(sam_rad=angle,sam_valid=valid,ssim=maps,mse_spatial=squared.mean(axis=-1))
