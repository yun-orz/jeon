# -*- coding: utf-8 -*-
"""光学对齐诊断：极坐标测角、独立尺寸统计及固定顺时针设计变体。"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .doe import DOEHeightProfile, optical_path_difference_delta
from .materials import refractive_index_fused_silica


def clockwise_profile(reference, origin_deg=0.):
    """先设计固定高度，再传播；不使用镜像图片生成PSF。

    图2(b)的波长标注沿顺时针增加。零角选择为实施假设，默认与G1共享+x零角。
    """
    if not np.isfinite(origin_deg):raise ValueError('角零点必须有限')
    X,Y=reference.grid.meshgrid();p=reference.params
    theta=np.mod(np.radians(origin_deg)-np.arctan2(Y,X),2*np.pi)
    period=2*np.pi/p['wings_N']
    wavelength=p['lambda_min_m']+(p['lambda_max_m']-p['lambda_min_m'])*np.mod(theta,period)/period
    wavelength=np.where((X==0)&(Y==0),p['lambda_min_m'],wavelength)
    delta=optical_path_difference_delta(np.hypot(X,Y),p['focal_length_m'])
    wrap=np.floor(delta/wavelength)
    height=(wrap*wavelength-delta)/(refractive_index_fused_silica(wavelength)-1)
    height=np.where(reference.mask==1,height,0.)
    params=dict(p,angular_chirality='clockwise',angular_origin_deg=float(origin_deg))
    return DOEHeightProfile(reference.grid,height,reference.mask.copy(),wavelength,'jeon2019_clockwise_diagnostic',params)


def polar_samples(bank,x,y,config):
    """仅作分析的双线性极坐标抽样；不生成成像核，不改变保存的PSF。"""
    nr,nt=config['r_samples'],config['theta_samples']
    if type(nr) is not int or type(nt) is not int or nr<3 or nt<12 or nt%6:
        raise ValueError('极坐标采样数非法；角样点数须为6的倍数')
    lo,hi=config['r_min_m'],config['r_max_m']
    if not 0<lo<hi or hi>min(max(x),-min(x),max(y),-min(y)):
        raise ValueError('极坐标环带必须在完整圆域内')
    r=np.linspace(lo,hi,nr);theta=np.arange(nt)*2*np.pi/nt
    points=np.column_stack(((r[:,None]*np.sin(theta)).ravel(),(r[:,None]*np.cos(theta)).ravel()))
    result=[]
    for image in bank:
        interpolator=RegularGridInterpolator((y,x),image,bounds_error=True)
        result.append(interpolator(points).reshape(nr,nt))
    return np.stack(result),r,theta


def angular_registration(a,b,r,correlation_min=.9):
    """对环带二维强度模式作相关；相邻谱角限制在[-60,60)度。

    去除各半径角向均值并乘sqrt(r)，避免轴对称中心光掩盖方向。
    分数是相似度诊断，不是测角误差概率或论文规定的置信度。
    """
    if a.shape!=b.shape or a.ndim!=2 or len(r)!=a.shape[0] or a.shape[1]%6:
        raise ValueError('角向配准形状非法')
    if not all(np.all(np.isfinite(v)) for v in [a,b,r]) or np.any(r<=0):
        raise ValueError('角向配准输入必须有限且半径为正')
    aa=(a-a.mean(axis=1,keepdims=True))*np.sqrt(r[:,None])
    bb=(b-b.mean(axis=1,keepdims=True))*np.sqrt(r[:,None])
    denominator=np.linalg.norm(aa)*np.linalg.norm(bb)
    if denominator<=np.finfo(float).tiny:
        return dict(angle_deg=None,score=None,reliable=False,reason='环带无角向结构')
    n=a.shape[1]
    curve=np.fft.ifft(np.sum(np.conj(np.fft.fft(aa,axis=1))*np.fft.fft(bb,axis=1),axis=0)).real/denominator
    offsets=np.arange(-n//6,n//6)
    values=curve[offsets%n]
    selected=int(np.argmax(values));offset=int(offsets[selected]);score=float(values[selected])
    interior=0<selected<len(offsets)-1
    return dict(angle_deg=float(offset*360/n) if score>=correlation_min and interior else None,
                score=score,reliable=bool(score>=correlation_min and interior),
                angular_resolution_deg=360/n,boundary_peak=not interior,
                raw_angle_deg=float(offset*360/n))


def registration_sequence(bank,x,y,config):
    samples,r,theta=polar_samples(bank,x,y,config)
    rows=[angular_registration(samples[k-1],samples[k],r,config['correlation_min']) for k in range(1,len(bank))]
    cumulative=[0.]
    # 不跨越不可靠配准续接累计角。
    for row in rows:
        cumulative.append(cumulative[-1]+row['angle_deg'] if cumulative[-1] is not None and row['reliable'] else None)
    return dict(adjacent=rows,cumulative_angle_deg=cumulative,all_reliable=all(v['reliable'] for v in rows)),samples,r,theta


def size_diagnostics(bank,x,y,pitch,config):
    """独立统计输入功率及窗内功率两种分母，保持同一物理尺寸单位。"""
    if not np.isfinite(pitch) or pitch<=0 or bank.ndim!=3 or bank.shape[1:]!=(len(y),len(x)):
        raise ValueError('尺寸网格非法')
    if not np.all(np.isfinite(bank)) or np.any(bank<0) or np.any(bank.sum(axis=(1,2))<=0):
        raise ValueError('尺寸核必须有限、非负且非零')
    roi=config['shape_roi_m'];qs=config['peak_thresholds']
    if not 0<roi<=min(max(x),-min(x),max(y),-min(y)) or not qs or any(not 0<q<1 for q in qs):
        raise ValueError('尺寸圆域或峰值阈值非法')
    X,Y=np.meshgrid(x,y);radius=np.hypot(X,Y)
    limit=min(max(x),-min(x),max(y),-min(y))
    mask=radius<=limit;rr=radius[mask];order=np.argsort(rr);sorted_radius=rr[order]
    # 同半径的一组像元全部累计后才判阈值，避免同半径像元顺序影响半径值。
    unique,ends=np.unique(sorted_radius,return_counts=True);ends=np.cumsum(ends)-1
    rows=[]
    for image in bank:
        total=float(image.sum());cdf=np.cumsum(image[mask][order])[ends]
        row=dict(eta_window=total)
        for name,denominator in [('input',1.),('window',total)]:
            for fraction in [.5,.8]:
                indices=np.flatnonzero(cdf>=fraction*denominator)
                row[f'R{round(fraction*100)}_{name}_um']=float(unique[indices[0]]*1e6) if len(indices) else None
        shape=radius<=roi;peak=float(image.max())
        for q in qs:
            area=int(np.count_nonzero(shape&(image>=q*peak)))*pitch**2
            row[f'peak_q{q}_equivalent_radius_um']=float(np.sqrt(area/np.pi)*1e6)
        core=image[shape];r_core=radius[shape]
        row['core_rms_radius_um']=float(np.sqrt(np.sum(core*r_core**2)/core.sum())*1e6)
        row['outer_power_over_input']=float(image[~shape].sum())
        row['outer_fraction_of_window']=row['outer_power_over_input']/total
        rows.append(row)
    return rows
