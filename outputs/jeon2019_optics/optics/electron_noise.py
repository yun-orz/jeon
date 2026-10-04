# -*- coding: utf-8 -*-
"""相对辐射功率到电子数的合成探测器模型；不使用恢复真值调参。"""
import hashlib,json
import numpy as np
from .imaging import real_finite
PLANCK_J_S=6.62607015e-34
LIGHT_M_S=299792458.

def electron_response(wavelengths_m,qe,reference_wavelength_m):
    lam=real_finite(wavelengths_m,'波长');q=real_finite(qe,'量子效率')
    if lam.ndim!=1 or q.shape!=lam.shape or np.any(lam<=0) or np.any(q<0) or np.any(q>1):raise ValueError('波长须正，QE须与波段对应且在[0,1]')
    ref=float(reference_wavelength_m)
    indices=np.flatnonzero(np.isclose(lam,ref,rtol=1e-12,atol=0))
    if not np.isfinite(ref) or ref<=0 or len(indices)!=1 or q[indices[0]]<=0:raise ValueError('参考波长须唯一属于波段且参考QE为正')
    response=(q/q[indices[0]])*(lam/ref)
    return response,PLANCK_J_S*LIGHT_M_S/lam

def sanitize_poisson_mean(raw,relative_tolerance):
    a=real_finite(raw,'原始期望电子数')
    if not np.isfinite(relative_tolerance) or relative_tolerance<=0:raise ValueError('舍入容差须正有限')
    scale=max(float(np.max(np.abs(a))),1e-300);tol=relative_tolerance*scale
    if np.any(a < -tol):raise ValueError('期望电子数出现超出舍入界的负值')
    cleaned=np.maximum(a,0.)
    return cleaned,dict(relative_tolerance=float(relative_tolerance),absolute_tolerance=tol,
        negative_count=int(np.count_nonzero(a<0)),minimum_raw=float(a.min()),added_electrons=float(np.sum(cleaned-a)))

def random_stream(seed,identity,role):
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('随机种子须非负整数')
    if not isinstance(identity,str) or not isinstance(role,str) or not identity or not role:raise ValueError('随机流身份和角色须非空字符串')
    payload=json.dumps([identity,role],ensure_ascii=False,separators=(',',':')).encode('utf-8')
    digest=hashlib.sha256(payload).digest();words=np.frombuffer(digest,dtype='<u4').astype(np.uint32).tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed]+words)))

def validate_mean_and_read(mean,read_sigma_e):
    mu=real_finite(mean,'采样期望电子数')
    if mu.size<1 or np.any(mu<0):raise ValueError('泊松期望须非负且非空')
    if isinstance(read_sigma_e,bool) or not np.isfinite(read_sigma_e) or read_sigma_e<0:raise ValueError('读出标准差须非负有限')
    return mu

def sample_electrons(mean,read_sigma_e,seed,identity):
    mu=validate_mean_and_read(mean,read_sigma_e)
    shot=random_stream(seed,identity,'sample_shot').poisson(mu)
    read=random_stream(seed,identity,'sample_read').normal(0,read_sigma_e,mu.shape)
    return dict(shot_counts=shot,read_noise_e=read,noisy_measurement_e=shot.astype(float)+read)

def noise_statistics(mean,read_sigma_e,seed,identity,repeats,block_size,z_tolerance):
    mu=validate_mean_and_read(mean,read_sigma_e)
    for value in [repeats,block_size]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('重复和分块数须正整数')
    if repeats<2 or not np.isfinite(z_tolerance) or z_tolerance<=0:raise ValueError('统计重复至少2次，标准化门槛须正有限')
    shot_rng=random_stream(seed,identity,'statistics_shot');read_rng=random_stream(seed,identity,'statistics_read')
    sums={k:0. for k in ['shot','shot_square','read','read_square','joint','joint_square']}
    for start in range(0,repeats,block_size):
        n=min(block_size,repeats-start);shape=(n,)+mu.shape
        shot=shot_rng.poisson(mu,size=shape).astype(float)-mu
        read=read_rng.normal(0,read_sigma_e,shape);joint=shot+read
        for label,a in [('shot',shot),('read',read),('joint',joint)]:
            sums[label]+=float(np.sum(a));sums[label+'_square']+=float(np.sum(a*a))
    read_var=read_sigma_e**2;variance=mu+read_var;N=repeats*mu.size
    theory=dict(shot=(float(mu.sum()),float(np.sum(mu+2*mu**2))),
        read=(mu.size*read_var,mu.size*2*read_var**2),joint=(float(variance.sum()),float(np.sum(mu+2*variance**2))))
    checks={};metrics={}
    for label,(sum_var,var_squared_sum) in theory.items():
        denom_mean=np.sqrt(repeats*sum_var);denom_second=np.sqrt(repeats*var_squared_sum)
        def standardized(num,den):
            if den>0:return float(num/den)
            if num==0:return 0.
            raise ValueError('零方差控制出现非零误差')
        zm=standardized(sums[label],denom_mean);zs=standardized(sums[label+'_square']-repeats*sum_var,denom_second)
        checks[label+'_mean']=dict(z=zm,passed=bool(abs(zm)<=z_tolerance))
        checks[label+'_second_moment']=dict(z=zs,passed=bool(abs(zs)<=z_tolerance))
        metrics[label]=dict(residual_mean_e=sums[label]/N,observed_second_moment_e2=sums[label+'_square']/N,
            expected_second_moment_e2=sum_var/mu.size,second_moment_ratio=sums[label+'_square']/(repeats*sum_var) if sum_var>0 else None)
    return dict(passed=all(v['passed'] for v in checks.values()),checks=checks,metrics=metrics,repeats=repeats,
        sufficient_statistics=sums,block_size=block_size,sample_count=N,z_tolerance=float(z_tolerance),
        interpretation='按已知像元均值的残差二阶矩；shot比值检验泊松Fano，不能对结构化图像的原始像素直接计算Fano',
        theory='Var(eps)=mu+sigma²；Var(eps²)=mu+2*(mu+sigma²)²。标准化门槛是大样本标准误筛查，非确定性百分比容差。')
