# -*- coding: utf-8 -*-
"""用逐点前向与空间相关伴随独立核验电子重建，不调用生产重建模块。"""
import json,csv
from pathlib import Path
import numpy as np
from scipy.signal import correlate2d
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';run=ROOT/'results/stage02f2/run_20261004_095323'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'));source=ROOT/r['config']['source_run'];rows=[]
def norm(x):return float(np.sqrt(np.sum(np.asarray(x)**2)))
for rec in r['records']:
    with np.load(run/'arrays'/(rec['identity']+'.npz'),allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
    with np.load(source/'arrays'/(rec['source_identity']+'.npz'),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(v['measurement_e'],old['noisy_measurement_e']);np.testing.assert_array_equal(v['evaluation_truth'],old['truth'])
    x=v['reconstruction'];k=v['kernels'];crop=v['crop_indices'];sl=(slice(int(crop[0]),int(crop[1])),slice(int(crop[2]),int(crop[3])))
    full_shape=(x.shape[1]+k.shape[1]-1,x.shape[2]+k.shape[2]-1);lam=v['wavelengths_m'];qe=v['quantum_efficiency'];ref=float(v['reference_wavelength_m']);qref=qe[np.argmin(abs(lam-ref))]
    response=v['response']*float(v['gain_e_per_relative_power'])*(qe/qref)*(lam/ref)
    np.testing.assert_allclose(response,v['effective_electron_response'],rtol=1e-14,atol=0)
    def forward(a):
        full=np.zeros(full_shape)
        for b in range(a.shape[0]):
            for y,z in np.argwhere(a[b]!=0):full[y:y+k.shape[1],z:z+k.shape[2]]+=response[b]*a[b,y,z]*k[b]
        return full[sl]
    def adjoint(a):
        full=np.zeros(full_shape);full[sl]=a
        return np.stack([response[b]*correlate2d(full,k[b],mode='valid') for b in range(k.shape[0])])
    z=v['measurement_e'];background=float(v['background_e']);sigma=float(v['read_sigma_e']);alpha=float(v['alpha']);w=v['weights']
    expected_variance=np.maximum(np.maximum(z,0)+sigma*sigma,r['config']['reconstruction']['variance_floor_e2'])
    if rec['method']=='fixed_weighted':
        np.testing.assert_array_equal(v['variance_estimate_e2'],expected_variance);np.testing.assert_allclose(w,1/np.sqrt(expected_variance),rtol=1e-14,atol=0)
    else:np.testing.assert_array_equal(w,np.ones(z.shape));assert v['variance_estimate_e2'].size==0
    np.testing.assert_array_equal(v['weighted_target'],w*(z-background))
    prediction=forward(x)+background;residual=prediction-z
    np.testing.assert_allclose(prediction,v['prediction_e'],rtol=1e-11,atol=1e-10);np.testing.assert_allclose(residual,v['residual_e'],rtol=1e-9,atol=1e-10)
    assert np.all(x>=0) and np.all(np.isfinite(x));assert (z<0).sum()==rec['negative_measurement_count']
    gradient=adjoint(w*w*residual)+alpha*x
    h=json.loads((run/'metrics'/(rec['identity']+'_optimization.json')).read_text(encoding='utf-8'))['history'];last=h[-1];L=last['L']
    objective=.5*np.sum((w*residual)**2)+.5*alpha*np.sum(x*x)
    assert np.isclose(objective,last['objective'],rtol=1e-11,atol=1e-9)
    pg=norm(L*(x-np.maximum(0,x-gradient/L)))/max(norm(adjoint(w*w*(z-background))),1e-300)
    assert np.isclose(pg,last['projected_gradient_relative'],rtol=1e-5,atol=1e-12)
    bound=norm(np.where(x>0,gradient,np.minimum(gradient,0)))/alpha
    relative=bound/norm(x)
    assert np.isclose(relative,rec['certificate']['distance_upper_bound_relative'],rtol=1e-5,atol=1e-12) and relative<=.01
    error=norm(x-v['evaluation_truth'])/norm(v['evaluation_truth']);assert np.isclose(error,rec['evaluation']['cube_relative_l2'],rtol=1e-13,atol=0)
    assert alpha==r['config']['reconstruction']['alpha_factor']*float(v['L_data_estimate'])
    assert rec['solver_status']=='converged' and pg<=r['config']['reconstruction']['solver']['gradient_tolerance'] and last['objective_relative_change']<=r['config']['reconstruction']['solver']['objective_tolerance']
    assert len(h)==rec['iterations'] and all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:]))
    with (run/'metrics'/(rec['identity']+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==len(h)
    for crow,step in zip(csvrows,h):
        for key,value in step.items():
            if isinstance(value,bool):assert crow[key]==str(value)
            elif value is None:assert crow[key]==''
            else:assert float(crow[key])==value
    # 带符号随机向量检验独立空间参考的加权伴随。
    rng=np.random.default_rng(2026);testx=rng.normal(size=x.shape);testy=rng.normal(size=z.shape)
    lhs=float(np.sum(w*forward(testx)*testy));rhs=float(np.sum(testx*adjoint(w*testy)))
    assert abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)<1e-10
    rows.append(dict(identity=rec['identity'],objective=objective,projected_gradient=pg,certificate_relative=relative,cube_error=error,negative_observations=int((z<0).sum())))
assert len(rows)==12 and r['validation_passed']
result=dict(passed=True,jobs=12,records=rows,checks=['逐点电子数前向','空间相关加权伴随','原始负观测与固定权重','目标/投影梯度/强凸证书','不使用真值的alpha规则','轨迹CSV逐字段','真值误差仅作评价'])
(HERE/'独立审核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print('passed',len(rows))
