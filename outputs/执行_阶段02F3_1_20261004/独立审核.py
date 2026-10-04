# -*- coding: utf-8 -*-
"""独立空间域算子复算两目标、证书及误差分解，不调用生产控制模块。"""
import csv,json
from pathlib import Path
import numpy as np
from scipy.signal import correlate2d
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';run=ROOT/'results/stage02f3_1/run_20261004_100605'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'));source=ROOT/r['config']['source_run'];rows=[]
def norm(x):return float(np.sqrt(np.sum(np.asarray(x)**2)))
for rec in r['records']:
    with np.load(run/'arrays'/(rec['identity']+'.npz'),allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
    with np.load(source/'arrays'/(rec['source_identity']+'.npz'),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    for key in ['weights','alpha','L_data_estimate','kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','evaluation_truth','gain_e_per_relative_power','quantum_efficiency','effective_electron_response']:
        np.testing.assert_array_equal(v[key],old[key])
    np.testing.assert_array_equal(v['noisy_reconstruction'],old['reconstruction']);np.testing.assert_array_equal(v['noisy_measurement_e'],old['measurement_e'])
    assert rec['noisy_execution']=='reused_verified' and rec['noisy_new_iterations']==0 and rec['noisy_elapsed_this_run_s']==0
    t=v['evaluation_truth'];k=v['kernels'];crop=v['crop_indices'];sl=(slice(int(crop[0]),int(crop[1])),slice(int(crop[2]),int(crop[3])))
    full_shape=(t.shape[1]+k.shape[1]-1,t.shape[2]+k.shape[2]-1);response=v['effective_electron_response'];w=v['weights'];alpha=float(v['alpha']);background=float(v['background_e'])
    def forward(a):
        full=np.zeros(full_shape)
        for b in range(a.shape[0]):
            for y,x in np.argwhere(a[b]!=0):full[y:y+k.shape[1],x:x+k.shape[2]]+=response[b]*a[b,y,x]*k[b]
        return full[sl]
    def adjoint(a):
        full=np.zeros(full_shape);full[sl]=a
        return np.stack([response[b]*correlate2d(full,k[b],mode='valid') for b in range(k.shape[0])])
    np.testing.assert_allclose(v['ideal_measurement_e'],forward(t)+background,rtol=1e-11,atol=1e-10)
    np.testing.assert_array_equal(v['ideal_weighted_target'],w*(v['ideal_measurement_e']-background))
    certificates={}
    for name,x,z,pred,res,cert in [('noisy',v['noisy_reconstruction'],v['noisy_measurement_e'],v['noisy_prediction_e'],v['noisy_residual_e'],rec['noisy_certificate']),('ideal',v['ideal_reconstruction'],v['ideal_measurement_e'],v['ideal_prediction_e'],v['ideal_residual_e'],rec['ideal_certificate'])]:
        prediction=forward(x)+background;residual=prediction-z
        np.testing.assert_allclose(prediction,pred,rtol=1e-11,atol=1e-10);np.testing.assert_allclose(residual,res,rtol=1e-9,atol=1e-10)
        gradient=adjoint(w*w*residual)+alpha*x;bound=norm(np.where(x>0,gradient,np.minimum(gradient,0)))/alpha
        assert np.isclose(bound,cert['distance_upper_bound'],rtol=1e-5,atol=1e-11);certificates[name]=bound
        assert norm(x)>0 and bound/norm(x)<=.01 and np.all(x>=0)
        if name=='ideal':
            h=json.loads((run/'metrics'/(rec['identity']+'_optimization.json')).read_text(encoding='utf-8'))['history'];last=h[-1];L=last['L']
            objective=.5*np.sum((w*residual)**2)+.5*alpha*np.sum(x*x)
            assert np.isclose(objective,last['objective'],rtol=1e-11,atol=1e-9)
            pg=norm(L*(x-np.maximum(0,x-gradient/L)))/max(norm(adjoint(w*w*(z-background))),1e-300)
            assert np.isclose(pg,last['projected_gradient_relative'],rtol=1e-5,atol=1e-12) and pg<=rec['solver_config']['gradient_tolerance']
            assert rec['solver_status']=='converged' and last['objective_relative_change']<=rec['solver_config']['objective_tolerance']
            assert len(h)==rec['iterations'] and all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:]))
            with (run/'metrics'/(rec['identity']+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
            assert len(csvrows)==len(h)
            for crow,step in zip(csvrows,h):
                for key,value in step.items():
                    if isinstance(value,bool):assert crow[key]==str(value)
                    elif value is None:assert crow[key]==''
                    else:assert float(crow[key])==value
    xn=v['noisy_reconstruction'];x0=v['ideal_reconstruction'];total=xn-t;baseline=x0-t;noise=xn-x0;tn=norm(t);d=rec['decomposition']
    for key,a in [('error_total',total),('error_baseline',baseline),('error_noise_perturbation',noise)]:np.testing.assert_array_equal(v[key],a)
    np.testing.assert_allclose(total,baseline+noise,rtol=1e-13,atol=1e-15)
    cross=2*float(np.sum(baseline*noise));assert abs(norm(total)**2-norm(baseline)**2-norm(noise)**2-cross)<=1e-12*norm(total)**2
    for key,a in [('total',total),('baseline',baseline),('noise_perturbation',noise)]:assert np.isclose(d['norms_relative'][key],norm(a)/tn,rtol=1e-13,atol=0)
    assert np.isclose(d['cross_term'],cross,rtol=1e-13,atol=1e-15) and np.isclose(d['cross_term_relative_squared'],cross/(tn*tn),rtol=1e-13,atol=1e-15)
    for key,a,delta in [('baseline_interval',baseline,certificates['ideal']),('noise_perturbation_interval',noise,certificates['ideal']+certificates['noisy'])]:
        for bound,expected in [('lower',max(0.,norm(a)-delta)),('upper',norm(a)+delta)]:assert np.isclose(d[key][bound],expected,rtol=1e-5,atol=1e-11)
        assert np.isclose(d[key]['lower_relative'],d[key]['lower']/tn,rtol=1e-13,atol=0) and np.isclose(d[key]['upper_relative'],d[key]['upper']/tn,rtol=1e-13,atol=0)
    baseline_dominant=d['baseline_interval']['lower_relative']>d['noise_perturbation_interval']['upper_relative']
    rows.append(dict(identity=rec['identity'],baseline_relative=norm(baseline)/tn,noise_perturbation_relative=norm(noise)/tn,cross_term_relative_squared=cross/(tn*tn),baseline_dominant_under_optimization_bounds=baseline_dominant))
assert len(rows)==12 and r['validation_passed']
result=dict(passed=True,jobs=12,records=rows,all_baseline_dominant_under_optimization_bounds=all(a['baseline_dominant_under_optimization_bounds'] for a in rows),checks=['来源W/alpha/L/观测/旧恢复逐值冻结','空间域双目标前向/伴随/证书','控制停止与轨迹CSV逐字段','三误差cube与有符号交叉项','绝对/相对优化误差区间'])
with (HERE/'独立审核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('passed',len(rows),'baseline_dominant',result['all_baseline_dominant_under_optimization_bounds'])
