# -*- coding: utf-8 -*-
"""空间域复算训练/留出/完整目标与选择；不评价候选truth误差。"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.signal import correlate2d
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';run=ROOT/'results/stage02f3_2/run_20261004_231220'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
physics=load(run/'arrays/measurement_problem.npz');split=load(run/'arrays/split.npz');z=physics['measurement_e'];B=float(physics['background_e']);k=physics['kernels'];response=physics['effective_electron_response'];crop=physics['crop_indices'];sl=(slice(int(crop[0]),int(crop[1])),slice(int(crop[2]),int(crop[3])))
scene_shape=(len(physics['scene_y_m']),len(physics['scene_x_m']));full_shape=(scene_shape[0]+k.shape[1]-1,scene_shape[1]+k.shape[2]-1)
def norm(a):return float(np.sqrt(np.sum(np.asarray(a)**2)))
def forward(a):
    full=np.zeros(full_shape)
    for b in range(a.shape[0]):
        for y,x in np.argwhere(a[b]!=0):full[y:y+k.shape[1],x:x+k.shape[2]]+=response[b]*a[b,y,x]*k[b]
    return full[sl]
def adjoint(a):
    full=np.zeros(full_shape);full[sl]=a
    return np.stack([response[b]*correlate2d(full,k[b],mode='valid') for b in range(k.shape[0])])
# 独立重现索引随机流，不调用生产拆分函数。
c=r['config'];identity=c['split']['identity']+'_'+str(list(z.shape));payload=json.dumps([identity,'detector_split'],ensure_ascii=False,separators=(',',':')).encode('utf-8');digest=hashlib.sha256(payload).digest();rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([c['split']['seed']]+np.frombuffer(digest,dtype='<u4').tolist())))
order=rng.permutation(z.size);n=int(np.floor(c['split']['validation_fraction']*z.size));ti=split['train_indices'];vi=split['validation_indices']
np.testing.assert_array_equal(ti,order[n:]);np.testing.assert_array_equal(vi,order[:n]);assert not set(ti)&set(vi) and set(ti)|set(vi)==set(range(z.size))
np.testing.assert_array_equal(split['target_train_signal_e'],(z-B).ravel()[ti].reshape(1,-1));np.testing.assert_array_equal(split['target_validation_signal_e'],(z-B).ravel()[vi].reshape(1,-1))
assert not any('truth' in key or 'ideal' in key for key in physics)
records=[]
for rec in r['candidate_records']:
    v=load(run/'arrays'/(rec['identity']+'.npz'));x=v['reconstruction'];alpha=float(v['alpha']);prediction=forward(x);train_res=(prediction-(z-B)).ravel()[ti].reshape(1,-1);validation_res=(prediction-(z-B)).ravel()[vi].reshape(1,-1)
    np.testing.assert_allclose(v['prediction_train_signal_e'],prediction.ravel()[ti].reshape(1,-1),rtol=1e-11,atol=1e-10);np.testing.assert_allclose(v['residual_train_e'],train_res,rtol=1e-9,atol=1e-10);np.testing.assert_allclose(v['residual_validation_e'],validation_res,rtol=1e-9,atol=1e-10)
    score=float(np.mean(validation_res**2));assert np.isclose(score,rec['validation_mse_e2'],rtol=1e-11,atol=1e-10)
    scattered=np.zeros(z.shape);scattered.ravel()[ti]=train_res.ravel();gradient=adjoint(scattered)+alpha*x
    h=json.loads((run/'metrics'/(rec['identity']+'_optimization.json')).read_text(encoding='utf-8'))['history'];last=h[-1];L=last['L']
    objective=.5*np.sum(train_res**2)+.5*alpha*np.sum(x*x);assert np.isclose(objective,last['objective'],rtol=1e-11,atol=1e-8)
    target_scatter=np.zeros(z.shape);target_scatter.ravel()[ti]=(z-B).ravel()[ti]
    pg=norm(L*(x-np.maximum(0,x-gradient/L)))/max(norm(adjoint(target_scatter)),1e-300)
    bound=norm(np.where(x>0,gradient,np.minimum(gradient,0)))/alpha
    assert np.isclose(pg,last['projected_gradient_relative'],rtol=1e-4,atol=1e-12) and np.isclose(bound,rec['certificate']['distance_upper_bound'],rtol=1e-4,atol=1e-10)
    assert rec['solver_status']=='converged' and pg<=c['reconstruction']['solver']['gradient_tolerance'] and last['objective_relative_change']<=c['reconstruction']['solver']['objective_tolerance'] and bound/norm(x)<=.01
    assert np.all(x>=0) and alpha==rec['alpha_factor']*r['train_spectral_estimate']['value']
    assert len(h)==rec['iterations'] and all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:]))
    with (run/'metrics'/(rec['identity']+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==len(h)
    for row,step in zip(csvrows,h):
        for key,value in step.items():
            if isinstance(value,bool):assert row[key]==str(value)
            elif value is None:assert row[key]==''
            else:assert float(row[key])==value
    records.append(dict(identity=rec['identity'],factor=rec['alpha_factor'],validation_mse_e2=score,iterations=rec['iterations'],certificate_relative=bound/norm(x)))
assert len(records)==4
scores=[v['validation_mse_e2'] for v in records];chosen=int(np.argmin(scores));assert chosen==r['selection']['selected_index'];factor=records[chosen]['factor'];assert factor==r['selection']['selected_factor'] and r['selection']['truth_used'] is False and r['selection']['ideal_measurement_used'] is False
assert json.loads((run/'metrics/selection.json').read_text(encoding='utf-8'))==r['selection']
v=load(run/'arrays/full_refit.npz');x=v['reconstruction'];alpha=float(v['alpha']);prediction=forward(x)+B;res=prediction-z;rec=r['full_refit']
np.testing.assert_allclose(prediction,v['prediction_e'],rtol=1e-11,atol=1e-10);np.testing.assert_allclose(res,v['residual_e'],rtol=1e-9,atol=1e-10)
assert alpha==factor*float(v['L_full_estimate']) and alpha==rec['alpha']
gradient=adjoint(res)+alpha*x;bound=norm(np.where(x>0,gradient,np.minimum(gradient,0)))/alpha
assert np.isclose(bound,rec['certificate']['distance_upper_bound'],rtol=1e-4,atol=1e-10) and bound/norm(x)<=.01
last=rec['final_optimization'];L=last['L'];pg=norm(L*(x-np.maximum(0,x-gradient/L)))/max(norm(adjoint(z-B)),1e-300)
assert np.isclose(pg,last['projected_gradient_relative'],rtol=1e-4,atol=1e-12) and pg<=c['reconstruction']['solver']['gradient_tolerance'] and rec['solver_status']=='converged'
assert np.isclose(.5*np.sum(res**2)+.5*alpha*np.sum(x*x),last['objective'],rtol=1e-11,atol=1e-8)
# 只在选择已经核验后评价最终方案与固定旧基线。
e=load(run/'arrays/final_evaluation.npz');t=e['evaluation_truth'];old=e['baseline_reconstruction'];tn=norm(t);error=norm(x-t)/tn;baseline_error=norm(old-t)/tn
assert np.isclose(error,r['evaluation']['selected']['cube_relative_l2'],rtol=1e-13,atol=0) and np.isclose(baseline_error,r['evaluation']['baseline']['cube_relative_l2'],rtol=1e-13,atol=0)
old_delta=r['evaluation']['baseline_certificate']['distance_upper_bound']/tn;new_delta=rec['certificate']['distance_upper_bound']/tn
result=dict(passed=True,candidates=4,refits=1,selected_factor=factor,selected_at_weakest_grid_boundary=chosen==len(records)-1,candidate_records=records,final_cube_error=error,baseline_cube_error=baseline_error,final_error_optimization_interval=[max(0,error-new_delta),error+new_delta],baseline_error_optimization_interval=[max(0,baseline_error-old_delta),baseline_error+old_delta],checks=['独立复现随机拆分/索引/训练目标','空间域训练散射伴随与目标/证书','逐候选留出MSE和严格最小选择','实际训练/完整alpha标度','选择后最终评价与旧基线优化误差区间','候选轨迹CSV逐字段'])
with (HERE/'独立审核.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
with (HERE/'候选审核.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print(json.dumps(result,ensure_ascii=False,indent=2))
