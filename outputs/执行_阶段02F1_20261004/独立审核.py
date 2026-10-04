# -*- coding: utf-8 -*-
"""独立从落盘充分统计量和原始数组审核；不调用生产噪声函数。"""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'jeon2019_optics'
run=ROOT/'results/stage02f1/run_20261004_093359'
r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
def stream(seed,identity,role):
    digest=hashlib.sha256(json.dumps([identity,role],ensure_ascii=False,separators=(',',':')).encode('utf-8')).digest()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed]+np.frombuffer(digest,dtype='<u4').tolist())))
maximum=0.;negative=0
for rec in r['records']:
    with np.load(run/'arrays'/(rec['identity']+'.npz'),allow_pickle=False) as z:
        v={k:z[k].copy() for k in z.files}
    lam=v['wavelengths_m'];q=v['quantum_efficiency'];ref=float(v['reference_wavelength_m']);qref=q[np.argmin(abs(lam-ref))]
    response=q/qref*lam/ref
    np.testing.assert_allclose(v['electron_response'],response,rtol=1e-14,atol=0)
    np.testing.assert_allclose(v['photon_energy_J'],6.62607015e-34*299792458/lam,rtol=1e-14,atol=0)
    bands=[];truth=v['truth'];kernels=v['kernels'];crop=v['crop_indices'];ys=slice(int(crop[0]),int(crop[1]));xs=slice(int(crop[2]),int(crop[3]))
    for b in range(truth.shape[0]):
        full=np.zeros((truth.shape[1]+kernels.shape[1]-1,truth.shape[2]+kernels.shape[2]-1))
        for y,x in np.argwhere(truth[b]!=0):full[y:y+kernels.shape[1],x:x+kernels.shape[2]]+=truth[b,y,x]*kernels[b]*v['response'][b]
        bands.append(full[ys,xs])
    direct=np.asarray(bands);assert np.linalg.norm(direct-v['per_band_relative_power'])/np.linalg.norm(direct)<1e-12
    raw=float(v['gain_e_per_relative_power'])*np.einsum('b,byx->yx',response,v['per_band_relative_power'])+float(v['background_e'])
    np.testing.assert_allclose(v['mean_raw_e'],raw,rtol=1e-14,atol=1e-14)
    mu=np.maximum(raw,0);np.testing.assert_array_equal(mu,v['mean_sampling_e'])
    sigma=float(v['read_sigma_e']);np.testing.assert_array_equal(v['theoretical_variance_e2'],mu+sigma*sigma)
    ident=str(v['identity']);seed=int(v['seed'])
    shot=stream(seed,ident,'sample_shot').poisson(mu);read=stream(seed,ident,'sample_read').normal(0,sigma,mu.shape)
    np.testing.assert_array_equal(shot,v['shot_counts']);np.testing.assert_array_equal(read,v['read_noise_e']);np.testing.assert_array_equal(shot+read,v['noisy_measurement_e'])
    negative+=int((v['noisy_measurement_e']<0).sum())
    stat=rec['statistics'];n=stat['repeats'];sums=stat['sufficient_statistics'];variances={'shot':mu,'read':np.full(mu.shape,sigma*sigma),'joint':mu+sigma*sigma}
    for name,var in variances.items():
        second_var=(mu if name!='read' else np.zeros_like(mu))+2*var*var
        zm=sums[name]/np.sqrt(n*var.sum()) if var.sum()>0 else 0.
        zs=(sums[name+'_square']-n*var.sum())/np.sqrt(n*second_var.sum()) if second_var.sum()>0 else 0.
        assert np.isclose(zm,stat['checks'][name+'_mean']['z'],rtol=1e-12,atol=1e-12)
        assert np.isclose(zs,stat['checks'][name+'_second_moment']['z'],rtol=1e-12,atol=1e-12)
        assert max(abs(zm),abs(zs))<=stat['z_tolerance'];maximum=max(maximum,abs(zm),abs(zs))
assert len(r['records'])==18 and r['validation_passed']
result=dict(passed=True,jobs=18,max_abs_z=maximum,negative_readout_pixels=negative,checks=['独立逐点卷积','光子能量与QE响应','电子数均值/方差','独立随机流复现','原始充分统计量复算六个Z'])
(Path(__file__).parent/'独立审核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
