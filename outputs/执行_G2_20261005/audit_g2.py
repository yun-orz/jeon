# -*- coding: utf-8 -*-
"""独立审核G2保存结果：显式逐点卷积、响应混合及伴随。"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ROOT))
from main_g0_verify import verify_manifest

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def explicit_point(image,kernel,iy,ix):
    """从卷积定义展开有效输入域，不使用FFT或成像模块。"""
    h,w=image.shape;kh,kw=kernel.shape;cy,cx=kh//2,kw//2
    ys=np.arange(max(0,iy-cy),min(h,iy+cy+1));xs=np.arange(max(0,ix-cx),min(w,ix+cx+1))
    return float(np.sum(image[np.ix_(ys,xs)]*kernel[np.ix_(cy+iy-ys,cx+ix-xs)]))

def main(run):
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    validation=read('metrics/validation.json');assert validation['status']=='completed' and validation['numerical_validation_passed']
    assert validation['paper_alignment_passed'] is False and validation['real_response_verified'] is False
    assert all(sha(ENGINE/name)==value for name,value in read('source_manifest.json').items())
    evidence=read('source_evidence.json');source=Path(evidence['path'])
    assert {p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}==evidence['sha256']
    with np.load(run/'arrays/g2_measurement.npz',allow_pickle=False) as data:
        x=data['cube'];k=data['kernels'];r=data['response'];y=data['rgb'];blur=data['blurred_bands'];back=data['backprojection']
        waves=data['wavelengths_m'];unit=str(data['input_unit']);width=float(data['bin_width_nm'])
    c=read('config_effective.json');factor=1. if unit=='photons_per_bin' else width
    expected=np.asarray(c['response']['peak_qe'])[:,None]*np.exp(-.5*((waves[None,:]*1e9-np.asarray(c['response']['centers_nm'])[:,None])/np.asarray(c['response']['sigma_nm'])[:,None])**2)
    assert np.array_equal(r,expected)
    with np.load(source/'arrays/psf_bank.npz',allow_pickle=False) as data:assert np.array_equal(k,data['kernels'])
    errors=[];backerrors=[]
    for iy,ix in [(0,0),(0,95),(95,0),(95,95),(48,48),(37,63),(71,30)]:
        bands=np.array([explicit_point(x[:,:,b],kernel,iy,ix) for b,kernel in enumerate(k)])
        errors.extend(np.abs(bands-blur[iy,ix]).tolist())
        assert np.allclose(r@bands*factor,y[iy,ix],rtol=1e-12,atol=1e-10)
        for b,kernel in enumerate(k):
            mixed=np.sum(y*r[:,b],axis=-1)*factor
            direct=explicit_point(mixed,kernel[::-1,::-1],iy,ix)
            backerrors.append(abs(direct-back[iy,ix,b]))
    ferr=max(errors)/max(float(blur.max()),1.);berr=max(backerrors)/max(float(back.max()),1.)
    assert ferr<1e-12 and berr<1e-12
    full=np.array([sum(float(x[:,:,b].sum())*float(k[b].sum())*float(r[ch,b])*factor for b in range(25)) for ch in range(3)])
    actual=y.sum(axis=(0,1));ledger=read('metrics/flux_ledger.json')
    # 不同累加次序在约百万电子总量上出现约1e-14相对舍入差。
    # 审核容差仍严于配置中的1e-10，且不改变任何物理结果。
    np.testing.assert_allclose(full,ledger['full_convolution_electrons'],rtol=1e-12)
    np.testing.assert_allclose(actual,ledger['finite_image_electrons'],rtol=1e-12)
    frozen=verify_manifest(json.loads((ROOT/'baseline/g0_20261005/manifest.json').read_text(encoding='utf-8')));assert frozen['passed']
    result=dict(passed=True,run=str(run),source_unchanged=True,source_manifest_passed=True,
                explicit_forward_relative_error=ferr,explicit_adjoint_relative_error=berr,
                crop_loss_fraction=((full-actual)/full).tolist(),g0=frozen,
                limitations=['响应为合成QE','论文光学对齐仍未通过','Φᵀ仅为伴随，不是重建'])
    with (Path(__file__).parent/'independent_audit.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main(Path(sys.argv[1]))
