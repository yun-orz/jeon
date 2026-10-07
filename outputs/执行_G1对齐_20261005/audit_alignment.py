# -*- coding: utf-8 -*-
"""独立审核保存数据：直接Fresnel求和、逐像元积分及冻结来源。"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ENGINE))
from main_g0_verify import verify_manifest
from optics.materials import refractive_index_fused_silica

RUN=ENGINE/'results/g1_alignment/run_20261005_234023'

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main():
    read=lambda name:json.loads((RUN/name).read_text(encoding='utf-8'))
    validation=read('metrics/validation.json')
    assert validation['status']=='completed' and validation['numerical_validation_passed']
    assert validation['paper_alignment_passed'] is False
    source=read('source_evidence.json');folder=Path(source['path'])
    assert {p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}==source['sha256']
    assert all(digest(ENGINE/name)==sha for name,sha in read('source_manifest.json').items())
    cfg=read('config_effective.json')['source_config'];zdist=cfg['optical']['distance_m'];dx=cfg['input']['spacing_m']
    with np.load(RUN/'arrays/height_cw_fixed.npz',allow_pickle=False) as z:
        height=z['delta_h_m'];mask=z['mask'];X,Y=np.meshgrid(z['x_m'],z['y_m'])
        design=z['lambda_design_m']
    # 高度公式重新展开，使用稳定光程差，不调用被审核的高度构造函数。
    radius=np.hypot(X,Y);focal=cfg['optical']['focal_length_m']
    delta=radius**2/(np.sqrt(focal**2+radius**2)+focal)
    expected=(np.floor(delta/design)*design-delta)/(refractive_index_fused_silica(design)-1)*mask
    height_error=float(np.max(np.abs(expected-height)));assert height_error<1e-18
    with np.load(RUN/'arrays/alignment_data.npz',allow_pickle=False) as z:
        bank=z['cw_kernels'];old=z['source_kernels'];waves=z['wavelengths_m'];x=z['pixel_x_m'];y=z['pixel_y_m'];pitch=float(z['pitch_m'])
    rows=[]
    for nm in [420,540,660]:
        with np.load(RUN/f'arrays/cw_field_{nm}nm.npz',allow_pickle=False) as data:
            lam=float(data['wavelength_m']);u=data['u2_complex'];intensity=data['intensity_raw'];pin=float(data['Pin'])
            nx=data['node_x_m'];ny=data['node_y_m'];power=data['pixel_power']
        u1=mask*np.exp(2j*np.pi/lam*(refractive_index_fused_silica(lam)-1)*height)
        assert abs(float(np.sum(np.abs(u1)**2)*dx**2)/pin-1)<1e-14
        errors=[]
        # 三点直接二维积分，独立于可分离矩阵乘法实现。
        for iy,ix in [(388,388),(430,350),(500,440)]:
            exact=np.exp(2j*np.pi*zdist/lam)/(1j*lam*zdist)*dx**2*np.sum(
                u1*np.exp(1j*np.pi/(lam*zdist)*((nx[ix]-X)**2+(ny[iy]-Y)**2)))
            errors.append(float(abs(exact-u[iy,ix])/max(np.max(np.abs(u)),1e-30)))
        assert max(errors)<1e-10
        reconstructed=np.empty((97,97))
        for iy in range(97):
            for ix in range(97):
                reconstructed[iy,ix]=intensity[iy*8:(iy+1)*8,ix*8:(ix+1)*8].sum()*(pitch/8)**2
        integration_error=float(np.max(np.abs(reconstructed-power))/power.max());assert integration_error<1e-14
        k=list(np.rint(waves*1e9).astype(int)).index(nm)
        assert np.array_equal(power/pin,bank[k])
        rows.append(dict(wavelength_nm=nm,direct_field_error=max(errors),pixel_integration_error=integration_error))
    reflection=max(float(np.abs(a-b[::-1]).sum()/b.sum()) for a,b in zip(bank,old));assert reflection<1e-10
    # 用保存核直接重新计算三阶角向矩，检查角度符号与累计量。
    XX,YY=np.meshgrid(x,y);rr=np.hypot(XX,YY);theta=np.arctan2(YY,XX);ring=(rr>=20e-6)&(rr<=100e-6)
    moments=np.array([np.sum(a[ring]*np.exp(3j*theta[ring])) for a in bank])
    angle=np.degrees(np.unwrap(np.angle(moments))/3)
    span=float(angle[-1]-angle[0]);assert abs(span-validation['cw_c3_span_deg'])<1e-10
    size=read('metrics/sizes.json');radial_error=max(abs(a[k]-b[k]) for a,b in zip(size['source'],size['cw']) for k in a if a[k] is not None)
    assert radial_error<1e-10
    checks=read('metrics/cw_checks.json');assert all(v['passed'] for key in ['sampling','native_energy','reflection'] for v in checks[key])
    frozen=verify_manifest(json.loads((ROOT/'baseline/g0_20261005/manifest.json').read_text(encoding='utf-8')))
    assert frozen['passed']
    result=dict(passed=True,run=str(RUN),source_unchanged=True,source_manifest_passed=True,
                height_formula_max_error_m=height_error,independent_field_checks=rows,reflection_l1_max=reflection,
                independent_c3_span_deg=span,radial_metric_max_difference=radial_error,g0=frozen)
    with (Path(__file__).parent/'independent_audit.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
