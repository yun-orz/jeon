# -*- coding: utf-8 -*-
"""只读独立审核G1复场、功率、测角及核库，向新文件保存证据。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO/'outputs/jeon2019_optics'
sys.path.insert(0,str(PROJECT))
from optics.materials import refractive_index_fused_silica


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    source=(PROJECT/args.run).resolve()
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if not report['numerical_validation_passed'] or report['status']!='completed':
        raise ValueError('G1数值未完整通过')
    for name,sha in json.loads((source/'source_manifest.json').read_text(encoding='utf-8')).items():
        if hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()!=sha:raise ValueError('源码指纹改变：'+name)
    with np.load(source/'arrays/height_fixed.npz',allow_pickle=False) as z:
        h=z['delta_h_m'];mask=z['mask'];x=z['x_m'];y=z['y_m'];dx=float(z['spacing_m'])
    with np.load(source/'arrays/psf_bank.npz',allow_pickle=False) as z:
        bank=z['kernels'];coarse=z['kernels_q_coarse'];fine=z['kernels_input_fine']
        wavelengths=z['wavelengths_m'];px=z['pixel_x_m'];py=z['pixel_y_m'];pitch=float(z['pitch_m'])
    np.testing.assert_array_equal(wavelengths,np.array([n/1e9 for n in range(420,661,10)]))
    assert bank.shape==(25,97,97)
    metrics=json.loads((source/'metrics/psf_metrics.json').read_text(encoding='utf-8'))
    comparisons=json.loads((source/'metrics/convergence.json').read_text(encoding='utf-8'))
    direct=[];angle_checks=[]
    X,Y=np.meshgrid(x,y);PX,PY=np.meshgrid(px,py);r=np.hypot(PX,PY);theta=np.arctan2(PY,PX)
    pin=float(mask.sum()*dx**2)
    for k,lam in enumerate(wavelengths):
        np.testing.assert_allclose(bank[k].sum(),metrics[k]['eta_window'],rtol=1e-14)
        for j,(a,b) in enumerate([(coarse[k],bank[k]),(bank[k],fine[k])]):
            value=float(np.abs(a-b).sum()/b.sum())
            np.testing.assert_allclose(value,comparisons[2*k+j]['kernel_l1'],rtol=1e-13)
        for item in metrics[k]['abs_encircled']:
            value=float(bank[k][r<=item['R_m']].sum())
            np.testing.assert_allclose(value,item['Eabs'],rtol=1e-13)
        band=(r>=20e-6)&(r<=100e-6)
        c3=np.sum(bank[k][band]*np.exp(3j*theta[band]))
        angle=float(np.degrees(np.angle(c3)/3))
        np.testing.assert_allclose(angle,metrics[k]['rotation_bands']['primary']['alpha_wrapped_deg'],atol=1e-11)
        angle_checks.append(angle)
        if round(lam*1e9) not in [420,540,660]:continue
        with np.load(source/f'arrays/field_{round(lam*1e9)}nm.npz',allow_pickle=False) as z:
            # 直接完整位移核求和，独立于生产程序的可分离矩阵乘法。
            u2=z['u2_complex'];intensity=z['intensity_raw'];nx=z['node_x_m'];ny=z['node_y_m']
            q=int(z['q'])
            u1=mask*np.exp(2j*np.pi*(refractive_index_fused_silica(lam)-1)*h/lam)
            distance=report['config']['optical']['distance_m']
            points=[(len(ny)//2,len(nx)//2),(len(ny)//3,len(nx)//4),(0,len(nx)-1)]
            expected=[];actual=[]
            for j,i in points:
                field=np.sum(u1*np.exp(1j*np.pi*((nx[i]-X)**2+(ny[j]-Y)**2)/(lam*distance)))
                field*=dx**2*np.exp(2j*np.pi*distance/lam)/(1j*lam*distance)
                expected.append(field);actual.append(u2[j,i])
            error=float(np.linalg.norm(np.array(actual)-expected)/np.linalg.norm(expected))
            if error>1e-10:raise ValueError('独立复场点求和不一致')
            np.testing.assert_array_equal(intensity,np.abs(u2)**2)
            # 逐像元循环独立积分，不采用生产reshape聚合路径。
            p=np.empty((len(py),len(px)))
            for j in range(len(py)):
                for i in range(len(px)):
                    p[j,i]=intensity[j*q:(j+1)*q,i*q:(i+1)*q].sum()*(pitch/q)**2
            np.testing.assert_allclose(p/pin,bank[k],rtol=2e-14,atol=0)
            direct.append(dict(wavelength_nm=round(lam*1e9),complex_point_relative_error=error))
    energy=json.loads((source/'metrics/native_energy.json').read_text(encoding='utf-8'))
    rotations=json.loads((source/'metrics/rotation.json').read_text(encoding='utf-8'))
    out=dict(passed=True,run=source.relative_to(REPO).as_posix(),independent_direct_field=direct,
             bank_shape=list(bank.shape),eta_min=float(bank.sum(axis=(1,2)).min()),eta_max=float(bank.sum(axis=(1,2)).max()),
             max_quadrature_kernel_l1=max(v['kernel_l1'] for v in comparisons if v['method']=='quadrature'),
             max_input_kernel_l1=max(v['kernel_l1'] for v in comparisons if v['method']=='input_refinement'),
             max_parseval_error=max(v['relative_error'] for v in energy),
             primary_rotation_change_deg=rotations['primary']['alpha_unwrapped_deg'][-1]-rotations['primary']['alpha_unwrapped_deg'][0],
             all_primary_angles_reliable=rotations['primary']['all_reliable'],angle_ambiguity=rotations['primary']['has_ambiguity'],
             R50_range_um=[min(v['R50']['radius_um'] for v in metrics),max(v['R50']['radius_um'] for v in metrics)],
             R80_range_um=[min(v['R80']['radius_um'] for v in metrics),max(v['R80']['radius_um'] for v in metrics)],
             threshold50_radius_range_um=[min(v['shape']['q0.5']['r_equiv_um'] for v in metrics),max(v['shape']['q0.5']['r_equiv_um'] for v in metrics)],
             outer150_power_over_Pin_range=[min(v['eta_window']-v['abs_encircled'][-1]['Eabs'] for v in metrics),max(v['eta_window']-v['abs_encircled'][-1]['Eabs'] for v in metrics)],
             outer_energy_definition='固定方窗内r>150μm功率/Pin；含螺旋翼和外围弱光，不称为严格旁瓣率',
             paper_alignment_passed=False)
    target=Path(__file__).resolve().parent/('independent_'+source.name+'.json')
    with target.open('x',encoding='utf-8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
