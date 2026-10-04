# -*- coding: utf-8 -*-
"""只读正式run，独立重算输入相位、功率及半微米量化DOE传播。保留所有证据。"""
import json
import logging
from pathlib import Path
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]/'jeon2019_optics'
sys.path.insert(0,str(ROOT))
from optics.coordinates import make_grid
from optics.doe import DOEHeightProfile, design_jeon2019_spiral_height, compute_doe_transmission_field
from optics.fabrication_detector import detector_grids, integrate_square_pixels, quantized_profile
from optics.propagation import fresnel_kernel_separable
from main_stage02c import evaluate, differences, convergence_pass


def audit(run):
    report = json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    cfg = report['config']; gi = make_grid(cfg['input']['n'],cfg['input']['spacing_m'])
    profiles = {}
    results = []
    for key in report['fabrication']:
        with np.load(run/'arrays'/('height_'+key+'.npz'),allow_pickle=False) as z:
            p = DOEHeightProfile(gi,z['delta_h'].copy(),z['mask'].copy(),z['lambda_design_m'].copy(),
                                str(z['design_type']),json.loads(str(z['params_json'])))
            assert p.compute_fingerprint()==str(z['fingerprint'])
            profiles[key]=p
    for ident, rec in report['records'].items():
        with np.load(run/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
            q=int(z['q']); lam=float(z['wavelength_m']); p=profiles[rec['device']]
            camera,nodes,edges=detector_grids(cfg['detector']['pixels_per_axis'],float(z['pitch_m']),q)
            assert np.array_equal(z['node_x_m'],nodes.x.coords) and np.array_equal(z['node_y_m'],nodes.y.coords)
            assert np.array_equal(z['pixel_edges_m'],edges) and np.array_equal(z['pixel_x_m'],camera.x.coords)
            assert lam==rec['wavelength_nm']*1e-9 or np.isclose(lam,rec['wavelength_nm']*1e-9,rtol=1e-15)
            assert str(z['fingerprint'])==p.compute_fingerprint()
            u1=compute_doe_transmission_field(p,lam)
            pin=float(np.sum(np.abs(u1)**2)*gi.cell_area)
            I=np.abs(z['u2_complex'])**2
            assert np.array_equal(I,z['intensity_raw'])
            # 独立逐像元循环求和，避免仅重复模块中的reshape路径。
            power=np.empty(camera.shape)
            for y in range(camera.shape[0]):
                for x in range(camera.shape[1]): power[y,x]=I[y*q:(y+1)*q,x*q:(x+1)*q].sum()*nodes.cell_area
            assert np.allclose(power,z['pixel_power'],rtol=1e-13,atol=0)
            assert np.isclose(pin,float(z['Pin']),rtol=1e-13)
            assert np.isclose(power.sum(),float(z['Pwindow']),rtol=1e-13)
            results.append(ident)
    finegrid=make_grid(2201,.5e-6)
    o=cfg['optical']; fab=cfg['fabrication']; d=cfg['detector']
    continuous=design_jeon2019_spiral_height(finegrid,o['diameter_m'],o['focal_length_m'],3,o['lambda_min_m'],o['lambda_max_m'])
    fine={'continuous':continuous}
    for key in fab['methods']: fine[key],_=quantized_profile(continuous,fab['depth_step_m'],fab['levels'],key)
    camera,nodes,edges=detector_grids(d['pixels_per_axis'],d['pitch_m'],d['quadrature_per_axis'][-1])
    comparisons=[]
    out=Path(__file__).resolve().parent/('输入加密_'+run.name)
    out.mkdir(exist_ok=False)
    for key,p in fine.items():
        np.savez_compressed(out/('height_'+key+'.npz'),delta_h=p.delta_h,mask=p.mask,fingerprint=p.compute_fingerprint())
        for lam in o['wavelengths_m']:
            nm=round(lam*1e9); ident=key+'_'+str(nm)+'nm_q'+str(d['quadrature_per_axis'][-1])
            u1=compute_doe_transmission_field(p,lam)
            u2=fresnel_kernel_separable(u1,finegrid,lam,o['distance_m'],nodes.x.coords,nodes.y.coords,True)
            I=np.abs(u2)**2; power,avg=integrate_square_pixels(I,d['pixels_per_axis'],d['pitch_m'],d['quadrature_per_axis'][-1])
            pin=float(np.sum(np.abs(u1)**2)*finegrid.cell_area)
            metric=evaluate(avg,camera,pin,float(power.sum()))
            with np.load(run/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                comp=differences(z['pixel_power'],power,report['records'][ident],metric)
            comp.update(device=key,wavelength_nm=nm,passed=convergence_pass(comp,cfg['thresholds'],d['pitch_m']))
            comparisons.append(comp);print(json.dumps(comp,ensure_ascii=False))
            np.savez_compressed(out/(ident+'.npz'),u2_complex=u2,intensity_raw=I,pixel_power=power,
                node_x_m=nodes.x.coords,node_y_m=nodes.y.coords,Pin=pin,wavelength_m=lam,fingerprint=p.compute_fingerprint())
    payload={'source_run':str(run),'rechecked_fields':results,'input_fine_comparisons':comparisons,
             'all_passed':all(r['passed'] for r in comparisons),'reference_input_spacing_m':.5e-6,
             'reference_is_not_analytic_truth':True}
    (out/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('独立重读',len(results),'组；输入加密通过',payload['all_passed'])


if __name__=='__main__': audit(Path(sys.argv[1]))
