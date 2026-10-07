# -*- coding: utf-8 -*-
"""只读载入02C的真实像元积分核，拒绝缺失或不兼容证据。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .coordinates import make_grid
from .doe import design_jeon2019_spiral_height, compute_doe_transmission_field
from .fabrication_detector import quantized_profile, detector_grids, integrate_square_pixels


def tree_sha(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(path).rglob('*')) if p.is_file()}


def load_source(source, project_root, cfg):
    """三个高度及全部九组q8场重验；不能用元数据替代原始场检查。"""
    source, project_root = Path(source), Path(project_root)
    report = json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if report.get('status') != 'completed' or report.get('numerical_convergence_passed') is not True:
        raise ValueError('02C源run尚未通过求积筛查')
    frozen = json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    required = {'main_stage02c.py','main_stage02b.py','optics/doe.py','optics/materials.py',
                'optics/coordinates.py','optics/propagation.py','optics/fabrication_detector.py'}
    if not required <= set(frozen): raise ValueError('源run缺少光学实现源码指纹')
    for name, digest in frozen.items():
        path = (project_root/name).resolve()
        if not path.is_relative_to(project_root.resolve()): raise ValueError('源码清单路径越出项目')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('源光学源码不兼容：'+name)
    c = report['config']; o, f, d = c['optical'],c['fabrication'],c['detector']
    if c['optical']['wavelengths_m'] != cfg['wavelengths_m'] or cfg['wavelengths_m'] != [420e-9,540e-9,660e-9]:
        raise ValueError('本批要求源run的精确420/540/660 nm波长集合')
    if d['pitch_m'] != cfg['pitch_m'] or cfg['quadrature_per_axis'] != 8 or d['quadrature_per_axis'][-1] != 8:
        raise ValueError('必须使用6.22μm一致像元及已审核的q8积分核')
    if d['response'] != 'uniform_unit_radiometric' or d['fill_factor'] != 1: raise ValueError('源像元响应不兼容')
    if o['wings_N'] != 3: raise ValueError('本阶段仅载入单完整N=3 DOE')
    before = tree_sha(source)
    gi = make_grid(c['input']['n'],c['input']['spacing_m'])
    continuous = design_jeon2019_spiral_height(gi,o['diameter_m'],o['focal_length_m'],3,o['lambda_min_m'],o['lambda_max_m'])
    profiles = {'continuous':continuous}
    for method in ['nearest_depth','floor_depth']:
        profiles[method],_ = quantized_profile(continuous,f['depth_step_m'],f['levels'],method)
    powers, records, fingerprints = {}, [], {}
    expected = {key+'_'+str(round(lam*1e9))+'nm_q8' for key in profiles for lam in cfg['wavelengths_m']}
    actual = {key for key,value in report['records'].items() if value['q'] == 8}
    if actual != expected: raise ValueError('源q8场身份集合不完整或有额外身份')
    camera,nodes,edges = detector_grids(d['pixels_per_axis'],d['pitch_m'],8)
    for key,p in profiles.items():
        fp = p.compute_fingerprint(); fingerprints[key] = fp
        with np.load(source/'arrays'/('height_'+key+'.npz'),allow_pickle=False) as z:
            if not (np.array_equal(z['delta_h'],p.delta_h) and np.array_equal(z['mask'],p.mask)
                    and np.array_equal(z['lambda_design_m'],p.lambda_design)
                    and str(z['fingerprint']) == fp and str(z['design_type']) == p.design_type
                    and json.loads(str(z['params_json'])) == p.params):
                raise ValueError('固定高度/局部设计波长/指纹内容不一致：'+key)
        if report['fabrication'][key]['fingerprint'] != fp: raise ValueError('报告高度指纹不一致')
        kernels = []
        for lam in cfg['wavelengths_m']:
            ident = key+'_'+str(round(lam*1e9))+'nm_q8'; rec = report['records'][ident]
            with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                if not (int(z['q'])==8 and float(z['wavelength_m'])==lam and str(z['device'])==key
                        and str(z['fingerprint'])==fp and rec['fingerprint']==fp
                        and float(z['distance_m'])==o['distance_m'] and float(z['pitch_m'])==cfg['pitch_m']
                        and z['u2_complex'].shape==nodes.shape and z['intensity_raw'].shape==nodes.shape
                        and np.all(np.isfinite(z['u2_complex']))
                        and np.array_equal(z['node_x_m'],nodes.x.coords) and np.array_equal(z['node_y_m'],nodes.y.coords)
                        and np.array_equal(z['pixel_x_m'],camera.x.coords) and np.array_equal(z['pixel_y_m'],camera.y.coords)
                        and np.array_equal(z['pixel_edges_m'],edges) and float(z['node_spacing_m'])==nodes.dx
                        and np.array_equal(z['intensity_raw'],np.abs(z['u2_complex'])**2)):
                    raise ValueError('源复场身份、坐标或模平方不一致：'+ident)
                pin = float(np.sum(np.abs(compute_doe_transmission_field(p,lam))**2)*gi.cell_area)
                power,avg = integrate_square_pixels(z['intensity_raw'],d['pixels_per_axis'],d['pitch_m'],8)
                pwindow = float(power.sum()); eta = pwindow/pin
                if not (np.array_equal(power,z['pixel_power']) and np.array_equal(avg,z['pixel_mean_intensity'])
                        and np.isclose(pin,float(z['Pin']),rtol=1e-13,atol=0)
                        and np.isclose(pwindow,float(z['Pwindow']),rtol=1e-13,atol=0)
                        and np.isclose(eta,rec['eta_window'],rtol=1e-13,atol=0) and 0<eta<=1.02):
                    raise ValueError('源输入功率、像元积分或窗口捕获量不一致：'+ident)
                kernels.append(power/pin)
                records.append(dict(identity=ident,wavelength_m=lam,Pin=pin,Pwindow=pwindow,eta=eta,fingerprint=fp))
        powers[key] = np.stack(kernels)
    if tree_sha(source) != before: raise RuntimeError('载入期间源run被修改')
    return powers,dict(source_run=str(source.resolve()),sha256=before,validated_fields=records,
                      fingerprints=fingerprints,source_config=c,source_unchanged=True)
