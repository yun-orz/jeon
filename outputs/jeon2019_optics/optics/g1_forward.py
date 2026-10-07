# -*- coding: utf-8 -*-
"""G1固定连续DOE的25波段PSF；复用冻结物理模块，不修改旧模型。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .coordinates import make_grid
from .doe import design_jeon2019_spiral_height, compute_doe_transmission_field
from .fabrication_detector import detector_grids, integrate_square_pixels
from .propagation import fresnel_kernel_separable
from .psf_analysis import psf_metrics, unwrap_angles_over_reliable_segments
from main_stage02c import differences, convergence_pass


def validate_config(c):
    """G1正式入口固定25波段；拒绝悄悄改变任务规模或无效采样。"""
    if c['wavelengths_nm'] != list(range(420, 661, 10)) or any(type(v) is not int for v in c['wavelengths_nm']):
        raise ValueError('G1要求420:10:660nm共25波段')
    o, g, d = c['optical'], c['input'], c['detector']
    for v in [o['diameter_m'], o['focal_length_m'], o['distance_m'], g['spacing_m'], d['pitch_m'], c['budget_seconds'], *c['thresholds'].values()]:
        if isinstance(v, bool) or not np.isfinite(v) or v <= 0:
            raise ValueError('物理量、预算和容差必须有限且为正')
    if type(o['wings_N']) is not int or o['wings_N'] != 3:
        raise ValueError('本批只验证单完整孔径N=3连续DOE')
    if o['lambda_min_m'] != 420e-9 or o['lambda_max_m'] != 660e-9:
        raise ValueError('本批器件设计范围固定420–660nm')
    if type(g['n']) is not int or g['n'] < 3 or g['n'] % 2 != 1 or (g['n']-1)*g['spacing_m'] < o['diameter_m']:
        raise ValueError('输入必须为覆盖完整孔径的奇数网格')
    if type(g['refinement_factor']) is not int or g['refinement_factor'] != 2:
        raise ValueError('本批固定二倍输入加密，物理窗口不变')
    if type(d['quadrature_coarse']) is not int or type(d['quadrature_main']) is not int or not 1 <= d['quadrature_coarse'] < d['quadrature_main']:
        raise ValueError('求积密度必须为递增正整数')
    _, _, edges = detector_grids(d['pixels_per_axis'], d['pitch_m'], d['quadrature_main'])
    if edges[-2] < 150e-6:
        raise ValueError('窗口须容纳150微米完整圆域')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results', 'show_plots']):
        raise ValueError('运行开关必须为布尔值')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi'] < 50:
        raise ValueError('图片dpi必须为至少50的整数')


def profile_for(c, refined=False):
    """只在不同空间网格采样同一解析设计，不随入射波长设计新器件。"""
    factor = c['input']['refinement_factor'] if refined else 1
    grid = make_grid((c['input']['n']-1)*factor+1, c['input']['spacing_m']/factor)
    o = c['optical']
    return design_jeon2019_spiral_height(grid, o['diameter_m'], o['focal_length_m'],
                                        o['wings_N'], o['lambda_min_m'], o['lambda_max_m'])


def measure(profile, wavelength, c, q):
    """传播复场，再在同一像元边界内积分强度；不作通量归一化补救。"""
    d = c['detector']
    camera, nodes, edges = detector_grids(d['pixels_per_axis'], d['pitch_m'], q)
    u1 = compute_doe_transmission_field(profile, wavelength, amplitude=1., h_offset=0.)
    u2 = fresnel_kernel_separable(u1, profile.grid, wavelength, c['optical']['distance_m'],
                                  nodes.x.coords, nodes.y.coords, include_global_phase=True)
    intensity = np.abs(u2)**2
    if not np.all(np.isfinite(u2)) or not np.all(np.isfinite(intensity)):
        raise ValueError('传播产生非有限复場或强度')
    pin = float(np.sum(np.abs(u1)**2)*profile.grid.cell_area)
    power, average = integrate_square_pixels(intensity, d['pixels_per_axis'], d['pitch_m'], q)
    eta = float(power.sum()/pin)
    if not (pin > 0 and power.max() > 0 and 0 < eta <= 1.02):
        raise ValueError('非零或有限窗口能量健康检查失败')
    metrics = psf_metrics(average, camera, pin, float(power.sum()),
        {'primary': {'r_min_m': 20e-6, 'r_max_m': 100e-6},
         'sensitivity': {'r_min_m': 30e-6, 'r_max_m': 120e-6}},
        {'r_max_m': 150e-6, 'quantiles': [.5, .1]},
        {'capture_radii_m': [50e-6, 100e-6, 150e-6]}, .05, .9*np.pi)
    metrics.update(wavelength_nm=round(wavelength*1e9), q=q, pin=pin,
                   fingerprint=profile.compute_fingerprint(), eta_window=eta)
    # 核值为每像元功率/入射孔径功率；和等于有限窗口捕获比例。
    return dict(u1=u1, u2=u2, intensity=intensity, power=power, kernel=power/pin,
                metrics=metrics, camera=camera, nodes=nodes, edges=edges)


def compare(a, b, c):
    """同一物理像元比较；细输入的Pin略异，另报告成像核差异。"""
    row = differences(a['power'], b['power'], a['metrics'], b['metrics'])
    row['kernel_l1'] = float(np.abs(a['kernel']-b['kernel']).sum()/b['kernel'].sum())
    row['kernel_l2'] = float(np.linalg.norm(a['kernel']-b['kernel'])/np.linalg.norm(b['kernel']))
    row['passed'] = bool(convergence_pass(row, c['thresholds'], c['detector']['pitch_m'])
                         and row['kernel_l1'] <= c['thresholds']['pixel_l1']
                         and row['kernel_l2'] <= c['thresholds']['pixel_l2'])
    return row


def native_parseval(u1, grid, wavelength, distance):
    """独立原生FFT全网格功率检查；不是相机网格、不是无混叠证明。"""
    n = grid.x.n
    size = n + n % 2
    padded = np.zeros((size, size), dtype=np.complex128)
    offset = (size-n+1)//2
    quadratic = np.exp(1j*np.pi*grid.radius()**2/(wavelength*distance))
    padded[offset:offset+n, offset:offset+n] = u1*quadratic
    transformed = np.fft.fft2(padded)*grid.cell_area/(wavelength*distance)
    out_spacing = wavelength*distance/(size*grid.dx)
    pin = float(np.sum(np.abs(u1)**2)*grid.cell_area)
    pout = float(np.sum(np.abs(transformed)**2)*out_spacing**2)
    return dict(pin=pin, pout=pout, relative_error=abs(pout/pin-1),
                native_spacing_m=out_spacing, passed=abs(pout/pin-1) <= 1e-10)


def rotation_summary(metrics, wavelengths):
    result = {}
    for band in ['primary', 'sensitivity']:
        rows = [m['rotation_bands'][band] for m in metrics]
        out = unwrap_angles_over_reliable_segments(wavelengths,
            [r['alpha_wrapped_rad'] for r in rows], [r['reliable'] for r in rows])
        out['alpha_unwrapped_deg'] = [None if v is None else float(np.degrees(v)) for v in out['alpha_unwrapped_rad']]
        out['A3'] = [r['A3'] for r in rows]
        out['all_reliable'] = all(r['reliable'] for r in rows)
        result[band] = out
    return result


def g0_physics_check(manifest_path, repo_root):
    """核验旧物理依赖原始字节；G1新增文件不改变G0快照。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    selected = [r for r in manifest['files'] if r['path'].startswith('outputs/jeon2019_optics/optics/')
                or r['path'] in ['outputs/jeon2019_optics/main_stage02c.py', 'outputs/jeon2019_optics/main_stage02b.py']]
    if not selected:
        raise ValueError('G0物理源码清单为空')
    for r in selected:
        path = (repo_root/r['path']).resolve()
        if not path.is_relative_to(repo_root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('G0冻结物理源码已变化：'+r['path'])
    return dict(passed=True, files=len(selected))
