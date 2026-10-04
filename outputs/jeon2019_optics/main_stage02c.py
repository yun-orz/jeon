# -*- coding: utf-8 -*-
"""02C：固定制造高度及理想像元积分；可直接在PyCharm运行。"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback
import numpy as np
from optics.coordinates import make_grid
from optics.doe import design_jeon2019_spiral_height
from optics.fabrication_detector import quantized_profile, detector_grids, integrate_square_pixels
from optics.materials import refractive_index_fused_silica
from optics.psf_analysis import psf_metrics
from optics.convergence import periodic_angle_difference_deg
from optics.stage02_runtime import configure_plotting, effective_runtime
from optics.runutil import environment_info
from main_stage02b import propagate_device, field_health_check, allocate_unique_run_dir, setup_logger

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate_config(c):
    o, inp, f, d = (c[k] for k in ('optical', 'input', 'fabrication', 'detector'))
    for v in [o['diameter_m'], o['focal_length_m'], o['distance_m'], o['lambda_min_m'],
              o['lambda_max_m'], inp['spacing_m'], f['depth_step_m'], f['substrate_height_m']]:
        if not np.isfinite(v) or v <= 0: raise ValueError('物理参数必须有限且为正')
    if o['wings_N'] != 3: raise ValueError('本阶段仅验证N=3')
    w = o['wavelengths_m']
    if not w or w != sorted(set(w)) or not all(o['lambda_min_m'] <= v <= o['lambda_max_m'] for v in w):
        raise ValueError('波长必须递增、不重复并位于设计范围内')
    if len(set(round(v*1e9) for v in w)) != len(w): raise ValueError('波长的整数nm文件标识冲突')
    n = inp['n']
    if isinstance(n, bool) or not isinstance(n, int) or n % 2 != 1 or (n-1)*inp['spacing_m'] < o['diameter_m']:
        raise ValueError('输入网格须为覆盖完整孔径的奇数网格')
    if isinstance(f['levels'], bool) or not isinstance(f['levels'], int) or not 2 <= f['levels'] <= 32767:
        raise ValueError('制造级数须为2至32767的整数')
    if f['substrate_height_m'] < (f['levels']-1)*f['depth_step_m']:
        raise ValueError('基底薄于最大刻蚀深度')
    if f['methods'] != ['nearest_depth', 'floor_depth']: raise ValueError('本批比较两种指定深度舍入方法')
    qs = d['quadrature_per_axis']
    if len(qs) != 2 or qs != sorted(set(qs)): raise ValueError('本批要求两个递增求积密度')
    for q in qs: detector_grids(d['pixels_per_axis'], d['pitch_m'], q)
    if (d['pixels_per_axis']-1)*d['pitch_m']/2 < 150e-6: raise ValueError('评价圆域必须完整容纳150微米半径')
    if d['response'] != 'uniform_unit_radiometric' or d['fill_factor'] != 1:
        raise ValueError('本阶段仅实现单位辐射响应、满填充理想像元')
    for v in c['thresholds'].values():
        if not np.isfinite(v) or v <= 0: raise ValueError('验证容差必须有限且为正')


def evaluate(I, grid, pin, power):
    # 延用02B的固定物理环带，像元指标按像元中心离散归属计算。
    return psf_metrics(I, grid, pin, power,
        {'primary': {'r_min_m': 20e-6, 'r_max_m': 100e-6},
         'sensitivity': {'r_min_m': 30e-6, 'r_max_m': 120e-6}},
        {'r_max_m': 150e-6, 'quantiles': [.5, .1]},
        {'capture_radii_m': [50e-6, 100e-6, 150e-6]}, .05, .9*np.pi)


def differences(a, b, ma, mb):
    delta = a-b
    out = {'pixel_l1': float(np.abs(delta).sum()/b.sum()),
           'pixel_l2': float(np.linalg.norm(delta)/np.linalg.norm(b)),
           'eta_absolute': abs(ma['eta_window']-mb['eta_window'])}
    for band in ('primary', 'sensitivity'):
        ra, rb = ma['rotation_bands'][band], mb['rotation_bands'][band]
        out['angle_'+band+'_deg'] = periodic_angle_difference_deg(ra['alpha_wrapped_deg'], rb['alpha_wrapped_deg'])
        out['reliability_'+band+'_match'] = ra['reliable'] == rb['reliable']
    for r in ('R50', 'R80'):
        out[r+'_status_match'] = ma[r]['status'] == mb[r]['status']
        x, y = ma[r]['radius_um'], mb[r]['radius_um']
        out[r+'_difference_um'] = abs(x-y) if x is not None and y is not None else None
    return out


def convergence_pass(row, thresholds, pitch):
    checks = [row[k] <= thresholds[k] for k in ('pixel_l1', 'pixel_l2', 'eta_absolute')]
    for b in ('primary', 'sensitivity'):
        checks.append(row['reliability_'+b+'_match'])
        v = row['angle_'+b+'_deg']
        if v is not None: checks.append(abs(v) <= thresholds['angle_deg'])
    for r in ('R50', 'R80'):
        checks.append(row[r+'_status_match'])
        v = row[r+'_difference_um']
        if v is not None: checks.append(v <= thresholds['radius_pixels']*pitch*1e6)
    return bool(all(checks))


def run(c, no_save=False, show_plots=False):
    validate_config(c)
    runtime = effective_runtime(c, no_save=no_save, show_plots=show_plots)
    plt, backend = configure_plotting(runtime['show_plots'])
    saved = runtime['save_results']
    dest = allocate_unique_run_dir('stage02c') if saved else None
    logger = setup_logger(dest) if saved else logging.getLogger('stage02c_memory')
    if not saved:
        logger.setLevel(logging.INFO)
        if not logger.handlers: logger.addHandler(logging.StreamHandler())
    start = time.perf_counter()
    o, f, d = c['optical'], c['fabrication'], c['detector']
    gi = make_grid(c['input']['n'], c['input']['spacing_m'])
    continuous = design_jeon2019_spiral_height(gi, o['diameter_m'], o['focal_length_m'],
                                              o['wings_N'], o['lambda_min_m'], o['lambda_max_m'])
    profiles = {'continuous': continuous}
    codes = {}
    for method in f['methods']:
        profiles[method], codes[method] = quantized_profile(continuous, f['depth_step_m'], f['levels'], method)
    fingerprints = {key: p.compute_fingerprint() for key, p in profiles.items()}
    rows, convergences, impacts, center_comparisons, fabrication = {}, [], [], [], {}
    arrays, expected_arrays = {}, []
    camera, _, edges = detector_grids(d['pixels_per_axis'], d['pitch_m'], d['quadrature_per_axis'][-1])
    if saved: write_json(dest/'config_effective.json', dict(c, runtime=runtime))
    try:
        for key, profile in profiles.items():
            inside = profile.mask == 1
            error = profile.delta_h[inside]-continuous.delta_h[inside]
            fabrication[key] = {'fingerprint': fingerprints[key], 'max_abs_error_nm': float(np.max(np.abs(error))*1e9),
                'mean_error_nm': float(error.mean()*1e9), 'rms_error_nm': float(np.sqrt(np.mean(error**2))*1e9),
                'occupied_codes': sorted(np.unique(codes[key][inside]).tolist()) if key in codes else None}
            if saved:
                name = 'height_'+key+'.npz'; expected_arrays.append(name)
                np.savez_compressed(dest/'arrays'/name, delta_h=profile.delta_h, mask=profile.mask,
                    absolute_height_inside_m=np.where(inside, f['substrate_height_m']+profile.delta_h, 0),
                    codes=codes.get(key, np.full(profile.delta_h.shape, -1, dtype=np.int16)),
                    x_m=gi.x.coords, y_m=gi.y.coords, lambda_design_m=profile.lambda_design,
                    fingerprint=fingerprints[key], params_json=json.dumps(profile.params, ensure_ascii=False),
                    design_type=profile.design_type)
            for wavelength in o['wavelengths_m']:
                nm = round(wavelength*1e9)
                base = key+'_'+str(nm)+'nm'
                qrows = []
                for q in [1]+d['quadrature_per_axis']:
                    _, nodes, _ = detector_grids(d['pixels_per_axis'], d['pitch_m'], q)
                    record = propagate_device(profile, wavelength, gi, nodes, True, 1., 0.,
                                              o['distance_m'], logger, base+'_q'+str(q))
                    u2, I = record['u2'], record['intensity']
                    health = field_health_check(record['u1'], u2, I, gi, nodes, 1.02 if q != 1 else 100.)
                    if health['status'] != 'pass': raise RuntimeError(str(health))
                    power, avg = integrate_square_pixels(I, d['pixels_per_axis'], d['pitch_m'], q)
                    if not np.isclose(power.sum(), health['Pwindow'], rtol=1e-13, atol=0):
                        raise RuntimeError('像元求积与节点总功率不一致')
                    metrics = evaluate(avg, camera, health['Pin'], float(power.sum()))
                    phase_error = 2*np.pi*(refractive_index_fused_silica(wavelength)-1)*error/wavelength
                    metrics.update(device=key, wavelength_nm=nm, q=q, fingerprint=fingerprints[key],
                        phase_error_mean_rad=float(phase_error.mean()),
                        phase_error_rms_rad=float(np.sqrt(np.mean(phase_error**2))),
                        phase_error_std_rad=float(phase_error.std()), field_health=health,
                        measurement='center_point_approximation' if q == 1 else 'pixel_area_integral')
                    ident = base+'_q'+str(q); rows[ident] = metrics
                    arrays[ident] = power
                    if q != 1: qrows.append(ident)
                    if saved:
                        name = ident+'.npz'; expected_arrays.append(name)
                        np.savez_compressed(dest/'arrays'/name, u2_complex=u2, intensity_raw=I,
                            node_x_m=nodes.x.coords, node_y_m=nodes.y.coords, node_spacing_m=nodes.dx,
                            pixel_x_m=camera.x.coords, pixel_y_m=camera.y.coords, pixel_edges_m=edges,
                            pixel_power=power, pixel_mean_intensity=avg, pitch_m=d['pitch_m'], q=q,
                            wavelength_m=wavelength, Pin=health['Pin'], Pwindow=float(power.sum()),
                            fingerprint=fingerprints[key], device=key, distance_m=o['distance_m'])
                    if profile.compute_fingerprint() != fingerprints[key]: raise RuntimeError('固定高度发生变化')
                lo, hi = qrows
                comp = differences(arrays[lo], arrays[hi], rows[lo], rows[hi])
                comp.update(device=key, wavelength_nm=nm, passed=convergence_pass(comp, c['thresholds'], d['pitch_m']))
                convergences.append(comp)
                center_comparisons.append(dict(device=key, wavelength_nm=nm,
                    **differences(arrays[base+'_q1'], arrays[hi], rows[base+'_q1'], rows[hi])))
        qhi = d['quadrature_per_axis'][-1]
        for method in f['methods']:
            for wavelength in o['wavelengths_m']:
                nm = round(wavelength*1e9)
                a, b = method+'_'+str(nm)+'nm_q'+str(qhi), 'continuous_'+str(nm)+'nm_q'+str(qhi)
                impacts.append(dict(device=method, wavelength_nm=nm,
                                    **differences(arrays[a], arrays[b], rows[a], rows[b])))
        figure_names = []
        fig, axs = plt.subplots(3, len(o['wavelengths_m']), figsize=(12, 10), squeeze=False)
        extent = [edges[0]*1e6, edges[-1]*1e6, edges[0]*1e6, edges[-1]*1e6]
        for i, key in enumerate(profiles):
            for j, wavelength in enumerate(o['wavelengths_m']):
                nm = round(wavelength*1e9); ident = key+'_'+str(nm)+'nm_q'+str(qhi)
                image = arrays[ident]
                axs[i,j].imshow(image/image.max(), origin='lower', extent=extent, cmap='inferno')
                axs[i,j].set_title('%s | %d nm\nη=%.4f' % (key, nm, rows[ident]['eta_window']))
                axs[i,j].set_xlabel('x (μm)'); axs[i,j].set_ylabel('y (μm)')
        fig.suptitle('固定DOE：像元积分PSF；每幅峰值归一化仅用于显示')
        fig.tight_layout()
        if saved:
            fig.savefig(dest/'figures'/'pixel_psfs.png', dpi=c['plots']['dpi']); figure_names.append('pixel_psfs.png')
        fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
        labels = [r['device']+' '+str(r['wavelength_nm']) for r in convergences]
        axs2[0].bar(range(len(labels)), [100*r['pixel_l1'] for r in convergences])
        axs2[0].axhline(100*c['thresholds']['pixel_l1'], color='r', linestyle='--')
        axs2[0].set_xticks(range(len(labels)), labels, rotation=70); axs2[0].set_ylabel('加密求积L1差异 (%)')
        axs2[1].bar(range(len(labels)), [100*r['pixel_l1'] for r in center_comparisons])
        axs2[1].set_xticks(range(len(labels)), labels, rotation=70); axs2[1].set_ylabel('像元中心近似相对面积积分的L1差异 (%)')
        fig2.tight_layout()
        if saved:
            fig2.savefig(dest/'figures'/'integration_checks.png', dpi=c['plots']['dpi']); figure_names.append('integration_checks.png')
        readback = []
        if saved:
            # 完成状态须在原始复场、像元功率及文件集合重新读取通过后产生。
            actual = sorted(p.name for p in (dest/'arrays').glob('*.npz'))
            if actual != sorted(expected_arrays): raise RuntimeError('原始数组交付集合不匹配')
            for key, profile in profiles.items():
                with np.load(dest/'arrays'/('height_'+key+'.npz'), allow_pickle=False) as z:
                    if not (np.array_equal(z['delta_h'], profile.delta_h) and np.array_equal(z['mask'], profile.mask)
                            and str(z['fingerprint']) == profile.compute_fingerprint()):
                        raise RuntimeError('保存的固定高度或掩膜不一致：'+key)
            for ident, metrics in rows.items():
                with np.load(dest/'arrays'/(ident+'.npz'), allow_pickle=False) as z:
                    p, avg = integrate_square_pixels(z['intensity_raw'], d['pixels_per_axis'], d['pitch_m'], metrics['q'])
                    ok = (np.array_equal(z['intensity_raw'], np.abs(z['u2_complex'])**2)
                          and np.array_equal(p, z['pixel_power']) and np.array_equal(avg, z['pixel_mean_intensity'])
                          and str(z['fingerprint']) == metrics['fingerprint']
                          and np.isclose(p.sum(), float(z['Pwindow']), rtol=1e-13, atol=0))
                    if not ok: raise RuntimeError('原始场重新读取检查失败：'+ident)
                    readback.append(ident)
            for name in figure_names:
                if not (dest/'figures'/name).is_file(): raise RuntimeError('结果图缺失')
        report = dict(status='completed', numerical_convergence_passed=all(r['passed'] for r in convergences),
            config=c, runtime=runtime, environment=environment_info(), backend=backend,
            fabrication=fabrication, records=rows, quadrature_comparisons=convergences,
            quantization_impacts=impacts, center_point_comparisons=center_comparisons,
            readback_fields=readback, elapsed_s=time.perf_counter()-start,
            scope='Jeon光学编码复现：单完整N=3孔径；非完整网络复现',
            assumptions=['正文PDF第6–7页：0.5mm熔融石英基底、16可用级、100nm间距、6.22μm像元。',
              '深度d=-Δh，最近半级取较深侧；floor是向下取深度级。原文未公开舍入规则。',
              '模型使用相对高度；0.5mm绝对厚度产生的全局相位不影响无吸收单器件强度。',
              '单位辐射响应、满填充、无CFA/QE/吸收/Fresnel反射；不是相机电子数或实际瓦数。',
              '97×97只是约603.34μm局部窗口；η为该窗口功率/孔径输入功率。',
              '像元C3及圆域能量按中心离散归属，圆边界像元未作部分面积裁剪。',
              '数学x右y上、C3逆时针为正；正文顺时针与观察面约定仍未确定。'])
        if saved:
            source_paths = ['main_stage02c.py', 'config_stage02c.json', 'main_stage02b.py',
                'optics/fabrication_detector.py', 'optics/doe.py', 'optics/materials.py',
                'optics/propagation.py', 'optics/coordinates.py', 'optics/psf_analysis.py',
                'optics/convergence.py', 'optics/stage02_runtime.py', 'optics/runutil.py']
            write_json(dest/'source_manifest.json', {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in source_paths})
            text = ['# 阶段02C实际运行报告', '', '本批为Jeon单完整N=3孔径光学编码复现，不含网络、场景或双孔径。', '',
                '状态：计算交付完成；求积筛查：'+('通过' if report['numerical_convergence_passed'] else '未通过，需加密求积')+'。', '',
                '参数和公式出处：本地正文main.pdf第5页式(10)–(12)固定高度，第6–7页制造与相机参数；',
                '相位为2π[n(λ)−1]Δh/λ；由既有Fresnel复振幅积分传播，再计算I=|u|²。',
                '像元功率Pjk≈Σab I(xja,ykb)(p/q)²，平均强度=Pjk/p²。', '',
                '## 制造误差', '', '|器件|最大高度误差/nm|高度RMS/nm|实际使用级数|', '|---|---:|---:|---:|']
            for key, value in fabrication.items():
                text.append('|%s|%.5f|%.5f|%s|' % (key,value['max_abs_error_nm'],value['rms_error_nm'],
                    len(value['occupied_codes']) if value['occupied_codes'] is not None else '连续'))
            text += ['', '## 求积密度比较', '', '|器件|波长/nm|像元L1差异/%|η绝对差|通过|', '|---|---:|---:|---:|---|']
            for r in convergences: text.append('|%s|%d|%.6f|%.3g|%s|' %
                (r['device'],r['wavelength_nm'],100*r['pixel_l1'],r['eta_absolute'],r['passed']))
            text += ['', '## 假设与限制', ''] + ['- '+s for s in report['assumptions']]
            text += ['', '环境与全部指标见metrics/validation.json；简表见metrics/summary.csv。',
                'arrays保留3张固定高度、18组积分节点复场和9组中心点近似复场；figures保留两张结果图。',
                '默认无参数CPU运行；--no-save不落盘；--show-plots尝试弹窗。实际PyCharm点击与真实弹窗尚未人工验证。']
            (dest/'report_stage02c.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
            with (dest/'metrics'/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['device','wavelength_nm','q','eta_window','A3','angle_deg','R50_um','R80_um'])
                for r in rows.values(): writer.writerow([r['device'], r['wavelength_nm'], r['q'], r['eta_window'],
                    r['rotation_bands']['primary']['A3'], r['rotation_bands']['primary']['alpha_wrapped_deg'],
                    r['R50']['radius_um'], r['R80']['radius_um']])
            # 最终完成标记最后写入；CSV写失败时不能留下已完成的验证报告。
            write_json(dest/'metrics'/'validation.json', report)
        print(json.dumps({'status': report['status'], 'numerical_convergence_passed': report['numerical_convergence_passed'],
                          'run_dir': str(dest) if dest else None, 'elapsed_s': report['elapsed_s']}, ensure_ascii=False))
        if runtime['show_plots']: plt.show()
        plt.close(fig); plt.close(fig2)
        return report
    except Exception:
        if saved: write_json(dest/'failed.json', {'status':'failed', 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'config_stage02c.json'))
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_absolute(): path = ROOT/path
    report = run(json.loads(path.read_text(encoding='utf-8')), args.no_save, args.show_plots)
    return 0 if report['numerical_convergence_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
