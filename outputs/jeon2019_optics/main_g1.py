# -*- coding: utf-8 -*-
"""G1：固定N=3连续DOE的25波段PSF及全波段采样筛查；PyCharm直接运行。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from optics.g1_forward import (validate_config, profile_for, measure, compare, native_parseval,
                               rotation_summary, g0_physics_check)
from optics.stage02_runtime import effective_runtime, configure_plotting, resolve_project_path
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir, setup_logger

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def plot_results(plt, bank, edges, metrics, rotations, comparisons, c):
    figures = []
    extent = [edges[0]*1e6, edges[-1]*1e6]*2
    fig, axes = plt.subplots(5, 5, figsize=(13, 13))
    for k, ax in enumerate(axes.flat):
        image = bank[k]
        ax.imshow(image/image.max(), origin='lower', extent=extent, cmap='inferno')
        ax.set_title(f"{c['wavelengths_nm'][k]} nm | η={image.sum():.3f}")
        ax.set_xlabel('x (μm)'); ax.set_ylabel('y (μm)')
    fig.suptitle('固定连续DOE的25波段像元积分PSF；各幅峰值归一化仅用于形状显示')
    fig.tight_layout(); figures.append(('psf25_shapes.png', fig))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    wave = c['wavelengths_nm']
    for band in ['primary', 'sensitivity']:
        axes[0,0].plot(wave, rotations[band]['alpha_unwrapped_deg'], '.-', label=band)
        axes[0,1].plot(wave, rotations[band]['A3'], '.-', label=band)
    axes[0,0].set_ylabel('C3展开角（度；逆时针为正）'); axes[0,0].legend()
    axes[0,1].set_ylabel('角向可靠性A3'); axes[0,1].axhline(.05, ls='--', c='gray'); axes[0,1].legend()
    axes[1,0].plot(wave, [m['eta_window'] for m in metrics], '.-')
    axes[1,0].set_ylabel('有限窗口捕获功率/Pin')
    for key in ['R50', 'R80']:
        axes[1,1].plot(wave, [m[key]['radius_um'] for m in metrics], '.-', label=key)
    axes[1,1].set_ylabel('绝对包围能量半径（μm）'); axes[1,1].legend()
    for ax in axes.flat: ax.set_xlabel('波长（nm）'); ax.grid(alpha=.3)
    fig.tight_layout(); figures.append(('rotation_energy_size.png', fig))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in ['quadrature', 'input_refinement']:
        rows = [r for r in comparisons if r['method'] == method]
        axes[0].plot(wave, [100*r['kernel_l1'] for r in rows], '.-', label=method)
        axes[1].plot(wave, [r['eta_absolute'] for r in rows], '.-', label=method)
    axes[0].axhline(100*c['thresholds']['pixel_l1'], ls='--', c='gray'); axes[0].set_ylabel('成像核L1差异（%）')
    axes[1].axhline(c['thresholds']['eta_absolute'], ls='--', c='gray'); axes[1].set_ylabel('捕获比例绝对差')
    for ax in axes: ax.set_xlabel('波长（nm）'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); figures.append(('sampling25.png', fig))
    # 统一绝对核尺度显示九个图3波长，避免逐幅归一化掩盖通光差异。
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for k, ax in zip(range(0,25,3), axes.flat):
        im = ax.imshow(bank[k], origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=bank.max())
        ax.set_title(f"{wave[k]} nm"); ax.set_xlabel('x (μm)'); ax.set_ylabel('y (μm)')
    fig.subplots_adjust(wspace=.4, hspace=.4, right=.86)
    fig.colorbar(im, cax=fig.add_axes([.89,.15,.02,.7]), label='像元功率/Pin；共同色标')
    figures.append(('figure3_nine_absolute.png', fig))
    return figures


def run(c, no_save=False, show_plots=False):
    validate_config(c)
    runtime = effective_runtime(c, no_save, show_plots)
    plt, backend = configure_plotting(runtime['show_plots'])
    dest = allocate_unique_run_dir('g1') if runtime['save_results'] else None
    logger = setup_logger(dest)
    start = time.perf_counter(); figures = []
    try:
        guard = g0_physics_check(resolve_project_path(c['g0_manifest'], ROOT), ROOT.parents[1])
        profile, fine = profile_for(c), profile_for(c, True)
        fingerprint, fine_fingerprint = profile.compute_fingerprint(), fine.compute_fingerprint()
        waves = np.array([v/1e9 for v in c['wavelengths_nm']])
        d = c['detector']
        source = resolve_project_path(c['legacy_source'], ROOT)
        # 只读三波段旧光学源，重新设计的主网格须与其高度逐元素一致。
        with np.load(source/'arrays/height_continuous.npz', allow_pickle=False) as z:
            if not (np.array_equal(z['delta_h'], profile.delta_h) and np.array_equal(z['mask'], profile.mask)
                    and np.array_equal(z['x_m'], profile.grid.x.coords) and np.array_equal(z['y_m'], profile.grid.y.coords)
                    and str(z['fingerprint']) == fingerprint):
                raise ValueError('G1主器件与02C固定连续高度不同；本批不允许悄悄换基线')
        bank, coarse_bank, fine_bank, metrics, comparisons, energy, anchors = [], [], [], [], [], [], []
        if dest:
            write_json(dest/'config_effective.json', dict(c, runtime=runtime))
            write_json(dest/'progress_initial.json', dict(status='in_progress', expected_bands=25))
            np.savez_compressed(dest/'arrays/height_fixed.npz', delta_h_m=profile.delta_h, mask=profile.mask,
                lambda_design_m=profile.lambda_design, x_m=profile.grid.x.coords, y_m=profile.grid.y.coords,
                spacing_m=profile.grid.dx, fingerprint=fingerprint, params_json=json.dumps(profile.params),
                fine_fingerprint=fine_fingerprint)
        for k, lam in enumerate(waves):
            if time.perf_counter()-start > c['budget_seconds']:
                raise TimeoutError('达到预算，保留已有结果；没有G1完成标记')
            a = measure(profile, float(lam), c, d['quadrature_main'])
            b = measure(profile, float(lam), c, d['quadrature_coarse'])
            f = measure(fine, float(lam), c, d['quadrature_main'])
            bank.append(a['kernel']); coarse_bank.append(b['kernel']); fine_bank.append(f['kernel'])
            metrics.append(a['metrics'])
            for method, left, right in [('quadrature', b, a), ('input_refinement', a, f)]:
                comparisons.append(dict(method=method, wavelength_nm=c['wavelengths_nm'][k], **compare(left, right, c)))
            energy.append(dict(wavelength_nm=c['wavelengths_nm'][k], **native_parseval(a['u1'], profile.grid, float(lam), c['optical']['distance_m'])))
            if c['wavelengths_nm'][k] in [420,540,660]:
                with np.load(source/f"arrays/continuous_{c['wavelengths_nm'][k]}nm_q8.npz", allow_pickle=False) as z:
                    passed = (d['quadrature_main'] == 8 and np.array_equal(z['pixel_power'], a['power'])
                              and np.array_equal(z['pixel_x_m'], a['camera'].x.coords)
                              and np.array_equal(z['pixel_y_m'], a['camera'].y.coords)
                              and str(z['fingerprint']) == fingerprint)
                anchors.append(dict(wavelength_nm=c['wavelengths_nm'][k], exact_equal=bool(passed)))
            if profile.compute_fingerprint() != fingerprint or fine.compute_fingerprint() != fine_fingerprint:
                raise RuntimeError('多波长计算期间固定高度变化')
            if dest:
                np.savez_compressed(dest/f"arrays/field_{c['wavelengths_nm'][k]}nm.npz", u2_complex=a['u2'],
                    intensity_raw=a['intensity'], pixel_power=a['power'], kernel=a['kernel'], Pin=a['metrics']['pin'],
                    node_x_m=a['nodes'].x.coords, node_y_m=a['nodes'].y.coords, wavelength_m=lam,
                    fingerprint=fingerprint, q=d['quadrature_main'], distance_m=c['optical']['distance_m'])
                write_json(dest/f'metrics/progress_{k:02d}.json', dict(completed_bands=k+1, elapsed_s=time.perf_counter()-start))
            logger.info('G1 %d/25：%dnm，η=%.6f，求积/输入筛查=%s/%s', k+1, c['wavelengths_nm'][k],
                        a['kernel'].sum(), comparisons[-2]['passed'], comparisons[-1]['passed'])
        bank, coarse_bank, fine_bank = map(np.stack, [bank, coarse_bank, fine_bank])
        rotations = rotation_summary(metrics, waves)
        passed = (all(r['passed'] for r in comparisons) and all(r['passed'] for r in energy)
                  and len(anchors) == 3 and all(r['exact_equal'] for r in anchors))
        figures = plot_results(plt, bank, a['edges'], metrics, rotations, comparisons, c)
        if dest:
            np.savez_compressed(dest/'arrays/psf_bank.npz', kernels=bank, kernels_q_coarse=coarse_bank,
                kernels_input_fine=fine_bank, wavelengths_m=waves, pixel_x_m=a['camera'].x.coords,
                pixel_y_m=a['camera'].y.coords, pixel_edges_m=a['edges'], pitch_m=d['pitch_m'],
                fingerprint=fingerprint, fine_fingerprint=fine_fingerprint,
                unit='pixel_power_over_input_aperture_power', axes='wavelength,y,x')
            for name, fig in figures: fig.savefig(dest/'figures'/name, dpi=c['plots']['dpi'])
            write_json(dest/'metrics/psf_metrics.json', metrics)
            write_json(dest/'metrics/convergence.json', comparisons)
            write_json(dest/'metrics/rotation.json', rotations)
            write_json(dest/'metrics/native_energy.json', energy)
            with (dest/'metrics/summary.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['wavelength_nm','eta_window','R50_um','R80_um','primary_angle_deg','primary_A3','quadrature_kernel_l1','input_kernel_l1'])
                for k,m in enumerate(metrics):
                    writer.writerow([c['wavelengths_nm'][k],m['eta_window'],m['R50']['radius_um'],m['R80']['radius_um'],
                        rotations['primary']['alpha_unwrapped_deg'][k],rotations['primary']['A3'][k],
                        comparisons[2*k]['kernel_l1'],comparisons[2*k+1]['kernel_l1']])
            # 重新读取复场并核对强度、像元积分与最终核，最后才写完成状态。
            with np.load(dest/'arrays/height_fixed.npz', allow_pickle=False) as z:
                if not (np.array_equal(z['delta_h_m'], profile.delta_h) and np.array_equal(z['mask'],profile.mask)
                        and np.array_equal(z['lambda_design_m'],profile.lambda_design)
                        and np.array_equal(z['x_m'],profile.grid.x.coords) and np.array_equal(z['y_m'],profile.grid.y.coords)
                        and str(z['fingerprint']) == fingerprint and float(z['spacing_m']) == profile.grid.dx):
                    raise RuntimeError('保存的固定高度或坐标不一致')
            for k,lam in enumerate(waves):
                with np.load(dest/f"arrays/field_{c['wavelengths_nm'][k]}nm.npz", allow_pickle=False) as z:
                    from optics.fabrication_detector import integrate_square_pixels
                    p, _ = integrate_square_pixels(z['intensity_raw'], d['pixels_per_axis'], d['pitch_m'], d['quadrature_main'])
                    if not (np.array_equal(np.abs(z['u2_complex'])**2, z['intensity_raw'])
                            and np.array_equal(p, z['pixel_power']) and np.array_equal(z['kernel'],bank[k])
                            and np.array_equal(p/float(z['Pin']),bank[k]) and float(z['wavelength_m']) == lam
                            and str(z['fingerprint']) == fingerprint
                            and np.array_equal(z['node_x_m'], a['nodes'].x.coords)
                            and np.array_equal(z['node_y_m'], a['nodes'].y.coords)
                            and int(z['q']) == d['quadrature_main']
                            and float(z['Pin']) == metrics[k]['pin']
                            and float(z['distance_m']) == c['optical']['distance_m']):
                        raise RuntimeError('落盘复场、波长、高度指纹或积分不一致')
            with np.load(dest/'arrays/psf_bank.npz', allow_pickle=False) as z:
                if not (np.array_equal(z['kernels'],bank) and np.array_equal(z['wavelengths_m'],waves)
                        and np.array_equal(z['pixel_x_m'], a['camera'].x.coords)
                        and np.array_equal(z['pixel_y_m'], a['camera'].y.coords)
                        and np.array_equal(z['pixel_edges_m'], a['edges'])
                        and np.array_equal(z['kernels_q_coarse'],coarse_bank)
                        and np.array_equal(z['kernels_input_fine'],fine_bank)):
                    raise RuntimeError('最终PSF库读取不一致')
            expected = {'height_fixed.npz','psf_bank.npz'} | {f'field_{nm}nm.npz' for nm in c['wavelengths_nm']}
            if {p.name for p in (dest/'arrays').glob('*.npz')} != expected:
                raise RuntimeError('原始数组交付集合不一致')
            names = ['main_g1.py','config_g1.json','optics/g1_forward.py','main_stage02c.py','main_stage02b.py']
            names += [str(p.relative_to(ROOT)).replace('\\','/') for p in sorted((ROOT/'optics').glob('*.py'))]
            write_json(dest/'source_manifest.json', {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(set(names))})
            text = f"# G1固定连续DOE的25波段PSF\n\n数值筛查通过：{passed}。单完整N=3孔径；同一高度；统一像元物理坐标。\n\n共75组传播（主q8、对照q4、二倍输入加密）；全25波段原生FFT能量核验。原始复场/强度和物理核已保存并重读。\n\n论文旋转方向对应仍未解决；paper_alignment_passed=false。本阶段没有RGB响应、光谱积分、网络或双孔径。\n\n核数组顺序[25,y,x]，每项为像元功率/入射孔径功率，核和为有限窗口捕获比例。逐幅峰值归一化仅用于形状图。\n\n更细输入是同一解析器件规则的空间加密，不是每波长重新设计。筛查容差为项目自定，不是严格误差界。完整物理说明见工程G1_README.md。\n"
            with (dest/'report_g1.md').open('x', encoding='utf-8') as stream: stream.write(text)
        if runtime['show_plots']: plt.show()
        if dest:
            required = [dest/'report_g1.md',dest/'source_manifest.json',dest/'metrics/summary.csv']
            required += [dest/'figures'/name for name,_ in figures]
            if not all(p.is_file() and p.stat().st_size > 0 for p in required):
                raise RuntimeError('必需报告、指标或图未交付')
        g0_physics_check(resolve_project_path(c['g0_manifest'], ROOT), ROOT.parents[1])
        report = dict(status='completed' if passed else 'numerical_checks_failed', numerical_validation_passed=bool(passed),
            paper_alignment_passed=False, paper_rotation_direction='正文为顺时针；与当前坐标对应未解决',
            g0_physics_check=guard, completed_bands=25, propagated_fields=75, readback_passed=True if dest else None,
            anchors=anchors, config=c, runtime=runtime, backend=backend, environment=environment_info(),
            height_fingerprint=fingerprint, fine_height_fingerprint=fine_fingerprint,
            elapsed_s=time.perf_counter()-start, scope='G1：Jeon连续光学编码25波段数值模型；非完整论文复现')
        if dest: write_json(dest/'metrics/validation.json', report)
        print('G1结果：'+str(dest) if dest else 'G1无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest: write_json(dest/'failed.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        for _, fig in figures: plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_g1.json')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    args = parser.parse_args()
    c = json.loads(resolve_project_path(args.config, ROOT).read_text(encoding='utf-8'))
    return 0 if run(c, args.no_save, args.show_plots)['numerical_validation_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
