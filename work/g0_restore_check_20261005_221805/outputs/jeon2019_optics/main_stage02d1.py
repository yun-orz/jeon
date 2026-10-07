# -*- coding: utf-8 -*-
"""02D-1：单通道非相干前向成像、裁剪和伴随验证；无参数可运行。"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback
import numpy as np
from optics.imaging import SpectralImager, direct_shift_sum, operator_checks
from optics.stage02c_source import load_source, tree_sha
from optics.stage02_runtime import effective_runtime, configure_plotting, resolve_project_path
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir, setup_logger

ROOT = Path(__file__).resolve().parent


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def validate_config(c):
    if not c['devices'] or len(set(c['devices'])) != len(c['devices']) or not set(c['devices']) <= {'continuous','nearest_depth','floor_depth'}:
        raise ValueError('器件必须来自02C的连续/量化固定高度且不可重复')
    if c['wavelengths_m'] != [420e-9,540e-9,660e-9] or c['response'] != [1.,1.,1.]:
        raise ValueError('本批限定三诊断波长、单位辐射响应')
    if c['pitch_m'] != 6.22e-6 or c['quadrature_per_axis'] != 8: raise ValueError('本批限定6.22μm、q8')
    if any(isinstance(n,bool) or not isinstance(n,int) or n<16 for n in c['scene_shape']) or len(c['scene_shape'])!=2:
        raise ValueError('图案要求场景为两个至少16的整数')
    if isinstance(c['seed'],bool) or not isinstance(c['seed'],int) or c['seed']<0: raise ValueError('随机种子须为非负整数')
    for v in c['thresholds'].values():
        if not np.isfinite(v) or v<=0: raise ValueError('检查容差必须有限且为正')
    # 在任何输出写入前检查裁剪合法性。
    SpectralImager(np.ones((3,97,97)),c['scene_shape'],c['pitch_m'],c['response'],c['crop'])


def scenes(shape):
    h,w = shape
    out = {}
    x = np.zeros((3,h,w)); x[:,h//2,w//2] = [1.,2.,.5]; out['coincident_points'] = x
    x = np.zeros_like(x); x[0,0,0] = 1.; x[1,h-1,w-1] = 2.; x[2,h//3,2*w//3] = .5
    out['separated_points'] = x
    x = np.zeros_like(x)
    x[0,h//4:h//4+2,w//8:7*w//8] = 1.
    x[1,h//8:7*h//8,w//2:w//2+2] = .8
    x[2,h//3:2*h//3,w//3:2*w//3] = .6
    out['lines_and_square'] = x
    return out


def relative_error(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))


def extent_um(x,y,pitch):
    return [(x[0]-pitch/2)*1e6,(x[-1]+pitch/2)*1e6,(y[0]-pitch/2)*1e6,(y[-1]+pitch/2)*1e6]


def run(c,no_save=False,show_plots=False):
    validate_config(c)
    runtime = effective_runtime(c,no_save,show_plots)
    plt,backend = configure_plotting(runtime['show_plots'])
    dest = allocate_unique_run_dir('stage02d1') if runtime['save_results'] else None
    logger = setup_logger(dest) if dest else logging.getLogger('stage02d1_memory')
    if not dest:
        logger.setLevel(logging.INFO)
        if not logger.handlers: logger.addHandler(logging.StreamHandler())
    start = time.perf_counter(); figures = []
    try:
        source = resolve_project_path(c['source_run'],ROOT)
        kernels,evidence = load_source(source,ROOT,c)
        logger.info('源run重验通过：三张固定高度、九组q8复场；源文件只读')
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime))
            write_json(dest/'source_evidence.json',evidence)
        checks,metrics,saved_arrays = [],[],{}
        patterns = scenes(c['scene_shape']); tol = c['thresholds']
        for device in c['devices']:
            kernel = kernels[device]
            full = SpectralImager(kernel,c['scene_shape'],c['pitch_m'],c['response'])
            cropped = SpectralImager(kernel,c['scene_shape'],c['pitch_m'],c['response'],c['crop'])
            for label,cube in patterns.items():
                if np.any(cube<0): raise ValueError('物理场景含负功率')
                contributions = full.per_band_full(cube)
                measurement = contributions.sum(axis=0)
                yc = cropped.forward(cube)
                direct = direct_shift_sum(cube,kernel,c['response'])
                expected_flux = float(np.dot(cube.sum(axis=(1,2)),np.asarray(c['response'])*kernel.sum(axis=(1,2))))
                flux_error = abs(float(measurement.sum())-expected_flux)/expected_flux
                direct_error = relative_error(measurement,direct)
                projection_error = relative_error(yc,measurement[cropped.slices])
                negative_min = float(measurement.min())
                # FFT舍入可出现约1e−18负数，不剪裁以免破坏线性/伴随；记录并检查尺度。
                tiny_negative_ok = negative_min >= -1e-13*float(measurement.max())
                record = dict(device=device,scene=label,input_band_power=cube.sum(axis=(1,2)).tolist(),
                    kernel_eta=kernel.sum(axis=(1,2)).tolist(),full_power=float(measurement.sum()),
                    expected_full_power=expected_flux,crop_power=float(yc.sum()),
                    crop_retained_fraction=float(yc.sum()/measurement.sum()),flux_relative_error=flux_error,
                    direct_relative_error=direct_error,crop_projection_error=projection_error,
                    minimum_fft_power=negative_min,tiny_negative_ok=tiny_negative_ok)
                metrics.append(record)
                checks.append(flux_error<=tol['flux_relative'] and direct_error<=tol['direct_relative']
                              and projection_error<=tol['direct_relative'] and tiny_negative_ok)
                for mode,op,y in [('full',full,measurement),('crop',cropped,yc)]:
                    diag = operator_checks(op,c['seed'],y)
                    diag.update(device=device,scene=label,mode=mode)
                    record[mode+'_operator_checks'] = diag
                    checks.append(diag['inner_product_relative_error']<=tol['adjoint_relative']
                                  and diag['gradient_relative_error']<=tol['gradient_relative'])
                coords = cropped.coordinates()
                values = dict(cube=cube,kernels=kernel,response=np.asarray(c['response']),
                    contributions_full=contributions,measurement_full=measurement,measurement_crop=yc,
                    wavelengths_m=np.asarray(c['wavelengths_m']),pitch_m=c['pitch_m'],crop_indices=np.asarray(cropped.crop),
                    fixed_height_fingerprint=evidence['fingerprints'][device],**coords)
                filename = device+'_'+label+'.npz'; saved_arrays[filename] = values
                if dest: np.savez_compressed(dest/'arrays'/filename,**values)
                logger.info('%s/%s：通量误差%.3g，直接平移叠加误差%.3g，crop保留%.4f',
                            device,label,flux_error,direct_error,record['crop_retained_fraction'])
            # 一张图展示三波段场景与编码功率，后两列共用原始测量色标。
            fig,axs = plt.subplots(3,5,figsize=(16,9)); figures.append(fig)
            coord = cropped.coordinates()
            scene_extent = extent_um(coord['scene_x_m'],coord['scene_y_m'],c['pitch_m'])
            full_extent = extent_um(coord['full_x_m'],coord['full_y_m'],c['pitch_m'])
            crop_extent = extent_um(coord['output_x_m'],coord['output_y_m'],c['pitch_m'])
            for row,(label,cube) in enumerate(patterns.items()):
                v = saved_arrays[device+'_'+label+'.npz']
                for b in range(3):
                    im = axs[row,b].imshow(cube[b],origin='lower',extent=scene_extent,cmap='viridis',vmin=0,vmax=cube.max())
                    axs[row,b].set_title('%s | %d nm' % (label,round(c['wavelengths_m'][b]*1e9)))
                    fig.colorbar(im,ax=axs[row,b],fraction=.046,pad=.04)
                for col,name,ex in [(3,'measurement_full',full_extent),(4,'measurement_crop',crop_extent)]:
                    im = axs[row,col].imshow(v[name],origin='lower',extent=ex,cmap='inferno',vmin=0,vmax=v['measurement_full'].max())
                    axs[row,col].set_title('full编码功率' if col==3 else '裁剪编码功率')
                    fig.colorbar(im,ax=axs[row,col],fraction=.046,pad=.04)
                for ax in axs[row]: ax.set_xlabel('x (μm)'); ax.set_ylabel('y (μm)')
            fig.suptitle(device+'：原始相对功率；三波段场景共用每行色标，full/crop测量共用每行色标')
            fig.tight_layout()
            if dest: fig.savefig(dest/'figures'/(device+'_encoding.png'),dpi=c['plots']['dpi'])
        if tree_sha(source) != evidence['sha256']: raise RuntimeError('执行期间源run文件发生变化')
        readback = []
        if dest:
            if sorted(p.name for p in (dest/'arrays').glob('*.npz')) != sorted(saved_arrays): raise RuntimeError('场景交付集合错误')
            for filename,values in saved_arrays.items():
                with np.load(dest/'arrays'/filename,allow_pickle=False) as z:
                    if set(z.files)!=set(values) or not all(np.array_equal(z[k],v) for k,v in values.items()):
                        raise RuntimeError('保存数组内容不匹配：'+filename)
                    op = SpectralImager(z['kernels'],c['scene_shape'],float(z['pitch_m']),z['response'],z['crop_indices'].tolist())
                    if not np.array_equal(op.forward(z['cube']),z['measurement_crop']): raise RuntimeError('落盘测量重算不一致')
                    readback.append(filename)
            for device in c['devices']:
                if not (dest/'figures'/(device+'_encoding.png')).is_file(): raise RuntimeError('结果图缺失')
        passed = bool(all(checks))
        report = dict(status='completed',validation_passed=passed,checks_count=len(checks),
            config=c,runtime=runtime,source_unchanged=True,source_validated_fields=9,
            metrics=metrics,readback=readback,backend=backend,environment=environment_info(),
            elapsed_s=time.perf_counter()-start,
            scope='Jeon光学编码复现：02D-1单通道非相干成像/伴随；未运行基础重建或原网络',
            assumptions=['理想像平面点源功率格点，平移不变卷积；无物距/放大率或离轴像差模型。',
                '三离散单色诊断波长；不是宽波段积分或每nm谱密度。',
                'PSF核=像元功率/Pin，不重复乘像元面积、不将通量强制归一到1。',
                '满填充、单位辐射响应；无CFA/QE、光子电子换算或噪声。',
                'full保留所选有限PSF核的通量；核外衍射尾部仍截断。',
                'FFT机器舍入级负数不剪裁，以保持线性/伴随；它们不是物理负功率。',
                '正文旋转方向对应仍未确定；未镜像PSF。真实PyCharm点击/图窗尚未人工验证。'])
        if dest:
            sources = ['main_stage02d1.py','config_stage02d1.json','optics/imaging.py','optics/stage02c_source.py',
                       'optics/stage02_runtime.py','optics/runutil.py','main_stage02b.py']
            # 仅对本批源码清单计算指纹，避免扫描整个历史结果目录。
            write_json(dest/'source_manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
            lines = ['# 阶段02D-1实际运行报告','','本批验证单通道非相干成像及伴随；尚未执行基础重建。',
                '验证状态：'+str(passed)+'；源run只读且三高度/九组q8场重验通过。','',
                '模型：y=CΣ_b R_b(X_b*K_b)，K_b=Ppixel_b/Pin_b；非相干按功率相加。',
                'full零延拓线性卷积，伴随先裁剪零回填再与翻转核作valid卷积。','',
                '|器件|场景|full通量|crop保留比例|通量相对误差|直接叠加相对误差|',
                '|---|---|---:|---:|---:|---:|']
            for r in metrics: lines.append('|%s|%s|%.6g|%.6f|%.3g|%.3g|' %
                (r['device'],r['scene'],r['full_power'],r['crop_retained_fraction'],r['flux_relative_error'],r['direct_relative_error']))
            lines += ['','## 假设与未验证项','']+['- '+s for s in report['assumptions']]
            lines += ['','详细内积/梯度检查和环境见metrics/validation.json。原始数组在arrays，结果图在figures。']
            (dest/'report_stage02d1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            # 完成标记最后写，图/报告/数组缺失或写入失败不会标completed。
            write_json(dest/'metrics'/'validation.json',report)
        print(json.dumps(dict(status='completed',validation_passed=passed,run_dir=str(dest) if dest else None,
                              elapsed_s=report['elapsed_s']),ensure_ascii=False))
        if runtime['show_plots']: plt.show()
        return report
    except Exception:
        if dest: write_json(dest/'failed.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figures: plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_stage02d1.json')
    parser.add_argument('--no-save',action='store_true')
    parser.add_argument('--show-plots',action='store_true')
    args = parser.parse_args()
    cfg = json.loads(resolve_project_path(args.config,ROOT).read_text(encoding='utf-8'))
    report = run(cfg,args.no_save,args.show_plots)
    return 0 if report['validation_passed'] else 2


if __name__=='__main__': raise SystemExit(main())
