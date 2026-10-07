# -*- coding: utf-8 -*-
"""同一CW高度的LightPipes传播交叉检查，不宣称作者原脚本复现。"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from main_g1 import write_json
from main_stage02b import allocate_unique_run_dir
from optics.coordinates import make_grid
from optics.doe import DOEHeightProfile, compute_doe_transmission_field
from optics.fabrication_detector import detector_grids, integrate_square_pixels
from optics.propagation import fresnel_kernel_separable
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def difference(a,b):
    """保留幅度/通光比例，不缩放预测图以迎合参考。"""
    return dict(relative_l1=float(np.sum(np.abs(a-b))/np.sum(np.abs(b))),
                relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                max_absolute=float(np.max(np.abs(a-b))))


def run(c,no_save=False,show_plots=False):
    start=time.perf_counter()
    if c['wavelengths_nm']!=[420,540,660] or c['padding_sizes']!=[1536,2048] or c['fresnel_padding_size']!=1536:
        raise ValueError('本批固定三控制波长、两种零填充，不搜索参数')
    threshold=c['comparison_relative_threshold']
    if isinstance(threshold,bool) or not np.isfinite(threshold) or not 0<threshold<1:
        raise ValueError('比较阈值非法')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']) or type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:
        raise ValueError('运行或图片配置非法')
    dependency=resolve_project_path(c['dependency_directory'],ROOT)
    if dependency.is_dir():sys.path.insert(0,str(dependency))
    import LightPipes as lp
    if lp.__version__!='2.1.5':raise ValueError('本批仅验证LightPipes 2.1.5')
    package=Path(lp.__file__).resolve().parent;package_before=tree_sha(package)
    source=resolve_project_path(c['source_window'],ROOT);alignment=resolve_project_path(c['source_alignment'],ROOT)
    source_before=tree_sha(source);alignment_before=tree_sha(alignment)
    audit=json.loads(resolve_project_path(c['source_audit'],ROOT).read_text(encoding='utf-8'))
    if not audit['independent_audit']['passed'] or source_before!=audit['run_sha256']:
        raise ValueError('窗口来源未审核或改变')
    for name,sha in audit['sha256'].items():
        if file_sha(Path(name))!=sha:raise ValueError('窗口源码锁改变')
    code=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if any(file_sha(ROOT/n)!=sha for n,sha in code.items()):raise ValueError('物理源码改变')
    with np.load(alignment/'arrays/height_cw_fixed.npz',allow_pickle=False) as z:
        grid=make_grid(len(z['x_m']),1e-6)
        profile=DOEHeightProfile(grid,z['delta_h_m'].copy(),z['mask'].copy(),z['lambda_design_m'].copy(),
                                 'jeon2019_clockwise_diagnostic',json.loads(str(z['params_json'])))
        expected=str(z['fingerprint'])
        if not np.array_equal(grid.x.coords,z['x_m']) or profile.compute_fingerprint()!=expected:
            raise ValueError('固定CW高度指纹或坐标改变')
    with np.load(source/'arrays/window_bank.npz',allow_pickle=False) as z:
        oldkernels=z['old_kernels'].copy();oldwaves=z['wavelengths_m'].copy()
    native_axis=np.arange(-304,305)*1e-6
    camera,nodes,edges=detector_grids(97,6.22e-6,8)
    points=np.column_stack((np.broadcast_to(nodes.y.coords[:,None],(len(nodes.y.coords),len(nodes.x.coords))).ravel(),
                            np.broadcast_to(nodes.x.coords[None,:],(len(nodes.y.coords),len(nodes.x.coords))).ravel()))
    arrays=dict(native_axis_m=native_axis,detector_centers_m=camera.x.coords,height_fingerprint=expected)
    rows=[];valid=True
    for wave in c['wavelengths_nm']:
        wavelength=wave*1e-9;u1=compute_doe_transmission_field(profile,wavelength)
        pin=float(np.sum(np.abs(u1)**2)*grid.cell_area)
        reference=fresnel_kernel_separable(u1,grid,wavelength,.05,native_axis,native_axis)
        index=int(np.argmin(np.abs(oldwaves-wavelength)));detector_reference=oldkernels[index]
        arrays[f'u1_{wave}']=u1;arrays[f'reference_native_{wave}']=reference;arrays[f'reference_kernel_{wave}']=detector_reference
        results={}
        variants=[('Forvard',n) for n in c['padding_sizes']]+[('Fresnel',c['fresnel_padding_size'])]
        for method,n in variants:
            print(f'LightPipes {method}，{wave}nm，N={n}，固定输入步长1µm',flush=True)
            field=lp.Begin(n*grid.dx,wavelength,n);field.field=np.zeros((n,n),dtype=np.complex128)
            offset=n//2-grid.x.n//2;field.field[offset:offset+grid.x.n,offset:offset+grid.x.n]=u1
            if not np.array_equal(field.xvalues[offset:offset+grid.x.n],grid.x.coords):raise ValueError('零填充物理坐标不匹配')
            now=time.perf_counter();out=getattr(lp,method)(field,.05)
            output=out.field;n0=n//2
            crop=output[n0-304:n0+305,n0-304:n0+305].copy()
            intensity=np.abs(crop)**2
            interpolator=RegularGridInterpolator((native_axis,native_axis),intensity,bounds_error=True)
            samples=interpolator(points).reshape(len(nodes.y.coords),len(nodes.x.coords))
            power,_=integrate_square_pixels(samples,97,6.22e-6,8);kernel=power/pin
            label=f'{method}_{n}';results[label]=kernel
            full_ratio=float(np.sum(np.abs(output)**2)*grid.cell_area/pin)
            # 全局相位常数约定不同，强度比较无需调整；复场仅另报最佳单位相位对齐误差。
            overlap=np.vdot(crop,reference);phase=overlap/abs(overlap) if abs(overlap)>0 else 1.
            row=dict(wavelength_nm=wave,method=method,n=n,declared_step_m=float(out.dx),
                     fresnel_legacy_internal_step_m=n*grid.dx/(n-1) if method=='Fresnel' else None,
                     native_intensity_difference=difference(intensity,np.abs(reference)**2),
                     detector_kernel_difference=difference(kernel,detector_reference),
                     complex_unit_phase_aligned_relative_l2=float(np.linalg.norm(crop*phase-reference)/np.linalg.norm(reference)),
                     unit_phase_alignment_only=True,full_grid_power_over_input=full_ratio,detector_efficiency=float(kernel.sum()),
                     input_power=pin,elapsed_seconds=time.perf_counter()-now)
            rows.append(row);arrays[f'{label}_native_{wave}']=crop;arrays[f'{label}_kernel_{wave}']=kernel
            valid=valid and np.isfinite(output).all() and np.isfinite(kernel).all() and (kernel>=0).all() and 0<float(kernel.sum())<=1.02
            del field,out,output
        arrays[f'padding_kernel_change_{wave}']=results['Forvard_1536']-results['Forvard_2048']
    padding=[dict(wavelength_nm=w,**difference(arrays[f'Forvard_1536_kernel_{w}'],arrays[f'Forvard_2048_kernel_{w}'])) for w in c['wavelengths_nm']]
    # 检查与参考是否接近和程序数值健康分开；不为了通过而改阈值。
    agreements=[dict(wavelength_nm=r['wavelength_nm'],method=r['method'],n=r['n'],
                     below_threshold=r['detector_kernel_difference']['relative_l1']<threshold and r['detector_kernel_difference']['relative_l2']<threshold) for r in rows]
    summary=dict(lightpipes_version=lp.__version__,fixed_height_fingerprint=expected,rows=[{k:v for k,v in r.items() if k!='elapsed_seconds'} for r in rows],
                 padding_change=padding,agreement=agreements,numerical_health_passed=bool(valid),comparison_threshold=threshold,paper_alignment_passed=False,
                 author_2019_version_and_propagator_known=False,detector_resampling='linear intensity interpolation at common q8 midpoint nodes',
                 interpolation_error_separately_bounded=False,coordinate_handling='array row mapped to project y increasing; display origin=lower; no flip of PSF')
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g1_lightpipes') if runtime['save_results'] else None
    try:
        fig,axes=plt.subplots(3,4,figsize=(12,9))
        labels=['reference','Forvard_1536','Forvard_2048','Fresnel_1536'];extent=[edges[0]*1e6,edges[-1]*1e6]*2
        for i,w in enumerate(c['wavelengths_nm']):
            for j,label in enumerate(labels):
                kernel=arrays[f'reference_kernel_{w}'] if label=='reference' else arrays[f'{label}_kernel_{w}']
                axes[i,j].imshow(kernel/kernel.max(),origin='lower',extent=extent,cmap='inferno');axes[i,j].set_title(f'{label} {w}nm',fontsize=9)
        fig.suptitle('同一固定CW高度：形状图峰值归一化，原核不缩放');fig.tight_layout();figures.append(('lightpipes_comparison.png',fig))
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));np.savez_compressed(dest/'arrays/crosscheck.npz',**arrays)
            write_json(dest/'source_evidence.json',dict(window=dict(path=str(source),sha256=source_before),alignment=dict(path=str(alignment),sha256=alignment_before),
                                                      lightpipes_package=dict(path=str(package),sha256=package_before)))
            names=sorted(set([*code,'main_g1_lightpipes.py','config_g1_lightpipes.json','G1_LIGHTPIPES_README.md','requirements_lightpipes.txt']))
            write_json(dest/'source_manifest.json',{name:file_sha(ROOT/name) for name in names})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            with np.load(dest/'arrays/crosscheck.npz',allow_pickle=False) as z:
                if any(not np.array_equal(z[k],v) for k,v in arrays.items()):raise RuntimeError('数组保存重载不一致')
        if tree_sha(source)!=source_before or tree_sha(alignment)!=alignment_before or tree_sha(package)!=package_before:
            raise RuntimeError('冻结来源或独立依赖改变')
        if dest:write_json(dest/'metrics/validation.json',dict(summary=summary,timings=rows,environment=environment_info(),backend=backend,elapsed_seconds=time.perf_counter()-start))
        if runtime['show_plots']:plt.show()
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2));print('交叉核对结果：'+str(dest) if dest else '交叉核对无保存：未写结果文件')
        return summary
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g1_lightpipes.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['numerical_health_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
