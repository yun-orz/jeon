# -*- coding: utf-8 -*-
"""扩大Forvard填充并分离Fresnel内部步长和核单元平均，不训练。"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from main_g1 import write_json
from main_g1_lightpipes import difference
from main_stage02b import allocate_unique_run_dir
from optics.g1_lp_sampling import cell_averaged_field
from optics.fabrication_detector import detector_grids,integrate_square_pixels
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def run(c,no_save=False,show_plots=False):
    start=time.perf_counter()
    if c['padding_sizes']!=[3072,4096] or c['fresnel_padding_sizes']!=[1536,2048] or c['wavelengths_nm']!=[420,540,660]:raise ValueError('本批固定两种扩大域、两种卷积域和三个控制波长')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']) or type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('运行参数非法')
    for k in ['comparison_relative_threshold','cell_model_relative_threshold']:
        if isinstance(c[k],bool) or not np.isfinite(c[k]) or not 0<c[k]<1:raise ValueError('容差须有限且介于0与1')
    source=resolve_project_path(c['source_run'],ROOT);before=tree_sha(source)
    audit=json.loads(resolve_project_path(c['source_audit'],ROOT).read_text(encoding='utf-8'))
    if not audit['independent_audit']['passed'] or before!=audit['run_sha256']:raise ValueError('来源改变或未审核')
    for path,sha in audit['sha256'].items():
        if file_sha(Path(path))!=sha:raise ValueError('冻结源码或说明改变')
    oldcodes=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if any(file_sha(ROOT/n)!=sha for n,sha in oldcodes.items()):raise ValueError('物理源码改变')
    evidence=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    for item in evidence.values():
        if tree_sha(Path(item['path']))!=item['sha256']:raise ValueError('原始来源或独立库改变')
    dependency=resolve_project_path(c['dependency_directory'],ROOT)
    if dependency.is_dir():sys.path.insert(0,str(dependency))
    import LightPipes as lp
    if lp.__version__!='2.1.5' or Path(lp.__file__).resolve().parent!=Path(evidence['lightpipes_package']['path']):raise ValueError('独立库版本或位置不同')
    with np.load(source/'arrays/crosscheck.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    axis=data['native_axis_m'];input_axis=(np.arange(1101)-550)*1e-6;dx=1e-6
    _,nodes,_=detector_grids(97,6.22e-6,8)
    yy,xx=np.meshgrid(nodes.y.coords,nodes.x.coords,indexing='ij');points=np.column_stack((yy.ravel(),xx.ravel()))
    def camera_kernel(field,pin):
        samples=RegularGridInterpolator((axis,axis),np.abs(field)**2,bounds_error=True)(points).reshape(len(nodes.y.coords),len(nodes.x.coords))
        return integrate_square_pixels(samples,97,6.22e-6,8)[0]/pin
    results=dict(native_axis_m=axis,input_axis_m=input_axis,height_fingerprint=data['height_fingerprint']);rows=[];changes=[];cell_errors=[];valid=True
    for wave in c['wavelengths_nm']:
        wavelength=wave*1e-9;u1=data[f'u1_{wave}'];pin=float(np.sum(np.abs(u1)**2)*dx**2)
        reference=data[f'reference_kernel_{wave}'];native_reference=data[f'reference_native_{wave}']
        results[f'u1_{wave}']=u1;results[f'reference_kernel_{wave}']=reference
        results[f'reference_native_{wave}']=native_reference
        for n in [1536,2048]:results[f'Forvard_{n}_kernel_{wave}']=data[f'Forvard_{n}_kernel_{wave}']
        results[f'Fresnel_nominal_native_{wave}']=data[f'Fresnel_1536_native_{wave}']
        results[f'Fresnel_nominal_kernel_{wave}']=data[f'Fresnel_1536_kernel_{wave}']
        for method,n,adapted in [('Forvard',3072,False),('Forvard',4096,False),('Fresnel',1536,True),('Fresnel',2048,True)]:
            label=f'{method}_{n}' if not adapted else f'Fresnel_adapted_{n}'
            print(f'{wave}nm：{label}，固定DOE场，未更换阈值',flush=True)
            size=(n-1)*dx if adapted else n*dx
            field=lp.Begin(size,wavelength,n);field.field=np.zeros((n,n),complex)
            offset=n//2-550;field.field[offset:offset+1101,offset:offset+1101]=u1
            now=time.perf_counter();out=getattr(lp,method)(field,.05)
            output=out.field;crop=output[n//2-304:n//2+305,n//2-304:n//2+305].copy()
            kernel=camera_kernel(crop,pin)
            row=dict(wavelength_nm=wave,label=label,n=n,nominal_step_m=float(out.dx),effective_step_m=size/(n-1) if adapted else size/n,
                     interface_size_adapted=adapted,native_intensity_difference=difference(abs(crop)**2,abs(native_reference)**2),
                     detector_kernel_difference=difference(kernel,reference),full_grid_power_over_input=float(np.sum(abs(output)**2)*dx**2/pin),
                     physical_area_step_used_m=dx,detector_efficiency=float(kernel.sum()),elapsed_seconds=time.perf_counter()-now)
            rows.append(row);results[f'{label}_native_{wave}']=crop;results[f'{label}_kernel_{wave}']=kernel
            valid=valid and np.isfinite(crop).all() and np.isfinite(kernel).all() and 0<float(kernel.sum())<=1.02
            del field,out,output
        print(f'{wave}nm：独立Fresnel核单元平均积分',flush=True)
        cell=cell_averaged_field(u1,input_axis,axis,dx,wavelength,.05)
        for n in c['fresnel_padding_sizes']:
            adapted=results[f'Fresnel_adapted_{n}_native_{wave}'];overlap=np.vdot(cell,adapted);phase=overlap/abs(overlap)
            err=float(np.linalg.norm(cell*phase-adapted)/np.linalg.norm(adapted))
            cell_errors.append(dict(wavelength_nm=wave,n=n,kernel_support_complete=n==2048,complex_unit_phase_aligned_relative_l2=err,native_intensity_difference=difference(abs(cell)**2,abs(adapted)**2)))
        results[f'cell_averaged_native_{wave}']=cell;results[f'cell_averaged_kernel_{wave}']=camera_kernel(cell,pin)
        for a,b in [(1536,2048),(2048,3072),(3072,4096)]:
            changes.append(dict(wavelength_nm=wave,from_n=a,to_n=b,**difference(results[f'Forvard_{a}_kernel_{wave}'],results[f'Forvard_{b}_kernel_{wave}'])))
    core_rows=[{k:v for k,v in r.items() if k!='elapsed_seconds'} for r in rows]
    last=[r for r in changes if r['from_n']==3072];threshold=c['comparison_relative_threshold']
    summary=dict(numerical_health_passed=bool(valid),rows=core_rows,padding_changes=changes,cell_model_checks=cell_errors,
                 last_padding_pair_below_threshold=all(r['relative_l1']<threshold and r['relative_l2']<threshold for r in last),
                 forvard_4096_reference_below_threshold=all(r['detector_kernel_difference']['relative_l1']<threshold and r['detector_kernel_difference']['relative_l2']<threshold for r in rows if r['label']=='Forvard_4096'),
                 cell_model_verified=all(r['complex_unit_phase_aligned_relative_l2']<c['cell_model_relative_threshold'] for r in cell_errors if r['kernel_support_complete']),
                 comparison_threshold=threshold,cell_model_threshold=c['cell_model_relative_threshold'],
                 fixed_height_fingerprint=str(data['height_fingerprint']),lightpipes_version=lp.__version__,
                 paper_alignment_passed=False,author_2019_implementation_known=False,infinite_domain_convergence_proved=False,
                 interpolation_error_status='same empirical interpolation diagnostic as prior stage; not a rigorous bound',
                 adaptation_status='Fresnel size=(N-1)*dx matches internal legacy dx; output nominal metadata differs and is not used as physical axis')
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g1_lp_sampling') if runtime['save_results'] else None
    try:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for wave in c['wavelengths_nm']:
            old=[dict(label=f'Forvard_{n}',detector_kernel_difference=difference(results[f'Forvard_{n}_kernel_{wave}'],results[f'reference_kernel_{wave}'])) for n in [1536,2048]]
            values=old+[r for r in core_rows if r['wavelength_nm']==wave and r['label'].startswith('Forvard')]
            axes[0].plot([1536,2048,3072,4096],[r['detector_kernel_difference']['relative_l1']*100 for r in values],'o-',label=f'{wave}nm')
            labels=['Fresnel_nominal','Fresnel_adapted_1536','Fresnel_adapted_2048','cell_averaged']
            axes[1].plot([0,1,2,3],[difference(results[f'{label}_kernel_{wave}'],results[f'reference_kernel_{wave}'])['relative_l1']*100 for label in labels],'o-',label=f'{wave}nm')
        axes[0].set_xlabel('Forvard域边长 / 格点');axes[1].set_xticks([0,1,2,3],['原接口1536','适配1536','适配2048','独立核平均'])
        for ax in axes:ax.set_ylabel('相机核相对L1差 / %');ax.axhline(threshold*100,color='gray',linestyle='--');ax.legend();ax.grid(alpha=.3)
        fig.suptitle('固定输入场，核不缩放；工程阈值不等于论文标准');fig.tight_layout();figures.append(('sampling_decomposition.png',fig))
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));np.savez_compressed(dest/'arrays/sampling.npz',**results)
            write_json(dest/'source_evidence.json',dict(prior_crosscheck=dict(path=str(source),sha256=before),**evidence))
            names=sorted(set([*oldcodes,'main_g1_lp_sampling.py','config_g1_lp_sampling.json','optics/g1_lp_sampling.py','tests/test_g1_lp_sampling.py','G1_LP_SAMPLING_README.md']))
            write_json(dest/'source_manifest.json',{n:file_sha(ROOT/n) for n in names})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            with np.load(dest/'arrays/sampling.npz',allow_pickle=False) as z:
                if any(not np.array_equal(z[k],v) for k,v in results.items()):raise RuntimeError('保存重载不一致')
        if tree_sha(source)!=before or any(tree_sha(Path(i['path']))!=i['sha256'] for i in evidence.values()):raise RuntimeError('来源改变')
        if dest:write_json(dest/'metrics/validation.json',dict(summary=summary,timings=rows,environment=environment_info(),backend=backend,elapsed_seconds=time.perf_counter()-start))
        if runtime['show_plots']:plt.show()
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2));print('采样拆分结果：'+str(dest) if dest else '采样拆分无保存：未写结果文件')
        return summary
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g1_lp_sampling.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['numerical_health_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
