# -*- coding: utf-8 -*-
"""固定DOE周期域5120补核，低内存等价算子先核对LightPipes4096。"""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from main_g1 import write_json
from main_g1_lightpipes import difference
from main_stage02b import allocate_unique_run_dir
from optics.g1_padding import periodic_fresnel,power_on_grid
from optics.fabrication_detector import detector_grids,integrate_square_pixels
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def run(c,no_save=False,show_plots=False):
    start=time.perf_counter()
    if c['wavelengths_nm']!=[420,540,660] or c['new_padding_size']!=5120:raise ValueError('本批固定三控制波长和5120域')
    if type(c['block_rows']) is not int or not 1<=c['block_rows']<=256:raise ValueError('块行数须1至256')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']) or type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('运行参数非法')
    for k in ['comparison_relative_threshold','equivalence_relative_threshold']:
        if isinstance(c[k],bool) or not np.isfinite(c[k]) or not 0<c[k]<1:raise ValueError('容差须有限且介于0与1')
    source=resolve_project_path(c['source_run'],ROOT);before=tree_sha(source)
    audit=json.loads(resolve_project_path(c['source_audit'],ROOT).read_text(encoding='utf-8'))
    if not audit['independent_audit']['passed'] or before!=audit['run_sha256']:raise ValueError('来源未审核或改变')
    for p,sha in audit['sha256'].items():
        if file_sha(Path(p))!=sha:raise ValueError('来源源码或说明锁改变')
    codes=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if any(file_sha(ROOT/n)!=sha for n,sha in codes.items()):raise ValueError('物理源码改变')
    evidence=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    if any(tree_sha(Path(i['path']))!=i['sha256'] for i in evidence.values()):raise ValueError('旧run或依赖改变')
    arrays={};rows=[];equivalence=[]
    with np.load(source/'arrays/sampling.npz',allow_pickle=False) as archive:
        axis=archive['native_axis_m'].copy();fingerprint=str(archive['height_fingerprint'])
        _,nodes,_=detector_grids(97,6.22e-6,8)
        yy,xx=np.meshgrid(nodes.y.coords,nodes.x.coords,indexing='ij');points=np.column_stack((yy.ravel(),xx.ravel()))
        for wave in c['wavelengths_nm']:
            u1=archive[f'u1_{wave}'].copy();reference=archive[f'reference_kernel_{wave}'].copy()
            saved4096=archive[f'Forvard_4096_native_{wave}'].copy();kernel4096=archive[f'Forvard_4096_kernel_{wave}'].copy()
            pin=power_on_grid(u1,1e-6,c['block_rows']);new=None
            arrays[f'u1_{wave}']=u1;arrays[f'reference_kernel_{wave}']=reference
            arrays[f'LightPipes_4096_native_{wave}']=saved4096;arrays[f'LightPipes_4096_kernel_{wave}']=kernel4096
            for n in [4096,5120]:
                print(f'{wave}nm：低内存周期Fresnel域{n}；4096核对库结果，5120补核',flush=True)
                field=np.zeros((n,n),complex);offset=n//2-550;field[offset:offset+1101,offset:offset+1101]=u1
                output=periodic_fresnel(field,1e-6,wave*1e-9,.05,c['block_rows']);del field
                crop=output[n//2-304:n//2+305,n//2-304:n//2+305].copy()
                ratio=power_on_grid(output,1e-6,c['block_rows'])/pin;del output
                if not np.isfinite(crop).all():raise ValueError('非有限输出场')
                intensity=np.abs(crop)**2
                samples=RegularGridInterpolator((axis,axis),intensity,bounds_error=True)(points).reshape(len(nodes.y.coords),len(nodes.x.coords))
                kernel=integrate_square_pixels(samples,97,6.22e-6,8)[0]/pin
                arrays[f'equivalent_{n}_native_{wave}']=crop;arrays[f'equivalent_{n}_kernel_{wave}']=kernel
                if n==4096:
                    phase=np.vdot(crop,saved4096);phase/=abs(phase)
                    equivalence.append(dict(wavelength_nm=wave,unit_phase_aligned_field_relative_l2=float(np.linalg.norm(crop*phase-saved4096)/np.linalg.norm(saved4096)),
                                            detector_kernel_difference=difference(kernel,kernel4096)))
                else:new=(kernel,ratio)
            if not equivalence[-1]['unit_phase_aligned_field_relative_l2']<c['equivalence_relative_threshold']:raise RuntimeError('等价实现未与实际LightPipes场匹配')
            change=difference(new[0],kernel4096);fit=difference(new[0],reference)
            rows.append(dict(wavelength_nm=wave,change_4096_to_5120=change,reference_difference=fit,full_grid_power_over_input=new[1],
                             detector_efficiency=float(new[0].sum()),same_input_height_fingerprint=fingerprint))
    threshold=c['comparison_relative_threshold']
    passed=all(r['change_4096_to_5120']['relative_l1']<threshold and r['change_4096_to_5120']['relative_l2']<threshold for r in rows)
    reference_passed=all(r['reference_difference']['relative_l1']<threshold and r['reference_difference']['relative_l2']<threshold for r in rows)
    summary=dict(equivalence=equivalence,rows=rows,last_padding_pair_below_threshold=passed,reference_below_threshold=reference_passed,
                 numerical_health_passed=all(abs(r['full_grid_power_over_input']-1)<1e-10 and 0<r['detector_efficiency']<=1.02 for r in rows),
                 comparison_threshold=threshold,equivalence_threshold=c['equivalence_relative_threshold'],new_padding_size=5120,
                 execution_backend='scipy FFT periodic Fresnel; not a direct LightPipes5120 call',
                 direct_lightpipes_5120_executed=False,lightpipes_4096_source_reproduced=True,
                 height_fingerprint=fingerprint,paper_alignment_passed=False,infinite_domain_convergence_proved=False,
                 scope='three control wavelengths and one adjacent padding pair; same sampled DOE and common detector pixels')
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g1_padding') if runtime['save_results'] else None
    try:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(c['wavelengths_nm'],[r['change_4096_to_5120']['relative_l1']*100 for r in rows],'o-',label='相邻域L1')
        axes[1].plot(c['wavelengths_nm'],[r['reference_difference']['relative_l1']*100 for r in rows],'o-',label='5120对参考L1')
        for ax in axes:ax.axhline(threshold*100,color='gray',linestyle='--');ax.set_xlabel('波长 / nm');ax.set_ylabel('相对差 / %');ax.legend();ax.grid(alpha=.3)
        fig.suptitle('等价周期算子；有限5120域诊断，非无限域证明');fig.tight_layout();figures.append(('padding_5120_checks.png',fig))
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime))
            np.savez_compressed(dest/'arrays/padding.npz',native_axis_m=axis,**arrays)
            write_json(dest/'source_evidence.json',dict(sampling=dict(path=str(source),sha256=before),**evidence))
            names=sorted(set([*codes,'main_g1_padding.py','config_g1_padding.json','optics/g1_padding.py','tests/test_g1_padding.py','G1_PADDING_README.md']))
            write_json(dest/'source_manifest.json',{n:file_sha(ROOT/n) for n in names})
            for n,fig in figures:fig.savefig(dest/'figures'/n,dpi=c['plots']['dpi'])
            with np.load(dest/'arrays/padding.npz',allow_pickle=False) as z:
                if any(not np.array_equal(z[k],v) for k,v in arrays.items()):raise RuntimeError('保存重载不一致')
        if tree_sha(source)!=before or any(tree_sha(Path(i['path']))!=i['sha256'] for i in evidence.values()):raise RuntimeError('旧来源改变')
        if dest:write_json(dest/'metrics/validation.json',dict(summary=summary,environment=environment_info(),backend=backend,elapsed_seconds=time.perf_counter()-start))
        if runtime['show_plots']:plt.show()
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2));print('5120域结果：'+str(dest) if dest else '5120域无保存：未写结果文件')
        return summary
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g1_padding.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['numerical_health_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
