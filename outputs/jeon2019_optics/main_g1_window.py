# -*- coding: utf-8 -*-
"""G1固定高度的探测窗口诊断：复场传播、公共像元、独立尺寸口径。"""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
from main_g1 import write_json
from main_stage02b import allocate_unique_run_dir
from optics.g1_forward import profile_for, measure
from optics.g1_alignment import clockwise_profile, size_diagnostics
from optics.materials import refractive_index_fused_silica
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path, effective_runtime, configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def run(c,no_save=False,show_plots=False):
    start=time.perf_counter()
    if c['wavelengths_nm']!=list(range(420,661,30)) or c['pixels_per_axis']!=161 or c['quadrature_main']!=8 or c['quadrature_reference']!=4:
        raise ValueError('本批固定图3九波长、161像元、8/4求积，不搜索高分参数')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']) or type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:
        raise ValueError('开关或dpi非法')
    if any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in c['thresholds'].values()):
        raise ValueError('阈值须有限正值')
    source=resolve_project_path(c['source_alignment'],ROOT);g1=resolve_project_path(c['source_g1'],ROOT)
    sources={str(p):tree_sha(p) for p in [source,g1]}
    for p in [source,g1]:
        if (p/'failed.json').exists() or not json.loads((p/'metrics/validation.json').read_text(encoding='utf-8'))['numerical_validation_passed']:
            raise ValueError('来源未通过数值核验')
        codes=json.loads((p/'source_manifest.json').read_text(encoding='utf-8'))
        for name,sha in codes.items():
            if file_sha(ROOT/name)!=sha:raise ValueError('冻结来源源码改变')
    original=json.loads((g1/'config_effective.json').read_text(encoding='utf-8'))
    config=copy.deepcopy(original);config['detector']['pixels_per_axis']=c['pixels_per_axis']
    profile=clockwise_profile(profile_for(original))
    with np.load(source/'arrays/height_cw_fixed.npz',allow_pickle=False) as z:
        if not np.array_equal(profile.delta_h,z['delta_h_m']) or profile.compute_fingerprint()!=str(z['fingerprint']):
            raise ValueError('未重现同一CW高度')
    with np.load(source/'arrays/alignment_data.npz',allow_pickle=False) as z:
        old_bank=z['cw_kernels'].copy();old_waves=z['wavelengths_m'].copy();pitch=float(z['pitch_m'])
    # 直接核对论文式(7)–(11)，局部设计波长下光程差必须为整数周期。
    radius=profile.grid.radius();f=original['optical']['focal_length_m']
    delta=radius**2/(np.sqrt(radius**2+f**2)+f)
    cycles=(delta+(refractive_index_fused_silica(profile.lambda_design)-1)*profile.delta_h)/profile.lambda_design
    pupil=profile.mask==1
    phase_error=float(np.max(np.abs(cycles[pupil]-np.rint(cycles[pupil]))))
    phase_range=float(np.max(delta[pupil]/profile.lambda_design[pupil]))
    banks=[];checks=[];coordinate=None;control=None
    for wave in c['wavelengths_nm']:
        print(f'同一高度扩大窗口：{wave}nm，161×161像元，q8及q4',flush=True)
        main=measure(profile,wave*1e-9,config,8);reference=measure(profile,wave*1e-9,config,4)
        bank=main['kernel'];crop=bank[32:-32,32:-32]
        k=int(np.argmin(np.abs(old_waves-wave*1e-9)))
        if abs(old_waves[k]-wave*1e-9)>1e-15:raise ValueError('旧波长匹配失败')
        crop_error=float(np.sum(np.abs(crop-old_bank[k]))/np.sum(old_bank[k]))
        q_l1=float(np.sum(np.abs(bank-reference['kernel']))/np.sum(bank))
        q_l2=float(np.linalg.norm(bank-reference['kernel'])/np.linalg.norm(bank))
        checks.append(dict(wavelength_nm=wave,crop_relative_l1=crop_error,quadrature_relative_l1=q_l1,quadrature_relative_l2=q_l2,
                           old_eta=float(old_bank[k].sum()),expanded_eta=float(bank.sum()),same_height_fingerprint=profile.compute_fingerprint()))
        banks.append(bank);coordinate=main['camera'].x.coords.copy()
        if wave==540:
            control=dict(u1_complex=main['u1'],u2_complex=main['u2'],pixel_power=main['power'],Pin=main['metrics']['pin'],
                         node_x_m=main['nodes'].x.coords,node_y_m=main['nodes'].y.coords,wavelength_m=wave*1e-9)
    bank=np.stack(banks);selected=[int(np.argmin(np.abs(old_waves-w*1e-9))) for w in c['wavelengths_nm']]
    size_config=dict(shape_roi_m=150e-6,peak_thresholds=[.5,.1,.05])
    new_sizes=size_diagnostics(bank,coordinate,coordinate,pitch,size_config)
    old_x=(np.arange(97)-48)*pitch
    old_sizes=size_diagnostics(old_bank[selected],old_x,old_x,pitch,size_config)
    threshold=c['thresholds']
    passed=phase_error<threshold['phase_cycles'] and all(r['crop_relative_l1']<threshold['crop_relative_l1'] and
        r['quadrature_relative_l1']<threshold['quadrature_relative_l1'] and r['quadrature_relative_l2']<threshold['quadrature_relative_l2'] for r in checks)
    report=dict(status='completed',numerical_validation_passed=bool(passed),paper_alignment_passed=False,
                fixed_height_fingerprint=profile.compute_fingerprint(),height_exactly_reproduced=True,
                design_phase_integer_cycle_max_error=phase_error,maximum_geometric_path_cycles=phase_range,
                propagated_fields=18,wavelengths_nm=c['wavelengths_nm'],checks=checks,
                input_refinement_repeated=False,window_pixels_old=97,window_pixels_new=161,pitch_m=pitch,
                old_sizes=old_sizes,expanded_sizes=new_sizes,environment=environment_info(),elapsed_compute_seconds=time.perf_counter()-start,
                limitations=['论文图3无物理标尺及定量尺寸容差','本批不改变固定高度/材料','新增大窗口未重复细输入网格筛查',
                             '九波长不是25波长的全带宽重建核','有限窗口效率未强制归一','图3模拟的具体材料色散和显示处理未知'])
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g1_window') if runtime['save_results'] else None
    try:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for rows,label in [(old_sizes,'97像元'),(new_sizes,'161像元')]:
            for key in ['R50_input_um','R80_input_um']:
                axes[0].plot(c['wavelengths_nm'],[r[key] for r in rows],label=f'{label} {key}')
            axes[1].plot(c['wavelengths_nm'],[r['R80_window_um'] for r in rows],label=label)
            axes[2].plot(c['wavelengths_nm'],[r['eta_window'] for r in rows],label=label)
        for ax,title in zip(axes,['输入总功率分母的R50/R80 / µm','窗口功率分母的R80 / µm','有限窗口效率']):
            ax.set_title(title);ax.set_xlabel('波长 / nm');ax.grid(alpha=.3);ax.legend(fontsize=7)
        fig.tight_layout();figures.append(('window_size_energy.png',fig))
        fig,axes=plt.subplots(3,3,figsize=(10,10));extent=[-(161/2)*pitch*1e6,(161/2)*pitch*1e6]*2
        for ax,image,w in zip(axes.flat,bank,c['wavelengths_nm']):
            ax.imshow(image/image.max(),origin='lower',extent=extent,cmap='inferno');ax.set_title(f'{w}nm');ax.set_xlabel('x / µm');ax.set_ylabel('y / µm')
        fig.suptitle('固定CW高度，统一物理窗口；峰值归一化仅用于形状显示');fig.tight_layout();figures.append(('expanded_psf_shapes.png',fig))
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime))
            np.savez_compressed(dest/'arrays/window_bank.npz',expanded_kernels=bank,old_kernels=old_bank[selected],wavelengths_m=np.array(c['wavelengths_nm'])*1e-9,
                                pixel_x_m=coordinate,pixel_y_m=coordinate,pitch_m=pitch,fingerprint=profile.compute_fingerprint())
            np.savez_compressed(dest/'arrays/control_540nm.npz',**control)
            write_json(dest/'source_evidence.json',{str(p):sha for p,sha in sources.items()})
            files=sorted(set([*codes,'main_g1_alignment.py','config_g1_alignment.json','optics/g1_alignment.py',
                              'main_g1_window.py','config_g1_window.json','G1_WINDOW_README.md']))
            write_json(dest/'source_manifest.json',{n:file_sha(ROOT/n) for n in files})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            with np.load(dest/'arrays/window_bank.npz',allow_pickle=False) as z:
                if not np.array_equal(z['expanded_kernels'],bank):raise RuntimeError('保存重载核不一致')
        if any(tree_sha(Path(p))!=sha for p,sha in sources.items()):raise RuntimeError('冻结来源改变')
        if dest:write_json(dest/'metrics/validation.json',dict(report,runtime=runtime,backend=backend))
        if runtime['show_plots']:plt.show()
        print(json.dumps(dict(numerical_validation_passed=bool(passed),design_phase_error=phase_error,checks=checks,old_sizes=old_sizes,expanded_sizes=new_sizes),ensure_ascii=False,allow_nan=False,indent=2))
        print('窗口诊断结果：'+str(dest) if dest else '窗口诊断无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g1_window.json')
    p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['numerical_validation_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
