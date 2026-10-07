# -*- coding: utf-8 -*-
"""G1光学对齐补核：只读旧核库、新固定CW高度的传播诊断。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from main_g1 import write_json
from main_stage02b import allocate_unique_run_dir, setup_logger
from optics.g1_forward import profile_for,measure,compare,native_parseval,rotation_summary
from optics.g1_alignment import clockwise_profile,registration_sequence,size_diagnostics
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import configure_plotting,effective_runtime,resolve_project_path
from optics.runutil import environment_info
from optics.fabrication_detector import detector_grids

ROOT=Path(__file__).resolve().parent


def run(c,no_save=False,show_plots=False):
    for value in [c['polar']['correlation_min'],*c['thresholds'].values()]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('诊断阈值须有限且为正')
    if c['polar']['correlation_min']>1:raise ValueError('相关分数阈值不可超过1')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('图片dpi须为至少50的整数')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']):raise ValueError('运行开关须为布尔值')
    if c['clockwise_variant']['enabled'] is not True or c['clockwise_variant']['angular_origin_deg']!=0.:
        raise ValueError('本批预声明启用同+x零角的顺时针诊断')
    if c['clockwise_variant']['refinement_wavelengths_nm']!=[420,540,660]:raise ValueError('本批三控制波长固定')
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('g1_alignment') if runtime['save_results'] else None
    logger=setup_logger(dest);start=time.perf_counter();figures=[]
    try:
        source=resolve_project_path(c['source_run'],ROOT);before=tree_sha(source)
        validation=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
        if validation['status']!='completed' or validation['numerical_validation_passed'] is not True or (source/'failed.json').exists():
            raise ValueError('G1来源未数值通过')
        source_code=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
        for name,sha in source_code.items():
            p=(ROOT/name).resolve()
            if not p.is_relative_to(ROOT) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('G1源码指纹不相容')
        with np.load(source/'arrays/psf_bank.npz',allow_pickle=False) as z:
            bank=z['kernels'];x=z['pixel_x_m'];y=z['pixel_y_m'];waves=z['wavelengths_m'];pitch=float(z['pitch_m']);edges=z['pixel_edges_m']
        cfg=validation['config']
        camera,_,expected_edges=detector_grids(cfg['detector']['pixels_per_axis'],cfg['detector']['pitch_m'],cfg['detector']['quadrature_main'])
        if not (pitch==cfg['detector']['pitch_m'] and np.array_equal(x,camera.x.coords)
                and np.array_equal(y,camera.y.coords) and np.array_equal(edges,expected_edges)):
            raise ValueError('源核库物理坐标与已审核配置不同')
        if bank.shape!=(25,97,97) or not np.array_equal(waves,np.array([n/1e9 for n in range(420,661,10)])):
            raise ValueError('本批要求已审核的25波段97×97核库')
        if not np.all(np.isfinite(bank)) or np.any(bank<0) or np.any(bank.sum(axis=(1,2))<=0):raise ValueError('源核非有限、负值或零场')
        sizes=size_diagnostics(bank,x,y,pitch,c['size'])
        registration,polar,r,theta=registration_sequence(bank,x,y,c['polar'])
        old_rotation=json.loads((source/'metrics/rotation.json').read_text(encoding='utf-8'))
        reference=profile_for(cfg)
        profile=clockwise_profile(reference);fine=clockwise_profile(profile_for(cfg,True))
        fingerprint=profile.compute_fingerprint();fine_fp=fine.compute_fingerprint()
        with np.load(source/'arrays/height_fixed.npz',allow_pickle=False) as z:
            if not np.array_equal(z['delta_h_m'],reference.delta_h):raise ValueError('源高度与固定设计不一致')
            height_reflection_error=float(np.max(np.abs(profile.delta_h-z['delta_h_m'][::-1,:])))
        cw_bank=[];cw_metrics=[];checks=[];sampling=[];energy=[];saved_fields=[]
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime,source_config=cfg))
            np.savez_compressed(dest/'arrays/height_cw_fixed.npz',delta_h_m=profile.delta_h,mask=profile.mask,
                lambda_design_m=profile.lambda_design,x_m=profile.grid.x.coords,y_m=profile.grid.y.coords,
                fingerprint=fingerprint,fine_fingerprint=fine_fp,params_json=json.dumps(profile.params))
        for k,lam in enumerate(waves):
            a=measure(profile,float(lam),cfg,cfg['detector']['quadrature_main']);cw_bank.append(a['kernel']);cw_metrics.append(a['metrics'])
            error=float(np.abs(a['kernel']-bank[k,::-1,:]).sum()/bank[k].sum())
            old=old_rotation['primary']['alpha_unwrapped_deg'][k];cw=a['metrics']['rotation_bands']['primary']['alpha_wrapped_deg']
            angle_error=abs((cw+old+60)%120-60) if cw is not None and old is not None else None
            checks.append(dict(wavelength_nm=round(lam*1e9),reflection_kernel_l1=error,angle_sign_error_deg=angle_error,
                passed=bool(error<=c['thresholds']['reflection_kernel_l1'] and angle_error is not None and angle_error<=c['thresholds']['angle_sign_deg'])))
            if round(lam*1e9) in c['clockwise_variant']['refinement_wavelengths_nm']:
                b=measure(profile,float(lam),cfg,cfg['detector']['quadrature_coarse']);f=measure(fine,float(lam),cfg,cfg['detector']['quadrature_main'])
                for method,left,right in [('quadrature',b,a),('input_refinement',a,f)]:
                    sampling.append(dict(wavelength_nm=round(lam*1e9),method=method,**compare(left,right,cfg)))
                energy.append(dict(wavelength_nm=round(lam*1e9),**native_parseval(a['u1'],profile.grid,float(lam),cfg['optical']['distance_m'])))
                if dest:
                    name=f"cw_field_{round(lam*1e9)}nm.npz";saved_fields.append(name)
                    np.savez_compressed(dest/'arrays'/name,u2_complex=a['u2'],intensity_raw=a['intensity'],
                        pixel_power=a['power'],Pin=a['metrics']['pin'],wavelength_m=lam,fingerprint=fingerprint,
                        node_x_m=a['nodes'].x.coords,node_y_m=a['nodes'].y.coords)
            if profile.compute_fingerprint()!=fingerprint or fine.compute_fingerprint()!=fine_fp:raise RuntimeError('CW高度在波长间改变')
            logger.info('CW实际传播%d/25：%dnm，反射诊断L1=%.3g',k+1,round(lam*1e9),error)
        cw_bank=np.stack(cw_bank);cw_rotation=rotation_summary(cw_metrics,waves)
        cw_sizes=size_diagnostics(cw_bank,x,y,pitch,c['size'])
        wave_nm=np.rint(waves*1e9).astype(int)
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        alpha=np.array(old_rotation['primary']['alpha_unwrapped_deg']);axes[0,0].plot(wave_nm,alpha-alpha[0],'.-',label='C3：旧G1')
        axes[0,0].plot(wave_nm,registration['cumulative_angle_deg'],'.-',label='环带模式相邻配准')
        cw_angle=np.array(cw_rotation['primary']['alpha_unwrapped_deg']);axes[0,0].plot(wave_nm,cw_angle-cw_angle[0],'.-',label='新CW高度实际传播')
        axes[0,0].set_ylabel('相对420nm角度（度；逆时针为正）');axes[0,0].legend()
        axes[0,1].plot(wave_nm[1:],[v['score'] for v in registration['adjacent']],'.-')
        axes[0,1].axhline(c['polar']['correlation_min'],ls='--',c='gray');axes[0,1].set_ylabel('相邻环带模式相关分数')
        for key in ['R50_input_um','R80_input_um','R50_window_um','R80_window_um']:
            axes[1,0].plot(wave_nm,[s[key] for s in sizes],'.-',label=key)
        axes[1,0].set_ylabel('包围能量半径（μm）');axes[1,0].legend(fontsize=8)
        for q in c['size']['peak_thresholds']:
            axes[1,1].plot(wave_nm,[s[f'peak_q{q}_equivalent_radius_um'] for s in sizes],'.-',label=f'峰值阈值{q}')
        axes[1,1].plot(wave_nm,[s['core_rms_radius_um'] for s in sizes],'.-',label='150μm圆域RMS半径')
        axes[1,1].set_ylabel('形状或均方半径（μm）');axes[1,1].legend(fontsize=8)
        for ax in axes.flat:ax.set_xlabel('波长（nm）');ax.grid(alpha=.3)
        fig.tight_layout();figures.append(('angle_size_diagnostics.png',fig))
        fig,axes=plt.subplots(2,3,figsize=(11,7));extent=[edges[0]*1e6,edges[-1]*1e6]*2
        for j,k in enumerate([0,12,24]):
            for i,images in enumerate([bank,cw_bank]):
                image=images[k];axes[i,j].imshow(image/image.max(),origin='lower',extent=extent,cmap='inferno')
                axes[i,j].set_title(f"{'旧CCW高度' if i==0 else '新CW固定高度'} | {wave_nm[k]}nm")
                axes[i,j].set_xlabel('x（μm）');axes[i,j].set_ylabel('y（μm）')
        fig.suptitle('两张不同高度的实际衍射；逐幅峰值归一化仅看形状')
        fig.tight_layout();figures.append(('cw_propagation_comparison.png',fig))
        passed=all(v['passed'] for v in checks+sampling+energy) and registration['all_reliable']
        if dest:
            np.savez_compressed(dest/'arrays/alignment_data.npz',source_kernels=bank,cw_kernels=cw_bank,wavelengths_m=waves,
                pixel_x_m=x,pixel_y_m=y,pitch_m=pitch,polar_samples=polar,r_m=r,theta_rad=theta,fingerprint=fingerprint)
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            write_json(dest/'metrics/sizes.json',dict(source=sizes,cw=cw_sizes))
            write_json(dest/'metrics/registration.json',registration)
            write_json(dest/'metrics/cw_rotation.json',cw_rotation)
            write_json(dest/'metrics/cw_checks.json',dict(reflection=checks,sampling=sampling,native_energy=energy,height_reflection_max_error_m=height_reflection_error))
            with (dest/'metrics/sizes.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['wavelength_nm',*sizes[0]])
                writer.writeheader();writer.writerows(dict(wavelength_nm=int(n),**s) for n,s in zip(wave_nm,sizes))
            # 原始复场重读；像元功率以独立循环在外层审核中再核验。
            with np.load(dest/'arrays/alignment_data.npz',allow_pickle=False) as z:
                expected=dict(cw_kernels=cw_bank,source_kernels=bank,wavelengths_m=waves,pixel_x_m=x,pixel_y_m=y,
                              pitch_m=pitch,polar_samples=polar,r_m=r,theta_rad=theta,fingerprint=fingerprint)
                if any(not np.array_equal(z[k],v) for k,v in expected.items()):raise RuntimeError('核库落盘不一致')
            with np.load(dest/'arrays/height_cw_fixed.npz',allow_pickle=False) as z:
                expected=dict(delta_h_m=profile.delta_h,mask=profile.mask,lambda_design_m=profile.lambda_design,
                              x_m=profile.grid.x.coords,y_m=profile.grid.y.coords,fingerprint=fingerprint,fine_fingerprint=fine_fp)
                if any(not np.array_equal(z[k],v) for k,v in expected.items()):raise RuntimeError('固定高度落盘不一致')
            for name in saved_fields:
                with np.load(dest/'arrays'/name,allow_pickle=False) as z:
                    k=list(wave_nm).index(round(float(z['wavelength_m'])*1e9))
                    if not (np.array_equal(np.abs(z['u2_complex'])**2,z['intensity_raw'])
                            and np.array_equal(z['pixel_power']/float(z['Pin']),cw_bank[k]) and str(z['fingerprint'])==fingerprint):
                        raise RuntimeError('CW保存场不一致')
            files=['main_g1_alignment.py','config_g1_alignment.json','optics/g1_alignment.py',*source_code]
            write_json(dest/'source_manifest.json',{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(set(files))})
            write_json(dest/'source_evidence.json',dict(path=str(source),sha256=before))
            with (dest/'report_alignment.md').open('x',encoding='utf-8') as stream:
                stream.write(f'# G1光学对齐诊断\n\n数值诊断通过：{passed}。保持旧G1只读，新CW固定高度实际计算25波长。输入/求积加密及能量仅在420/540/660nm验证，共31组传播。\n\n图2(b)标注波长顺时针增长，支持角向手性差异这一解释；角零点仍为实施假设，不据此宣称完全对齐。全局反射仅用于比较诊断，未用于生成新PSF。\n\n尺寸量未恒定；图3无物理标尺、无定量尺寸容差或原始PSF数组，paper_alignment_passed=false。完整公式对照及限制见G1_ALIGNMENT_README.md。\n')
        if runtime['show_plots']:plt.show()
        if tree_sha(source)!=before:raise RuntimeError('诊断期间旧G1来源变化')
        if dest:
            required=['arrays/height_cw_fixed.npz','arrays/alignment_data.npz',
                      *['arrays/'+n for n in saved_fields],*['figures/'+n for n,_ in figures],
                      'metrics/sizes.json','metrics/sizes.csv','metrics/registration.json','metrics/cw_rotation.json',
                      'metrics/cw_checks.json','source_manifest.json','source_evidence.json','report_alignment.md','config_effective.json']
            if any(not (dest/n).is_file() or (dest/n).stat().st_size==0 for n in required):raise RuntimeError('必需交付文件缺失')
        report=dict(status='completed' if passed else 'diagnostic_checks_failed',numerical_validation_passed=bool(passed),
            paper_alignment_passed=False,chirality_evidence_found=True,full_cw_bands=25,propagated_fields=31,
            refined_bands_nm=[420,540,660],source_unchanged=True,height_fingerprint=fingerprint,
            environment=environment_info(),runtime=runtime,backend=backend,elapsed_s=time.perf_counter()-start,
            source_c3_span_deg=float(alpha[-1]-alpha[0]),source_registration_span_deg=registration['cumulative_angle_deg'][-1],
            cw_c3_span_deg=float(cw_angle[-1]-cw_angle[0]),
            limitations=['角零点不是作者公开的坐标定义','CW仅三个波长重新进行采样筛查','尺寸变化仍存在','未取得原始PSF及图3物理标尺',
                         'C3累计角与模式配准累计角不同；高相邻相关不证明形状为严格刚性旋转'])
        if dest:write_json(dest/'metrics/validation.json',report)
        print('对齐诊断结果：'+str(dest) if dest else '无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g1_alignment.json')
    p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['numerical_validation_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
