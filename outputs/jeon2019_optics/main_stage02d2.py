# -*- coding: utf-8 -*-
"""Jeon光学编码复现＋基础重建验证；CPU无参数入口。"""
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback
import numpy as np
from optics.imaging import SpectralImager,operator_checks
from optics.reconstruction import solve_nonnegative_ridge,estimate_data_lipschitz,evaluate_cube,validate_solver,squared_norm
from optics.stage02d1_source import load_imaging_source
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir,setup_logger

ROOT=Path(__file__).resolve().parent


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def validate_config(c):
    validate_solver(c['solver'])
    if not np.isfinite(c['evaluation']['spectrum_norm_threshold']) or c['evaluation']['spectrum_norm_threshold']<=0:
        raise ValueError('评价光谱阈值须为正')
    if c['full_diagnostic_scene'] not in ['lines_and_square','coincident_points','separated_points']:
        raise ValueError('full诊断场景必须来自源run')
    for v in c['thresholds'].values():
        if not np.isfinite(v) or v<=0: raise ValueError('验证容差必须有限且为正')


def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots)
    plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02d2') if runtime['save_results'] else None
    logger=setup_logger(dest) if dest else logging.getLogger('stage02d2_memory')
    if not dest:
        logger.setLevel(logging.INFO)
        if not logger.handlers:logger.addHandler(logging.StreamHandler())
    start=time.perf_counter();figures=[]
    try:
        items,evidence=load_imaging_source(resolve_project_path(c['source_run'],ROOT),ROOT)
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime))
            write_json(dest/'source_evidence.json',evidence)
        records=[];spectral_cache={};saved={};operator_valid=True
        for item in items:
            v=item['values'];device=item['device'];label=item['scene']
            modes=['crop']+(['full'] if label==c['full_diagnostic_scene'] else [])
            for mode in modes:
                crop=v['crop_indices'].tolist() if mode=='crop' else None
                op=SpectralImager(v['kernels'],list(v['cube'].shape[1:]),float(v['pitch_m']),v['response'],crop)
                y=v['measurement_'+mode]
                ident=device+'_'+label+'_'+mode
                check=operator_checks(op,c['solver']['seed'],y)
                operator_valid=operator_valid and check['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and check['gradient_relative_error']<=c['thresholds']['gradient_relative']
                cache_key=(device,mode)
                if cache_key not in spectral_cache: spectral_cache[cache_key]=estimate_data_lipschitz(op,c['solver'])
                result=solve_nonnegative_ridge(op,y,c['solver'],spectral_cache[cache_key])
                x=result['reconstruction'];fit=op.forward(x)
                evaluation=evaluate_cube(v['cube'],x,c['evaluation']['spectrum_norm_threshold'])
                history=result['history'];last=history[-1]
                trajectory_monotone=all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300)
                                        for a,b in zip(history,history[1:]))
                if not trajectory_monotone or np.any(x<0) or not np.all(np.isfinite(x)):
                    raise RuntimeError('重建非负性、有限性或单调下降检查失败')
                rec=dict(identity=ident,device=device,scene=label,mode=mode,solver_status=result['status'],
                    iterations=result['iterations'],alpha=result['alpha'],L_data_estimate=result['L_data_estimate'],
                    elapsed_s=result['elapsed_s'],final_optimization=last,evaluation=evaluation,
                    operator_checks=check,trajectory_monotone=trajectory_monotone,initialization=result['initialization'],
                    projected_gradient_normalizer=result['projected_gradient_normalizer'])
                records.append(rec)
                vals=dict(truth=v['cube'],reconstruction=x,measurement=y,prediction=fit,residual=fit-y,
                    kernels=v['kernels'],response=v['response'],wavelengths_m=v['wavelengths_m'],pitch_m=v['pitch_m'],
                    fixed_height_fingerprint=v['fixed_height_fingerprint'],crop_indices=np.asarray(op.crop),
                    alpha=result['alpha'],**op.coordinates())
                saved[ident]=dict(values=vals,history=history,result=result)
                if dest:
                    np.savez_compressed(dest/'arrays'/(ident+'.npz'),**vals)
                    write_json(dest/'metrics'/(ident+'_optimization.json'),dict(status=result['status'],history=history,
                               spectral_estimate=result['spectral_estimate'],alpha=result['alpha']))
                    with (dest/'metrics'/(ident+'_optimization.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(last));writer.writeheader();writer.writerows(history)
                logger.info('%s：%s，%d轮，数据残差%.4g，cube误差%.4g，投影梯度%.4g',ident,result['status'],
                    result['iterations'],last['residual_relative'] or 0.,evaluation['cube_relative_l2'] or 0.,last['projected_gradient_relative'])
                fig,axs=plt.subplots(3,3,figsize=(10,9));figures.append(fig)
                coord=op.coordinates();p=float(v['pitch_m'])
                extent=[(coord['scene_x_m'][0]-p/2)*1e6,(coord['scene_x_m'][-1]+p/2)*1e6,
                        (coord['scene_y_m'][0]-p/2)*1e6,(coord['scene_y_m'][-1]+p/2)*1e6]
                vmax=max(float(v['cube'].max()),float(x.max()))
                for b in range(3):
                    for col,arr,title in [(0,v['cube'][b],'真值'),(1,x[b],'恢复'),(2,np.abs(x[b]-v['cube'][b]),'绝对误差')]:
                        im=axs[b,col].imshow(arr,origin='lower',extent=extent,cmap='viridis',vmin=0,vmax=vmax)
                        axs[b,col].set_title('%d nm %s'%(round(float(v['wavelengths_m'][b])*1e9),title))
                        axs[b,col].set_xlabel('x (μm)');axs[b,col].set_ylabel('y (μm)')
                        fig.colorbar(im,ax=axs[b,col],fraction=.046,pad=.04)
                fig.suptitle(ident+' | '+result['status']+'；所有面板共用原始功率色标');fig.tight_layout()
                if dest:fig.savefig(dest/'figures'/(ident+'_cube.png'),dpi=c['plots']['dpi'])
                diag,ax=plt.subplots(1,3,figsize=(12,3.8));figures.append(diag)
                ex=[(coord['output_x_m'][0]-p/2)*1e6,(coord['output_x_m'][-1]+p/2)*1e6,
                    (coord['output_y_m'][0]-p/2)*1e6,(coord['output_y_m'][-1]+p/2)*1e6]
                max_y=max(float(y.max()),float(fit.max()))
                for i,arr,title in [(0,y,'测量功率'),(1,fit,'前向拟合功率'),(2,fit-y,'有符号残差')]:
                    m=max(abs(float(arr.min())),abs(float(arr.max()))) if i==2 else max_y
                    im=ax[i].imshow(arr,origin='lower',extent=ex,cmap='coolwarm' if i==2 else 'inferno',vmin=-m if i==2 else 0,vmax=m)
                    ax[i].set_title(title);figlabel=diag.colorbar(im,ax=ax[i],fraction=.046,pad=.04)
                    ax[i].set_xlabel('x (μm)');ax[i].set_ylabel('y (μm)')
                diag.suptitle(ident+'：原始相对功率；残差色标单独标示');diag.tight_layout()
                if dest:diag.savefig(dest/'figures'/(ident+'_fit.png'),dpi=c['plots']['dpi'])
        curve,axs=plt.subplots(1,2,figsize=(12,4));figures.append(curve)
        for ident,value in saved.items():
            h=value['history'];iterations=[r['iteration'] for r in h]
            axs[0].semilogy(iterations,[max(r['objective'],1e-300) for r in h],label=ident)
            axs[1].semilogy(iterations,[max(r['projected_gradient_relative'],1e-300) for r in h],label=ident)
        axs[0].set_title('正则化目标');axs[1].set_title('归一化投影梯度映射');axs[1].axhline(c['solver']['gradient_tolerance'],ls='--',color='k')
        for ax in axs:ax.set_xlabel('迭代');ax.legend(fontsize=6)
        curve.tight_layout()
        if dest:curve.savefig(dest/'figures'/'optimization_curves.png',dpi=c['plots']['dpi'])
        if tree_sha(Path(evidence['source_run']))!=evidence['sha256'] or tree_sha(Path(evidence['optical_source_run']))!=evidence['optical_sha256']:
            raise RuntimeError('运行期间源成像/光学目录变化')
        readback=[]
        if dest:
            if {p.stem for p in (dest/'arrays').glob('*.npz')}!=set(saved):raise RuntimeError('重建数组身份集合错误')
            for ident,value in saved.items():
                with np.load(dest/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                    if set(z.files)!=set(value['values']) or not all(np.array_equal(z[k],v) for k,v in value['values'].items()):raise RuntimeError('重建落盘内容不一致')
                    op=SpectralImager(z['kernels'],list(z['truth'].shape[1:]),float(z['pitch_m']),z['response'],z['crop_indices'].tolist())
                    if not np.array_equal(op.forward(z['reconstruction']),z['prediction']):raise RuntimeError('重建落盘前向拟合不一致')
                    readback.append(ident)
                for filename in [ident+'_cube.png',ident+'_fit.png']:
                    if not (dest/'figures'/filename).is_file():raise RuntimeError('重建结果图缺失')
                trajectory=json.loads((dest/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))
                if trajectory['history']!=value['history'] or trajectory['status']!=value['result']['status']:
                    raise RuntimeError('落盘优化轨迹不一致')
                with (dest/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as stream:
                    if len(list(csv.DictReader(stream)))!=len(value['history']):raise RuntimeError('优化CSV轮数不一致')
            if not (dest/'figures'/'optimization_curves.png').is_file():raise RuntimeError('优化曲线缺失')
        report=dict(status='completed',validation_passed=bool(operator_valid),
            all_reconstructions_converged=all(r['solver_status']=='converged' for r in records),
            records=records,readback=readback,config=c,runtime=runtime,backend=backend,source_unchanged=True,
            environment=environment_info(),elapsed_s=time.perf_counter()-start,
            scope='Jeon光学编码复现＋基础重建验证；非原论文网络复现',
            assumptions=['无噪声、三单色诊断、理想像平面平移不变模型，单位辐射响应。',
                         'α由统一相对参数乘算子谱范数估计；幂迭代不是严格上界，回溯检查步长。',
                         'iteration_limit表示达到迭代上限，不能声称约束最优解已收敛。',
                         '真值仅用于来源验证和评价，不用于初始化/迭代或每场景参数选择。',
                         '小数据残差不能代替光谱恢复误差；有限场景支持是先验。',
                         '原论文旋转方向对应、真实PyCharm点击和弹窗仍未确定/人工验证。'])
        if dest:
            paths=['main_stage02d2.py','config_stage02d2.json','optics/reconstruction.py','optics/stage02d1_source.py',
                   'optics/imaging.py','optics/stage02c_source.py','optics/stage02_runtime.py','optics/runutil.py']
            write_json(dest/'source_manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
            with (dest/'metrics'/'evaluation.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['identity','status','iterations','alpha','data_residual_relative','cube_relative_l2','SAM_mean_deg','projected_gradient_relative'])
                for r in records:writer.writerow([r['identity'],r['solver_status'],r['iterations'],r['alpha'],
                    r['final_optimization']['residual_relative'],r['evaluation']['cube_relative_l2'],r['evaluation']['sam_mean_deg'],r['final_optimization']['projected_gradient_relative']])
            lines=['# 02D-2基础重建实际运行报告','','Jeon光学编码复现＋基础重建验证；非原论文网络复现。',
                '方法：零初始化、非负岭投影梯度与回溯；F=0.5||AX−y||²+α||X||²/2。','',
                '|实验|状态|轮数|数据残差相对值|cube相对L2|SAM/deg|', '|---|---|---:|---:|---:|---:|']
            for r in records:lines.append('|%s|%s|%d|%.5g|%.5g|%s|'%(r['identity'],r['solver_status'],r['iterations'],
                r['final_optimization']['residual_relative'] or 0.,r['evaluation']['cube_relative_l2'] or 0.,r['evaluation']['sam_mean_deg']))
            lines+=['','## 限制','']+['- '+s for s in report['assumptions']]
            lines+=['','逐轮目标/梯度/步长及评价见metrics；原始恢复与测量在arrays；所有图为相对功率。']
            (dest/'report_stage02d2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            write_json(dest/'metrics'/'validation.json',report)
        print(json.dumps(dict(status='completed',validation_passed=report['validation_passed'],
            all_reconstructions_converged=report['all_reconstructions_converged'],run_dir=str(dest) if dest else None,
            elapsed_s=report['elapsed_s']),ensure_ascii=False))
        if runtime['show_plots']:plt.show()
        return report
    except Exception:
        if dest:write_json(dest/'failed.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figures:plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_stage02d2.json');parser.add_argument('--no-save',action='store_true');parser.add_argument('--show-plots',action='store_true')
    args=parser.parse_args();c=json.loads(resolve_project_path(args.config,ROOT).read_text(encoding='utf-8'))
    r=run(c,args.no_save,args.show_plots)
    return 0 if r['validation_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
