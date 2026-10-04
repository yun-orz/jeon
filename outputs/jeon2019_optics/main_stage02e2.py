# -*- coding: utf-8 -*-
"""02E-2弱α数值稳定性复核，Windows/PyCharm CPU无参数入口。"""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from optics.imaging import SpectralImager,operator_checks
from optics.reconstruction import evaluate_cube
from optics.reconstruction_accelerated import solve_accelerated_ridge,validate_accelerated
from optics.ridge_certificate import ridge_certificate
from optics.stage02e1_source import load_sensitivity_source
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir
ROOT=Path(__file__).resolve().parent

def write_json(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_config(c):
    validate_accelerated(c['solver'])
    if not c['scenes'] or len(set(c['scenes']))!=len(c['scenes']) or any(s not in ['coincident_points','lines_and_square'] for s in c['scenes']):raise ValueError('本阶段仅复核共点和线条场景')
    if isinstance(c['alpha_factor'],bool) or c['alpha_factor']!=.01:raise ValueError('本阶段固定α倍数0.01')
    for key in ['certificate_relative_threshold','budget_seconds']:
        if isinstance(c[key],bool) or not np.isfinite(c[key]) or c[key]<=0:raise ValueError(key+'须正有限')
    if isinstance(c['check_seed'],bool) or not isinstance(c['check_seed'],int) or c['check_seed']<0:raise ValueError('种子须非负整数')
    if not np.isfinite(c['evaluation']['spectrum_norm_threshold']) or c['evaluation']['spectrum_norm_threshold']<=0:raise ValueError('SAM阈值须正有限')
    if any(not np.isfinite(v) or v<=0 for v in c['thresholds'].values()):raise ValueError('算子容差须正有限')
    if isinstance(c['plots']['dpi'],bool) or not isinstance(c['plots']['dpi'],int) or c['plots']['dpi']<=0:raise ValueError('dpi须正整数')

def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02e2') if runtime['save_results'] else None
    start=time.perf_counter();records=[];saved={};figures=[]
    try:
        items,evidence=load_sensitivity_source(resolve_project_path(c['source_run'],ROOT),ROOT)
        for key in ['gradient_tolerance','objective_tolerance']:
            if c['solver'][key]>=evidence['source_config']['solver'][key]:raise ValueError('本阶段停止门槛须严于原主扫描')
        jobs=[next(item for item in items if item['scene']==scene and item['factor']==c['alpha_factor']) for scene in c['scenes']]
        if dest:write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'source_evidence.json',evidence)
        valid=True;budget_exhausted=False
        for item in jobs:
            # 完整实验之间检查预算；不会删除或丢弃前组结果。
            if records and time.perf_counter()-start>c['budget_seconds']:budget_exhausted=True;break
            v=item['values'];ident=item['identity']+'_refined'
            op=SpectralImager(v['kernels'],list(v['truth'].shape[1:]),float(v['pitch_m']),v['response'],v['crop_indices'].tolist())
            check=operator_checks(op,c['check_seed'],v['measurement'])
            valid=valid and check['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and check['gradient_relative_error']<=c['thresholds']['gradient_relative']
            alpha=float(v['alpha']);result=solve_accelerated_ridge(op,v['measurement'],alpha,c['solver'],item['L_initial'])
            x=result['reconstruction'];prediction=op.forward(x);residual=prediction-v['measurement']
            h=result['history'];cert=ridge_certificate(op,v['measurement'],x,alpha)
            if np.any(x<0) or not np.all(np.isfinite(x)) or not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:])):raise RuntimeError('恢复非负性或轨迹错误')
            ev=evaluate_cube(v['truth'],x,c['evaluation']['spectrum_norm_threshold'])
            old=item['record'];certified=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=c['certificate_relative_threshold']
            rec=dict(identity=ident,scene=item['scene'],device='continuous',mode='crop',alpha_factor=item['factor'],alpha=alpha,
                baseline_alpha=old['baseline_alpha'],solver_config=c['solver'],solver_status=result['status'],iterations=result['iterations'],elapsed_s=result['elapsed_s'],
                initialization=result['initialization'],operator_checks=check,final_optimization=h[-1],certificate=cert,
                certificate_passed=bool(certified),certificate_relative_threshold=c['certificate_relative_threshold'],evaluation=ev,
                recovered_to_truth_total_power=float(np.sum(x)/np.sum(v['truth'])),baseline_record=old,
                baseline_reconstruction_relative_change=float(np.linalg.norm(x-v['reconstruction'])/max(np.linalg.norm(x),1e-300)),
                baseline_objective_decrease=old['final_optimization']['objective']-h[-1]['objective'],
                baseline_cube_error_change=ev['cube_relative_l2']-old['evaluation']['cube_relative_l2'])
            values=dict(v,reconstruction=x,prediction=prediction,residual=residual)
            saved[ident]=dict(values=values,history=h);records.append(rec)
            if dest:
                np.savez_compressed(dest/'arrays'/(ident+'.npz'),**values)
                write_json(dest/'metrics'/(ident+'_optimization.json'),dict(alpha=alpha,status=result['status'],history=h))
                with (dest/'metrics'/(ident+'_optimization.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(h[0]));w.writeheader();w.writerows(h)
                write_json(dest/'progress.json',dict(status='running',completed_jobs=len(records),planned_jobs=len(jobs),last_record=rec))
            print('%s: %s, %d轮, cube=%.6g, 最优解距离相对上界=%.6g, 1%%界通过=%s'%(ident,result['status'],result['iterations'],ev['cube_relative_l2'],cert['distance_upper_bound_relative'],certified),flush=True)
            # 真值、旧结果、新结果和绝对误差使用统一原始功率色标。
            fig,axes=plt.subplots(3,4,figsize=(12,8));figures.append(fig)
            vmax=max(float(v['truth'].max()),float(v['reconstruction'].max()),float(x.max()))
            for band in range(3):
                for col,(title,arr) in enumerate([('真值',v['truth'][band]),('02E-1',v['reconstruction'][band]),('02E-2',x[band]),('新绝对误差',abs(x[band]-v['truth'][band]))]):
                    im=axes[band,col].imshow(arr,vmin=0,vmax=vmax,origin='lower');axes[band,col].set_title(title+' %.0f nm'%(v['wavelengths_m'][band]*1e9));axes[band,col].set_xlabel('x/像元');axes[band,col].set_ylabel('y/像元')
            fig.suptitle(item['scene']+' '+result['status']);fig.subplots_adjust(right=.91,wspace=.4,hspace=.45);fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,label='相对功率')
            if dest:fig.savefig(dest/'figures'/(ident+'_comparison.png'),dpi=c['plots']['dpi'])
        curve,axs=plt.subplots(1,2,figsize=(12,4));figures.append(curve)
        for rec in records:
            h=saved[rec['identity']]['history']
            axs[0].semilogy([a['iteration'] for a in h],[max(a['projected_gradient_relative'],1e-300) for a in h],label=rec['scene'])
            axs[1].semilogy([a['iteration'] for a in h],[max(a['objective_relative_change'],1e-300) for a in h],label=rec['scene'])
        for ax,key,title in zip(axs,['gradient_tolerance','objective_tolerance'],['归一化投影梯度','目标相对变化']):
            ax.axhline(c['solver'][key],color='k',ls='--');ax.set_title(title);ax.set_xlabel('迭代');ax.legend()
        curve.tight_layout()
        if dest:curve.savefig(dest/'figures/optimization_curves.png',dpi=c['plots']['dpi'])
        for source in evidence['sources']:
            if tree_sha(Path(source['path']))!=source['sha256']:raise RuntimeError('运行期间源目录改变')
        if dest:
            if {p.stem for p in (dest/'arrays').glob('*.npz')}!=set(saved):raise RuntimeError('落盘身份集合错误')
            for ident,s in saved.items():
                with np.load(dest/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                    if set(z.files)!=set(s['values']) or not all(np.array_equal(z[k],v) for k,v in s['values'].items()):raise RuntimeError('落盘数组错误')
                if json.loads((dest/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))['history']!=s['history']:raise RuntimeError('落盘轨迹错误')
                with (dest/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:
                    if len(list(csv.DictReader(f)))!=len(s['history']):raise RuntimeError('轨迹CSV轮数错误')
            if {p.name for p in (dest/'figures').glob('*.png')}!={k+'_comparison.png' for k in saved}|{'optimization_curves.png'}:raise RuntimeError('结果图集合错误')
        # 显示失败不能留下完成标记；finally统一关闭图窗。
        if runtime['show_plots']:plt.show()
        complete=len(records)==len(jobs);converged=complete and all(r['solver_status']=='converged' for r in records)
        certpass=complete and all(r['certificate_passed'] for r in records)
        report=dict(status='completed',validation_passed=bool(valid),all_reconstructions_converged=converged,
            optimization_validation_passed=bool(valid and converged),certificate_validation_passed=certpass,
            stability_validation_passed=bool(valid and converged and certpass),budget_exhausted=budget_exhausted,
            planned_jobs=len(jobs),completed_jobs=len(records),records=records,config=c,runtime=runtime,backend=backend,
            environment=environment_info(),source_unchanged=True,elapsed_s=time.perf_counter()-start,
            scope='Jeon光学编码复现＋基础重建验证；非论文网络、非双孔径',
            assumptions=['固定continuous器件/三单色/无噪声/单位辐射响应/有限像平面支持；真值仅用于评价。',
                '停止门槛与1%最优解距离理论上界分开验收；界不是真值误差且未作区间舍入包络。',
                '旋转方向对应论文、真实PyCharm点击和GUI仍未验证；不宣称完整论文复现。'])
        if dest:
            paths=['main_stage02e2.py','config_stage02e2.json','optics/stage02e1_source.py','optics/ridge_certificate.py','optics/reconstruction_accelerated.py','optics/imaging.py','optics/stage02_runtime.py','optics/runutil.py','main_stage02b.py']
            write_json(dest/'source_manifest.json',{k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in paths})
            with (dest/'metrics/evaluation.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['identity','status','iterations','alpha','cube_error','SAM_deg','data_residual','relative_distance_bound','certificate_passed','baseline_relative_change'])
                for r in records:w.writerow([r['identity'],r['solver_status'],r['iterations'],r['alpha'],r['evaluation']['cube_relative_l2'],r['evaluation']['sam_mean_deg'],r['final_optimization']['residual_relative'],r['certificate']['distance_upper_bound_relative'],r['certificate_passed'],r['baseline_reconstruction_relative_change']])
            lines=['# 02E-2弱正则化数值稳定性报告','','Jeon光学编码复现＋基础重建验证。','',
                '|场景|状态|轮数|cube误差|距离相对上界|1%界通过|恢复变化|','|---|---|---:|---:|---:|---|---:|']
            for r in records:lines.append('|%s|%s|%d|%.7g|%.7g|%s|%.7g|'%(r['scene'],r['solver_status'],r['iterations'],r['evaluation']['cube_relative_l2'],r['certificate']['distance_upper_bound_relative'],r['certificate_passed'],r['baseline_reconstruction_relative_change']))
            lines+=['','## 限制','']+['- '+s for s in report['assumptions']]
            (dest/'report_stage02e2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            write_json(dest/'progress.json',dict(status='completed',planned_jobs=len(jobs),completed_jobs=len(records),stability_validation_passed=report['stability_validation_passed']))
            # 最终标记放在所有交付文件写入、重读和显示完成之后。
            write_json(dest/'metrics/validation.json',report)
        print(json.dumps(dict(status='completed',optimization_validation_passed=report['optimization_validation_passed'],certificate_validation_passed=certpass,stability_validation_passed=report['stability_validation_passed'],run_dir=str(dest) if dest else None,elapsed_s=report['elapsed_s']),ensure_ascii=False),flush=True)
        return report
    except Exception:
        if dest:write_json(dest/'failed.json',dict(status='failed',completed_jobs=len(records),traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figures:plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_stage02e2.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true')
    a=p.parse_args();c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    r=run(c,a.no_save,a.show_plots)
    return 0 if r['stability_validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
