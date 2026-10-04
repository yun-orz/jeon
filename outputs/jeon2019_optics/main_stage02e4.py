# -*- coding: utf-8 -*-
"""02E-4弱α数值稳定性复核，Windows/PyCharm CPU无参数入口。"""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from optics.imaging import SpectralImager,operator_checks
from optics.reconstruction import evaluate_cube
from optics.reconstruction_accelerated import solve_accelerated_ridge,validate_accelerated
from optics.ridge_certificate import ridge_certificate
from optics.stage02e3_source import load_common_alpha_source,reuse_compatible
from optics.comparison_uncertainty import error_difference_interval
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir
ROOT=Path(__file__).resolve().parent

def write_json(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_config(c):
    validate_accelerated(c['solver'])
    if not c['scenes'] or len(set(c['scenes']))!=len(c['scenes']) or any(s not in ['coincident_points','separated_points','lines_and_square'] for s in c['scenes']):raise ValueError('本阶段场景身份不合法')
    if not c['devices'] or len(set(c['devices']))!=len(c['devices']) or any(d not in ['continuous','nearest_depth'] for d in c['devices']):raise ValueError('本阶段器件身份不合法')
    if c['alpha_strategy']!='continuous_baseline_relative' or isinstance(c['reference_alpha_factor'],bool) or c['reference_alpha_factor']!=.01:raise ValueError('本批共同α固定为连续器件D3基线×0.01')
    if not isinstance(c['reuse_continuous'],bool):raise ValueError('复用开关须布尔值')
    for key in ['certificate_relative_threshold','budget_seconds']:
        if isinstance(c[key],bool) or not np.isfinite(c[key]) or c[key]<=0:raise ValueError(key+'须正有限')
    if isinstance(c['check_seed'],bool) or not isinstance(c['check_seed'],int) or c['check_seed']<0:raise ValueError('种子须非负整数')
    if not np.isfinite(c['evaluation']['spectrum_norm_threshold']) or c['evaluation']['spectrum_norm_threshold']<=0:raise ValueError('SAM阈值须正有限')
    if any(not np.isfinite(v) or v<=0 for v in c['thresholds'].values()):raise ValueError('算子容差须正有限')
    if isinstance(c['plots']['dpi'],bool) or not isinstance(c['plots']['dpi'],int) or c['plots']['dpi']<=0:raise ValueError('dpi须正整数')

def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02e4') if runtime['save_results'] else None
    start=time.perf_counter();records=[];saved={};figures=[]
    try:
        items,evidence=load_common_alpha_source(resolve_project_path(c['source_run'],ROOT),ROOT)
        common_alpha=evidence['common_alpha']
        for key in ['gradient_tolerance','objective_tolerance']:
            if c['solver'][key]>evidence['source_config']['solver'][key]:raise ValueError('本阶段停止门槛不能松于E3来源')
        jobs=[next(item for item in items if item['scene']==scene and item['device']==device) for scene in c['scenes'] for device in c['devices']]
        if dest:write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'source_evidence.json',evidence)
        valid=True;budget_exhausted=False
        for item in jobs:
            # 完整实验之间检查预算；不会删除或丢弃前组结果。
            if records and time.perf_counter()-start>c['budget_seconds']:budget_exhausted=True;break
            v=item['values'];ident=item['identity']+'_common_alpha'
            op=SpectralImager(v['kernels'],list(v['truth'].shape[1:]),float(v['pitch_m']),v['response'],v['crop_indices'].tolist())
            check=operator_checks(op,c['check_seed'],v['measurement'])
            valid=valid and check['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and check['gradient_relative_error']<=c['thresholds']['gradient_relative']
            alpha=common_alpha;reused=reuse_compatible(item,c,alpha)
            if reused:
                # 复用历史轨迹时保留其原始时间，不伪装成本轮重新迭代。
                result=dict(reconstruction=v['reconstruction'].copy(),history=item['history'],status=item['record']['solver_status'],iterations=item['record']['iterations'],elapsed_s=0.,initialization='zeros')
            else:
                result=solve_accelerated_ridge(op,v['measurement'],alpha,c['solver'],item['L_initial'])
            x=result['reconstruction'];prediction=op.forward(x);residual=prediction-v['measurement']
            h=result['history'];cert=ridge_certificate(op,v['measurement'],x,alpha)
            if np.any(x<0) or not np.all(np.isfinite(x)) or not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:])):raise RuntimeError('恢复非负性或轨迹错误')
            ev=evaluate_cube(v['truth'],x,c['evaluation']['spectrum_norm_threshold'])
            old=item['record'];certified=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=c['certificate_relative_threshold']
            rec=dict(identity=ident,scene=item['scene'],device=item['device'],mode='crop',alpha_factor=(.01 if item['device']=='continuous' else alpha/old['baseline_alpha']),alpha=alpha,
                baseline_alpha=old['baseline_alpha'],solver_config=c['solver'],solver_status=result['status'],iterations=result['iterations'],elapsed_s=result['elapsed_s'],
                execution='reused_verified' if reused else 'computed',new_iterations=0 if reused else result['iterations'],
                source_elapsed_s=old['elapsed_s'],reuse_source_run=item['source_run'] if reused else None,reuse_source_identity=item['identity'] if reused else None,
                initialization=result['initialization'],operator_checks=check,final_optimization=h[-1],certificate=cert,
                certificate_passed=bool(certified),certificate_relative_threshold=c['certificate_relative_threshold'],evaluation=ev,
                recovered_to_truth_total_power=float(np.sum(x)/np.sum(v['truth'])),baseline_record=old,
                baseline_reconstruction_relative_change=float(np.linalg.norm(x-v['reconstruction'])/max(np.linalg.norm(x),1e-300)),
                baseline_weak_alpha_objective_decrease=.5*float(np.sum(v['residual']**2))+.5*alpha*float(np.sum(v['reconstruction']**2))-h[-1]['objective'],
                baseline_cube_error_change=ev['cube_relative_l2']-old['evaluation']['cube_relative_l2'])
            rec['previous_relative_alpha_result']=dict(alpha=old['alpha'],cube_error=old['evaluation']['cube_relative_l2'],certificate=old['certificate'])
            values=dict(v,reconstruction=x,prediction=prediction,residual=residual,alpha=np.asarray(alpha),alpha_factor=np.asarray(rec['alpha_factor']))
            saved[ident]=dict(values=values,history=h);records.append(rec)
            if dest:
                np.savez_compressed(dest/'arrays'/(ident+'.npz'),**values)
                write_json(dest/'metrics'/(ident+'_optimization.json'),dict(alpha=alpha,status=result['status'],execution=rec['execution'],history_origin_run=item['source_run'] if reused else None,history_origin_identity=item['identity'] if reused else None,history=h))
                with (dest/'metrics'/(ident+'_optimization.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(h[0]));w.writeheader();w.writerows(h)
                write_json(dest/'progress.json',dict(status='running',completed_jobs=len(records),planned_jobs=len(jobs),last_record=rec))
            print('%s [%s]: %s, %d轮, cube=%.6g, 最优解距离相对上界=%.6g, 1%%界通过=%s'%(ident,rec['execution'],result['status'],result['iterations'],ev['cube_relative_l2'],cert['distance_upper_bound_relative'],certified),flush=True)
            # 真值、旧结果、新结果和绝对误差使用统一原始功率色标。
            fig,axes=plt.subplots(3,4,figsize=(12,8));figures.append(fig)
            vmax=max(float(v['truth'].max()),float(v['reconstruction'].max()),float(x.max()))
            for band in range(3):
                for col,(title,arr) in enumerate([('真值',v['truth'][band]),('02E-3相对α',v['reconstruction'][band]),('02E-4',x[band]),('新绝对误差',abs(x[band]-v['truth'][band]))]):
                    im=axes[band,col].imshow(arr,vmin=0,vmax=vmax,origin='lower');axes[band,col].set_title(title+' %.0f nm'%(v['wavelengths_m'][band]*1e9));axes[band,col].set_xlabel('x/像元');axes[band,col].set_ylabel('y/像元')
            fig.suptitle(item['scene']+' '+result['status']);fig.subplots_adjust(right=.91,wspace=.4,hspace=.45);fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,label='相对功率')
            if dest:fig.savefig(dest/'figures'/(ident+'_comparison.png'),dpi=c['plots']['dpi'])
        pairs=[];paired_figures=set()
        for scene in c['scenes']:
            group=[r for r in records if r['scene']==scene]
            if {r['device'] for r in group}!={'continuous','nearest_depth'}:continue
            cont=next(r for r in group if r['device']=='continuous');near=next(r for r in group if r['device']=='nearest_depth')
            cv=saved[cont['identity']]['values'];nv=saved[near['identity']]['values']
            pairs.append(dict(scene=scene,continuous_alpha=cont['alpha'],nearest_depth_alpha=near['alpha'],
                continuous_cube_error=cont['evaluation']['cube_relative_l2'],nearest_depth_cube_error=near['evaluation']['cube_relative_l2'],
                both_stable=bool(all(r['solver_status']=='converged' and r['certificate_passed'] for r in group)),
                **error_difference_interval(cont['evaluation']['cube_relative_l2'],near['evaluation']['cube_relative_l2'],cont['certificate']['distance_upper_bound'],near['certificate']['distance_upper_bound'],float(np.linalg.norm(cv['truth'])))))
            fig,axes=plt.subplots(3,5,figsize=(15,8));figures.append(fig)
            truth=cv['truth'];xc=cv['reconstruction'];xn=nv['reconstruction'];vmax=max(float(truth.max()),float(xc.max()),float(xn.max()))
            for band in range(3):
                for col,(title,arr) in enumerate([('真值',truth[band]),('连续高度',xc[band]),('最近深度级',xn[band]),('连续绝对误差',abs(xc[band]-truth[band])),('量化绝对误差',abs(xn[band]-truth[band]))]):
                    im=axes[band,col].imshow(arr,vmin=0,vmax=vmax,origin='lower');axes[band,col].set_title(title+' %.0f nm'%(cv['wavelengths_m'][band]*1e9));axes[band,col].set_xlabel('x/像元');axes[band,col].set_ylabel('y/像元')
            fig.suptitle(scene+'：固定共同实际α');fig.subplots_adjust(right=.91,wspace=.4,hspace=.45);fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,label='相对功率')
            filename=scene+'_device_pair.png';paired_figures.add(filename)
            if dest:fig.savefig(dest/'figures'/filename,dpi=c['plots']['dpi'])
        curve,axs=plt.subplots(1,2,figsize=(12,4));figures.append(curve)
        for rec in records:
            h=saved[rec['identity']]['history']
            axs[0].semilogy([a['iteration'] for a in h],[max(a['projected_gradient_relative'],1e-300) for a in h],label=rec['device']+' '+rec['scene'])
            axs[1].semilogy([a['iteration'] for a in h],[max(a['objective_relative_change'],1e-300) for a in h],label=rec['device']+' '+rec['scene'])
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
            if {p.name for p in (dest/'figures').glob('*.png')}!={k+'_comparison.png' for k in saved}|{'optimization_curves.png'}|paired_figures:raise RuntimeError('结果图集合错误')
        # 显示失败不能留下完成标记；finally统一关闭图窗。
        if runtime['show_plots']:plt.show()
        complete=len(records)==len(jobs);converged=complete and all(r['solver_status']=='converged' for r in records)
        certpass=complete and all(r['certificate_passed'] for r in records)
        report=dict(status='completed',validation_passed=bool(valid),all_reconstructions_converged=converged,
            optimization_validation_passed=bool(valid and converged),certificate_validation_passed=certpass,
            stability_validation_passed=bool(valid and converged and certpass),budget_exhausted=budget_exhausted,
            planned_jobs=len(jobs),completed_jobs=len(records),records=records,config=c,runtime=runtime,backend=backend,
            environment=environment_info(),source_unchanged=True,elapsed_s=time.perf_counter()-start,
            common_alpha=common_alpha,reused_count=sum(r['execution']=='reused_verified' for r in records),computed_count=sum(r['execution']=='computed' for r in records),comparison_strategy=evidence['comparison_strategy'],pairs=pairs,scope='Jeon光学编码复现＋基础重建验证；非论文网络、非双孔径',
            assumptions=['连续与最近深度级完整单孔径器件/三单色/无噪声/单位辐射响应/有限支持；真值仅用于评价。',
                evidence['comparison_strategy'],
                '停止门槛与1%最优解距离理论上界分开验收；界不是真值误差且未作区间舍入包络。',
                '旋转方向对应论文、真实PyCharm点击和GUI仍未验证；不宣称完整论文复现。'])
        if dest:
            paths=['main_stage02e4.py','config_stage02e4.json','optics/stage02e3_source.py','optics/comparison_uncertainty.py','optics/ridge_certificate.py','optics/reconstruction_accelerated.py','optics/imaging.py','optics/stage02_runtime.py','optics/runutil.py','main_stage02b.py']
            write_json(dest/'source_manifest.json',{k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in paths})
            with (dest/'metrics/evaluation.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['identity','status','iterations','alpha','cube_error','SAM_deg','data_residual','relative_distance_bound','certificate_passed','baseline_relative_change','execution','new_iterations'])
                for r in records:w.writerow([r['identity'],r['solver_status'],r['iterations'],r['alpha'],r['evaluation']['cube_relative_l2'],r['evaluation']['sam_mean_deg'],r['final_optimization']['residual_relative'],r['certificate']['distance_upper_bound_relative'],r['certificate_passed'],r['baseline_reconstruction_relative_change'],r['execution'],r['new_iterations']])
            lines=['# 02E-4弱正则化数值稳定性报告','','Jeon光学编码复现＋基础重建验证。','',
                '|场景|状态/执行|历史或新轮数|cube误差|距离相对上界|1%界通过|相对E3变化|','|---|---|---:|---:|---:|---|---:|']
            for r in records:lines.append('|%s|%s|%d|%.7g|%.7g|%s|%.7g|'%(r['device']+' '+r['scene'],r['solver_status']+'/'+r['execution'],r['iterations'],r['evaluation']['cube_relative_l2'],r['certificate']['distance_upper_bound_relative'],r['certificate_passed'],r['baseline_reconstruction_relative_change']))
            lines+=['','比较策略：'+evidence['comparison_strategy'],'','## 器件对照','',json.dumps(pairs,ensure_ascii=False,indent=2),'','## 限制','']+['- '+s for s in report['assumptions']]
            (dest/'report_stage02e4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            write_json(dest/'progress.json',dict(status='completed',planned_jobs=len(jobs),completed_jobs=len(records),stability_validation_passed=report['stability_validation_passed']))
            # 最终标记放在所有交付文件写入、重读和显示完成之后。
            write_json(dest/'metrics/validation.json',report)
        print(json.dumps(dict(status='completed',optimization_validation_passed=report['optimization_validation_passed'],certificate_validation_passed=certpass,stability_validation_passed=report['stability_validation_passed'],reused_count=report['reused_count'],computed_count=report['computed_count'],run_dir=str(dest) if dest else None,elapsed_s=report['elapsed_s']),ensure_ascii=False),flush=True)
        return report
    except Exception:
        if dest:write_json(dest/'failed.json',dict(status='failed',completed_jobs=len(records),traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figures:plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_stage02e4.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true')
    a=p.parse_args();c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    r=run(c,a.no_save,a.show_plots)
    return 0 if r['stability_validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
