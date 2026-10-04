# -*- coding: utf-8 -*-
"""02E-1固定单通道模型的α敏感性，CPU/PyCharm无参数入口。"""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from optics.imaging import SpectralImager,operator_checks
from optics.reconstruction import evaluate_cube
from optics.reconstruction_accelerated import solve_accelerated_ridge,validate_accelerated
from optics.ridge_certificate import ridge_certificate
from optics.stage02d3_source import load_accelerated_source
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir
ROOT=Path(__file__).resolve().parent

def write_json(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_config(c):
    validate_accelerated(c['solver'])
    for name in ['alpha_factors']:
        values=c[name]
        if not isinstance(values,list) or not values or any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in values) or len(set(values))!=len(values) or 1. not in values:
            raise ValueError('α倍数须唯一有限正值且包含1对照')
    strict=c['stricter']
    if not isinstance(strict['enabled'],bool):raise ValueError('更严诊断开关须布尔值')
    if strict['scene'] not in ['coincident_points','separated_points','lines_and_square']:raise ValueError('诊断场景不存在')
    if not strict['alpha_factors'] or len(set(strict['alpha_factors']))!=len(strict['alpha_factors']) or any(v not in c['alpha_factors'] for v in strict['alpha_factors']):raise ValueError('诊断α须属于主扫描')
    if not 0<strict['gradient_tolerance']<c['solver']['gradient_tolerance'] or not 0<strict['objective_tolerance']<c['solver']['objective_tolerance']:raise ValueError('诊断门槛须更严格')
    for name in ['certificate_relative_threshold','budget_seconds','baseline_relative_tolerance']:
        if isinstance(c[name],bool) or not np.isfinite(c[name]) or c[name]<=0:raise ValueError(name+'须正有限')
    if isinstance(c['check_seed'],bool) or not isinstance(c['check_seed'],int) or c['check_seed']<0:raise ValueError('检查种子须非负整数')
    if not np.isfinite(c['evaluation']['spectrum_norm_threshold']) or c['evaluation']['spectrum_norm_threshold']<=0:raise ValueError('SAM阈值须正有限')
    if any(not np.isfinite(v) or v<=0 for v in c['thresholds'].values()):raise ValueError('算子容差须正有限')
    if isinstance(c['plots']['dpi'],bool) or not isinstance(c['plots']['dpi'],int) or c['plots']['dpi']<=0:raise ValueError('dpi须正整数')

def run(c,no_save=False,show_plots=False):
    validate_config(c)
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02e1') if runtime['save_results'] else None
    start=time.perf_counter();figures=[];records=[];saved={}
    try:
        items,evidence=load_accelerated_source(resolve_project_path(c['source_run'],ROOT),ROOT)
        for key in ['gradient_tolerance','objective_tolerance']:
            if c['solver'][key]!=evidence['source_config']['solver'][key]:raise ValueError('主扫描门槛须与收敛基线一致')
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime))
            write_json(dest/'source_evidence.json',evidence)
        jobs=[(item,factor,False) for item in items for factor in c['alpha_factors']]
        if c['stricter']['enabled']:
            jobs += [(item,factor,True) for item in items if item['scene']==c['stricter']['scene'] for factor in c['stricter']['alpha_factors']]
        checks={};operator_valid=True;budget_exhausted=False
        for item,factor,strict in jobs:
            # 预算在完整实验之间检查，单组最多max_iterations轮，不遗失已完成结果。
            if records and time.perf_counter()-start>c['budget_seconds']:
                budget_exhausted=True;break
            v=item['values'];ident=item['identity']+'_alpha_'+format(factor,'.8g').replace('.','p')+('_strict' if strict else '')
            op=SpectralImager(v['kernels'],list(v['truth'].shape[1:]),float(v['pitch_m']),v['response'],v['crop_indices'].tolist())
            if item['identity'] not in checks:
                check=operator_checks(op,c['check_seed'],v['measurement']);checks[item['identity']]=check
                operator_valid=operator_valid and check['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and check['gradient_relative_error']<=c['thresholds']['gradient_relative']
            solver=dict(c['solver'])
            if strict:
                for key in ['gradient_tolerance','objective_tolerance']:solver[key]=c['stricter'][key]
            alpha=float(v['alpha'])*factor
            result=solve_accelerated_ridge(op,v['measurement'],alpha,solver,item['L_initial'])
            x=result['reconstruction'];prediction=op.forward(x);residual=prediction-v['measurement']
            certificate=ridge_certificate(op,v['measurement'],x,alpha)
            evaluation=evaluate_cube(v['truth'],x,c['evaluation']['spectrum_norm_threshold'])
            monotone=all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(result['history'],result['history'][1:]))
            if not monotone or np.any(x<0) or not np.all(np.isfinite(x)):raise RuntimeError('轨迹或非负恢复异常')
            diff=float(np.linalg.norm(x-v['reconstruction'])/max(np.linalg.norm(v['reconstruction']),1e-300))
            baseline_match=None if factor!=1 or strict or result['status']!='converged' else bool(diff<=c['baseline_relative_tolerance'])
            if baseline_match is False:raise RuntimeError('α=1基线重现失败')
            truth_power=float(np.sum(v['truth']));recovered_power=float(np.sum(x))
            rec=dict(identity=ident,scene=item['scene'],device='continuous',mode='crop',alpha_factor=float(factor),
                alpha=alpha,baseline_alpha=float(v['alpha']),stricter=strict,solver_config=solver,
                solver_status=result['status'],iterations=result['iterations'],elapsed_s=result['elapsed_s'],
                final_optimization=result['history'][-1],certificate=certificate,evaluation=evaluation,
                recovered_to_truth_total_power=recovered_power/truth_power if truth_power>0 else None,
                baseline_reconstruction_relative_difference=diff,baseline_match=baseline_match,
                certificate_relative_threshold=c['certificate_relative_threshold'],
                certified_relative_distance_small=bool(certificate['distance_upper_bound_relative'] is not None and certificate['distance_upper_bound_relative']<=c['certificate_relative_threshold']))
            values=dict(v,reconstruction=x,prediction=prediction,residual=residual,alpha=np.asarray(alpha),alpha_factor=np.asarray(factor))
            saved[ident]=dict(values=values,history=result['history']);records.append(rec)
            if dest:
                np.savez_compressed(dest/'arrays'/(ident+'.npz'),**values)
                write_json(dest/'metrics'/(ident+'_optimization.json'),dict(status=result['status'],alpha=alpha,history=result['history']))
                with (dest/'metrics'/(ident+'_optimization.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(result['history'][0]));w.writeheader();w.writerows(result['history'])
                write_json(dest/'progress.json',dict(status='running',completed_jobs=len(records),planned_jobs=len(jobs),last_record=rec))
            print('%s: %s, %d轮, cube误差=%.4g, 最优解距离相对上界=%s'%(ident,result['status'],result['iterations'],evaluation['cube_relative_l2'],certificate['distance_upper_bound_relative']),flush=True)
        # 每个场景所有倍数采用相同原始功率色标，避免显示归一化掩盖幅值偏差。
        for item in items:
            group=[r for r in records if r['scene']==item['scene'] and not r['stricter']]
            if not group:continue
            truth=item['values']['truth'];vmax=max(float(truth.max()),max(float(saved[r['identity']]['values']['reconstruction'].max()) for r in group))
            fig,axes=plt.subplots(3,1+len(group),figsize=(3*(1+len(group)),8),squeeze=False);figures.append(fig)
            for band in range(3):
                axes[band,0].imshow(truth[band],vmin=0,vmax=vmax,origin='lower');axes[band,0].set_title('真值 %.0f nm'%(item['values']['wavelengths_m'][band]*1e9))
                for col,r in enumerate(group,1):
                    im=axes[band,col].imshow(saved[r['identity']]['values']['reconstruction'][band],vmin=0,vmax=vmax,origin='lower')
                    axes[band,col].set_title('α×%g %s'%(r['alpha_factor'],r['solver_status']),fontsize=9)
                for ax in axes[band]:ax.set_xlabel('x/像元');ax.set_ylabel('y/像元')
            fig.suptitle(item['scene']+'：相同原始相对功率色标');fig.subplots_adjust(right=.91,wspace=.4,hspace=.45)
            fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,label='相对功率')
            if dest:fig.savefig(dest/'figures'/(item['scene']+'_alpha_cubes.png'),dpi=c['plots']['dpi'])
        trend,axes=plt.subplots(2,2,figsize=(12,8));figures.append(trend)
        metrics=[('cube_relative_l2','cube相对L2误差'),('residual_relative','测量相对残差'),('power','恢复/真值总功率'),('certificate','最优解距离相对上界')]
        for ax,(key,title) in zip(axes.ravel(),metrics):
            for item in items:
                group=sorted([r for r in records if r['scene']==item['scene'] and not r['stricter']],key=lambda r:r['alpha_factor'])
                def value(r):
                    if key=='residual_relative':return r['final_optimization'][key]
                    if key=='power':return r['recovered_to_truth_total_power']
                    if key=='certificate':return r['certificate']['distance_upper_bound_relative']
                    return r['evaluation'][key]
                ax.plot([r['alpha_factor'] for r in group],[value(r) for r in group],label=item['scene'])
                for r in group:
                    ax.scatter(r['alpha_factor'],value(r),marker='o' if r['solver_status']=='converged' else 'x')
            ax.set_xscale('log');ax.set_xlabel('相对于固定基线的α倍数');ax.set_title(title);ax.legend(fontsize=7);ax.grid(True)
        trend.suptitle('圆点=原门槛收敛，叉号=达到迭代上限；未按真值挑选α');trend.tight_layout()
        if dest:trend.savefig(dest/'figures'/'alpha_trends.png',dpi=c['plots']['dpi'])
        comparisons=[]
        for strict in [r for r in records if r['stricter']]:
            normal=next(r for r in records if not r['stricter'] and r['scene']==strict['scene'] and r['alpha_factor']==strict['alpha_factor'])
            x=saved[normal['identity']]['values']['reconstruction'];xs=saved[strict['identity']]['values']['reconstruction']
            comparisons.append(dict(normal_identity=normal['identity'],strict_identity=strict['identity'],
                reconstruction_relative_change=float(np.linalg.norm(xs-x)/max(np.linalg.norm(xs),1e-300)),
                cube_error_change=strict['evaluation']['cube_relative_l2']-normal['evaluation']['cube_relative_l2'],
                both_converged=normal['solver_status']==strict['solver_status']=='converged'))
        for source in evidence['sources']:
            if tree_sha(Path(source['path']))!=source['sha256']:raise RuntimeError('运行期间只读来源变化')
        if dest:
            if {p.stem for p in (dest/'arrays').glob('*.npz')}!=set(saved):raise RuntimeError('落盘数组身份集合不匹配')
            for ident,s in saved.items():
                with np.load(dest/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                    if set(z.files)!=set(s['values']) or not all(np.array_equal(z[k],value) for k,value in s['values'].items()):raise RuntimeError('落盘数组内容错误')
                h=json.loads((dest/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))['history']
                if h!=s['history']:raise RuntimeError('落盘轨迹错误')
                with (dest/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:
                    if len(list(csv.DictReader(f)))!=len(h):raise RuntimeError('CSV轮数错误')
            expected_figures={item['scene']+'_alpha_cubes.png' for item in items if any(r['scene']==item['scene'] and not r['stricter'] for r in records)}|{'alpha_trends.png'}
            if {p.name for p in (dest/'figures').glob('*.png')}!=expected_figures:raise RuntimeError('结果图集合错误')
        all_converged=all(r['solver_status']=='converged' for r in records) and len(records)==len(jobs)
        report=dict(status='completed',validation_passed=bool(operator_valid),optimization_validation_passed=bool(operator_valid and all_converged),
            all_reconstructions_converged=all_converged,budget_exhausted=budget_exhausted,planned_jobs=len(jobs),completed_jobs=len(records),
            records=records,stricter_comparisons=comparisons,operator_checks=checks,source_unchanged=True,
            config=c,runtime=runtime,backend=backend,environment=environment_info(),elapsed_s=time.perf_counter()-start,
            scope='Jeon光学编码复现＋基础重建验证；固定continuous单孔径，非论文网络',
            assumptions=['无噪声、三单色、单位辐射响应、固定像平面支持；未按真值挑选α。',
                '原数值停止门槛不保证严格最优解距离很小；强凸证书可能保守。',
                '所有参数及iteration_limit均报告；没有验证真实相机、原论文网络或双孔径。',
                '旋转方向对应论文仍未解决；真实PyCharm点击与GUI仍未人工验证。'])
        if dest:
            paths=['main_stage02e1.py','config_stage02e1.json','optics/ridge_certificate.py','optics/stage02d3_source.py','optics/reconstruction_accelerated.py','optics/imaging.py']
            write_json(dest/'source_manifest.json',{k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest() for k in paths})
            with (dest/'metrics/evaluation.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['identity','factor','actual_alpha','strict','status','iterations','data_residual','cube_error','SAM_deg','total_power_ratio','relative_distance_bound'])
                for r in records:w.writerow([r['identity'],r['alpha_factor'],r['alpha'],r['stricter'],r['solver_status'],r['iterations'],r['final_optimization']['residual_relative'],r['evaluation']['cube_relative_l2'],r['evaluation']['sam_mean_deg'],r['recovered_to_truth_total_power'],r['certificate']['distance_upper_bound_relative']])
            lines=['# 02E-1固定模型α敏感性运行报告','','Jeon光学编码复现＋基础重建验证；不是原论文网络。','',
                '|实验|α倍数|状态|轮数|cube误差|数据残差|功率比|最优解距离相对上界|','|---|---:|---|---:|---:|---:|---:|---:|']
            for r in records:lines.append('|%s|%g|%s|%d|%.6g|%.6g|%.6g|%s|'%(r['identity'],r['alpha_factor'],r['solver_status'],r['iterations'],r['evaluation']['cube_relative_l2'],r['final_optimization']['residual_relative'],r['recovered_to_truth_total_power'],r['certificate']['distance_upper_bound_relative']))
            lines += ['','## 更严门槛诊断','',json.dumps(comparisons,ensure_ascii=False,indent=2),'','## 限制','']+['- '+s for s in report['assumptions']]
            (dest/'report_stage02e1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            write_json(dest/'metrics/validation.json',report)
            write_json(dest/'progress.json',dict(status='completed',completed_jobs=len(records),planned_jobs=len(jobs),optimization_validation_passed=report['optimization_validation_passed']))
        print(json.dumps(dict(status='completed',optimization_validation_passed=report['optimization_validation_passed'],all_reconstructions_converged=all_converged,run_dir=str(dest) if dest else None,elapsed_s=report['elapsed_s']),ensure_ascii=False),flush=True)
        if runtime['show_plots']:plt.show()
        return report
    except Exception:
        if dest:write_json(dest/'failed.json',dict(status='failed',traceback=traceback.format_exc(),completed_jobs=len(records)))
        raise
    finally:
        for fig in figures:plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_stage02e1.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true')
    args=p.parse_args();c=json.loads(resolve_project_path(args.config,ROOT).read_text(encoding='utf-8'))
    r=run(c,args.no_save,args.show_plots)
    return 0 if r['optimization_validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
