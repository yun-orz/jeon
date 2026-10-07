# -*- coding: utf-8 -*-
"""02F-2电子响应及固定权重的非负岭基础重建，CPU无参数入口。"""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from optics.electron_reconstruction import make_electron_operator,reconstruct_electrons
from optics.stage02f1_source import load_electron_source
from optics.imaging import operator_checks,direct_shift_sum
from optics.reconstruction import evaluate_cube,squared_norm
from optics.reconstruction_accelerated import validate_accelerated
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir
ROOT=Path(__file__).resolve().parent

def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_config(c):
    for key,allowed in [('devices',['continuous','nearest_depth']),('scenes',['coincident_points','separated_points','lines_and_square']),('methods',['unweighted','fixed_weighted'])]:
        if not c[key] or len(set(c[key]))!=len(c[key]) or any(v not in allowed for v in c[key]):raise ValueError('实验身份不合法')
    for value in [c['gain_e_per_relative_power'],c['budget_seconds'],c['certificate_relative_threshold'],c['evaluation']['spectrum_norm_threshold'],*c['thresholds'].values()]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('增益/预算/容差须正有限')
    r=c['reconstruction'];validate_accelerated(r['solver'])
    for value in [r['alpha_factor'],r['variance_floor_e2'],r['initial_L_factor'],r['power']['power_tolerance']]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('重建标度须正有限')
    for value,minimum in [(c['check_seed'],0),(r['power']['seed'],0),(r['power']['power_iterations'],1),(c['plots']['dpi'],1)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError('种子/次数/dpi须合法整数')

def optimization_check(op,target,x,alpha,last,seed):
    residual=op.forward(x)-target;gradient=op.adjoint(residual)+alpha*x
    pg=np.sqrt(squared_norm(last['L']*(x-np.maximum(0,x-gradient/last['L']))))/max(np.sqrt(squared_norm(op.adjoint(target))),1e-300)
    objective=.5*squared_norm(residual)+.5*alpha*squared_norm(x)
    direction=np.random.default_rng(seed).normal(size=x.shape);direction/=np.sqrt(squared_norm(direction));eps=1e-3*max(1.,np.sqrt(squared_norm(x)))
    def f(z):return .5*squared_norm(op.forward(z)-target)+.5*alpha*squared_norm(z)
    fd=(f(x+eps*direction)-f(x-eps*direction))/(2*eps);analytic=float(np.sum(gradient*direction))
    return dict(objective_recomputed=objective,projected_gradient_recomputed=float(pg),ridge_gradient_relative_error=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12))

def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02f2') if runtime['save_results'] else None
    start=time.perf_counter();records=[];saved={};figs=[];budget=False;valid=True
    try:
        items,evidence=load_electron_source(resolve_project_path(c['source_run'],ROOT),ROOT)
        jobs=[i for i in items if i['device'] in c['devices'] and i['scene'] in c['scenes'] and float(i['values']['gain_e_per_relative_power'])==c['gain_e_per_relative_power']]
        if len(jobs)!=len(c['devices'])*len(c['scenes']):raise ValueError('来源缺所选G或场景')
        planned=len(jobs)*len(c['methods'])
        if dest:write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'source_evidence.json',evidence)
        for item in jobs:
            v=item['values'];base=make_electron_operator(v['kernels'],list(v['truth'].shape[1:]),float(v['pitch_m']),v['response'],v['wavelengths_m'],v['quantum_efficiency'],float(v['reference_wavelength_m']),float(v['gain_e_per_relative_power']),v['crop_indices'].tolist())
            measurement=v['noisy_measurement_e'];background=float(v['background_e']);sigma=float(v['read_sigma_e'])
            check=operator_checks(base,c['check_seed'],measurement-background)
            direct=direct_shift_sum(v['truth'],v['kernels'],base.response)[base.slices]
            direct_error=np.sqrt(squared_norm(direct-base.forward(v['truth'])))/max(np.sqrt(squared_norm(direct)),1e-300)
            source_error=np.sqrt(squared_norm(base.forward(v['truth'])-v['mean_signal_raw_e']))/max(np.sqrt(squared_norm(v['mean_signal_raw_e'])),1e-300)
            for method in c['methods']:
                if time.perf_counter()-start>c['budget_seconds']:budget=True;break
                # 真值只在来源物理核验和求解后的评价中使用，不进入重建接口。
                fit=reconstruct_electrons(base,measurement,background,sigma,method,c['reconstruction']);result=fit['result'];x=result['reconstruction'];op=fit['operator'];target=fit['target'];alpha=fit['alpha']
                weighted_check=operator_checks(op,c['check_seed'],target);last=result['history'][-1];optcheck=optimization_check(op,target,x,alpha,last,c['check_seed']);cert=fit['certificate']
                numerical=bool(check['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and check['gradient_relative_error']<=c['thresholds']['gradient_relative'] and weighted_check['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and weighted_check['gradient_relative_error']<=c['thresholds']['gradient_relative'] and direct_error<=c['thresholds']['direct_relative'] and source_error<=c['thresholds']['direct_relative'] and optcheck['ridge_gradient_relative_error']<=c['thresholds']['gradient_relative'] and np.isclose(optcheck['objective_recomputed'],last['objective'],rtol=1e-12,atol=0) and np.isclose(optcheck['projected_gradient_recomputed'],last['projected_gradient_relative'],rtol=1e-10,atol=1e-14))
                if np.any(x<0) or not np.all(np.isfinite(x)) or not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(result['history'],result['history'][1:])):raise RuntimeError('非负恢复或目标下降检查失败')
                certificate_pass=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=c['certificate_relative_threshold']
                prediction=base.forward(x)+background;residual=prediction-measurement;identity=item['identity']+'_'+method
                rec=dict(identity=identity,source_identity=item['identity'],device=item['device'],scene=item['scene'],method=method,alpha=alpha,alpha_factor=c['reconstruction']['alpha_factor'],spectral_estimate=fit['spectral_estimate'],solver_status=result['status'],iterations=result['iterations'],elapsed_s=result['elapsed_s'],initialization=result['initialization'],final_optimization=last,optimization_check=optcheck,electron_operator_checks=check,weighted_operator_checks=weighted_check,direct_relative_error=direct_error,source_mean_relative_error=source_error,numerical_validation_passed=numerical,certificate=cert,certificate_passed=bool(certificate_pass),evaluation=evaluate_cube(v['truth'],x,c['evaluation']['spectrum_norm_threshold']),negative_measurement_count=int(np.count_nonzero(measurement<0)),electron_residual_norm=np.sqrt(squared_norm(residual)))
                # 原始电子测量和评价真值明确保存；不把重建称为论文网络。
                values={k:v[k] for k in ['kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','quantum_efficiency','reference_wavelength_m','gain_e_per_relative_power','read_sigma_e','background_e','output_x_m','output_y_m','scene_x_m','scene_y_m','seed']}
                values.update(evaluation_truth=v['truth'],measurement_e=measurement,reconstruction=x,prediction_e=prediction,residual_e=residual,weights=fit['weights'],variance_estimate_e2=fit['variance_estimate_e2'] if fit['variance_estimate_e2'] is not None else np.asarray([]),effective_electron_response=base.response,weighted_target=target,alpha=np.asarray(alpha),L_data_estimate=np.asarray(fit['spectral_estimate']['value']),method=np.asarray(method),measurement_unit=np.asarray('electron'))
                saved[identity]=values;records.append(rec);valid=valid and numerical
                if dest:
                    np.savez_compressed(dest/'arrays'/(identity+'.npz'),**values);write_json(dest/'metrics'/(identity+'_optimization.json'),dict(alpha=alpha,status=result['status'],history=result['history']))
                    with (dest/'metrics'/(identity+'_optimization.csv')).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(last));w.writeheader();w.writerows(result['history'])
                    write_json(dest/'progress.json',dict(status='running',completed_jobs=len(records),planned_jobs=planned))
                print('%s: %s, %d轮, cube=%.6g, 距离相对上界=%.6g'%(identity,result['status'],result['iterations'],rec['evaluation']['cube_relative_l2'],cert['distance_upper_bound_relative']),flush=True)
                fig,axes=plt.subplots(3,3,figsize=(10,8));figs.append(fig);vmax=max(float(v['truth'].max()),float(x.max()))
                extent=[v['scene_x_m'][0]*1e6,v['scene_x_m'][-1]*1e6,v['scene_y_m'][0]*1e6,v['scene_y_m'][-1]*1e6]
                for b in range(3):
                    for col,(title,a) in enumerate([('评价真值',v['truth'][b]),('恢复',x[b]),('绝对误差',abs(x[b]-v['truth'][b]))]):
                        im=axes[b,col].imshow(a,vmin=0,vmax=vmax,origin='lower',extent=extent);axes[b,col].set_title(title+' %.0f nm'%(v['wavelengths_m'][b]*1e9));axes[b,col].set_xlabel('x / μm');axes[b,col].set_ylabel('y / μm')
                fig.suptitle(identity);fig.subplots_adjust(right=.9,wspace=.4,hspace=.5);fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,label='相对波段积分功率')
                if dest:fig.savefig(dest/'figures'/(identity+'.png'),dpi=c['plots']['dpi'])
            if budget:break
        for source in evidence['sources']:
            if tree_sha(Path(source['path']))!=source['sha256']:raise RuntimeError('运行期间旧来源改变')
        if dest:
            for ident,values in saved.items():
                with np.load(dest/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                    if set(z.files)!=set(values) or any(not np.array_equal(z[k],a) for k,a in values.items()):raise RuntimeError('落盘数组不一致')
                rec=next(r for r in records if r['identity']==ident);h=json.loads((dest/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))['history']
                if len(h)!=rec['iterations'] or h[-1]!=rec['final_optimization'] or not (dest/'figures'/(ident+'.png')).is_file():raise RuntimeError('落盘轨迹/图像错误')
                with (dest/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:
                    if len(list(csv.DictReader(f)))!=len(h):raise RuntimeError('CSV迭代数不一致')
            paths=['main_stage02f2.py','config_stage02f2.json','optics/electron_reconstruction.py','optics/stage02f1_source.py','optics/reconstruction_accelerated.py','optics/ridge_certificate.py','optics/imaging.py']
            write_json(dest/'source_manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
            with (dest/'metrics/evaluation.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['identity','method','status','iterations','alpha','cube_error','SAM_deg','electron_residual_norm','certificate_relative','certificate_passed'])
                for r in records:w.writerow([r['identity'],r['method'],r['solver_status'],r['iterations'],r['alpha'],r['evaluation']['cube_relative_l2'],r['evaluation']['sam_mean_deg'],r['electron_residual_norm'],r['certificate']['distance_upper_bound_relative'],r['certificate_passed']])
        if runtime['show_plots']:plt.show()
        complete=len(records)==planned;converged=complete and all(r['solver_status']=='converged' for r in records);certified=complete and all(r['certificate_passed'] for r in records)
        passed=valid and complete and converged and certified
        report=dict(status='completed' if passed else 'incomplete',validation_passed=bool(passed),numerical_validation_passed=bool(valid),all_reconstructions_converged=converged,certificate_validation_passed=certified,budget_exhausted=budget,planned_jobs=planned,completed_jobs=len(records),records=records,config=c,runtime=runtime,backend=backend,environment=environment_info(),source_unchanged=True,elapsed_s=time.perf_counter()-start,scope='Jeon光学编码复现＋合成含噪基础重建验证；非论文网络、非双孔径',assumptions=['G/QE/噪声仍为合成假设，非相机标定。','W从同一次含噪观测固定估计，近似Gaussian权重，有偏且相关；不声称精确似然或卡方置信。','alpha=预设factor*各方法L估计，不是真值最优参数，也不是共同实际alpha。','未按误差改善验收；证书是距正则化最优解的上界，不是真值误差。','论文旋转方向对应、真实GUI/PyCharm手动点击尚未验证。'])
        if dest:write_json(dest/'metrics/validation.json',report)
        print('结果目录：'+str(dest) if dest else '无保存模式：未写结果文件',flush=True);return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figs:plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_stage02f2.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'));return 0 if run(c,a.no_save,a.show_plots)['validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
