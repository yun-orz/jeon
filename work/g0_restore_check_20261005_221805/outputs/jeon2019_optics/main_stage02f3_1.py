# -*- coding: utf-8 -*-
"""02F-3.1冻结W/alpha的理想测量控制，Windows/PyCharm CPU入口。"""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from optics.fixed_target_control import solve_fixed_target,decompose_errors,check_frozen_parameters
from optics.stage02f2_source import load_fixed_control_source
from optics.electron_reconstruction import make_electron_operator
from optics.imaging import operator_checks
from optics.reconstruction import evaluate_cube,squared_norm
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02b import allocate_unique_run_dir
from main_stage02f2 import optimization_check
ROOT=Path(__file__).resolve().parent

def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_config(c):
    for key,allowed in [('devices',['continuous','nearest_depth']),('scenes',['coincident_points','separated_points','lines_and_square']),('methods',['unweighted','fixed_weighted'])]:
        if not c[key] or len(set(c[key]))!=len(c[key]) or any(v not in allowed for v in c[key]):raise ValueError('实验身份非法')
    for value in [c['budget_seconds'],*c['thresholds'].values()]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('预算/容差须正有限')
    for value,minimum in [(c['check_seed'],0),(c['plots']['dpi'],1)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<minimum:raise ValueError('种子/dpi不合法')

def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02f3_1') if runtime['save_results'] else None
    start=time.perf_counter();records=[];saved={};histories={};figs=[];budget=False;valid=True
    try:
        items,evidence=load_fixed_control_source(resolve_project_path(c['source_run'],ROOT),ROOT);source_config=evidence['source_config'];solver=source_config['reconstruction']['solver']
        jobs=[i for i in items if i['device'] in c['devices'] and i['scene'] in c['scenes'] and i['method'] in c['methods']]
        planned=len(c['devices'])*len(c['scenes'])*len(c['methods'])
        if len(jobs)!=planned:raise ValueError('来源缺少选择的控制身份')
        if dest:write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'source_evidence.json',evidence)
        for item in jobs:
            if time.perf_counter()-start>c['budget_seconds']:budget=True;break
            v=item['values'];truth=v['evaluation_truth'];base=make_electron_operator(v['kernels'],list(truth.shape[1:]),float(v['pitch_m']),v['response'],v['wavelengths_m'],v['quantum_efficiency'],float(v['reference_wavelength_m']),float(v['gain_e_per_relative_power']),v['crop_indices'].tolist())
            background=float(v['background_e']);ideal_measurement=base.forward(truth)+background
            source_error=np.sqrt(squared_norm(ideal_measurement-background-item['mean_signal_raw_e']))/max(np.sqrt(squared_norm(item['mean_signal_raw_e'])),1e-300)
            reference=dict(weights=v['weights'],alpha=v['alpha'],L_data_estimate=v['L_data_estimate'])
            fit=solve_fixed_target(base,ideal_measurement,background,v['weights'],float(v['alpha']),float(v['L_data_estimate']),source_config['reconstruction']['initial_L_factor'],solver,reference)
            result=fit['result'];x0=result['reconstruction'];xn=v['reconstruction'];cert=fit['certificate'];delta_n=item['record']['certificate']['distance_upper_bound'];delta_0=cert['distance_upper_bound']
            errors,decomposition=decompose_errors(truth,xn,x0,delta_n,delta_0)
            check=operator_checks(fit['operator'],c['check_seed'],fit['target']);optcheck=optimization_check(fit['operator'],fit['target'],x0,float(v['alpha']),result['history'][-1],c['check_seed'])
            check_frozen_parameters(fit['operator'].weights,result['alpha'],float(v['L_data_estimate']),reference)
            h=result['history'];last=h[-1]
            numerical=bool(source_error<=c['thresholds']['source_mean_relative'] and check['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and check['gradient_relative_error']<=c['thresholds']['gradient_relative'] and optcheck['ridge_gradient_relative_error']<=c['thresholds']['gradient_relative'] and np.isclose(optcheck['objective_recomputed'],last['objective'],rtol=1e-12,atol=0) and np.isclose(optcheck['projected_gradient_recomputed'],last['projected_gradient_relative'],rtol=1e-10,atol=1e-14) and decomposition['vector_identity_relative']<=c['thresholds']['decomposition_relative'] and decomposition['squared_identity_relative']<=c['thresholds']['decomposition_relative'])
            if np.any(x0<0) or not np.all(np.isfinite(x0)) or not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:])):raise RuntimeError('理想控制非负性/轨迹异常')
            certified=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=source_config['certificate_relative_threshold']
            ident=item['identity']+'_ideal_control'
            rec=dict(identity=ident,source_identity=item['identity'],device=item['device'],scene=item['scene'],method=item['method'],alpha=float(v['alpha']),L_data_estimate=float(v['L_data_estimate']),initial_L=source_config['reconstruction']['initial_L_factor']*(float(v['L_data_estimate'])+float(v['alpha'])),solver_config=solver,solver_status=result['status'],iterations=result['iterations'],elapsed_s=result['elapsed_s'],initialization=result['initialization'],execution='computed_ideal_control',noisy_execution='reused_verified',noisy_new_iterations=0,noisy_elapsed_this_run_s=0.,noisy_source_elapsed_s=item['record']['elapsed_s'],noisy_source_iterations=item['record']['iterations'],noisy_certificate=item['record']['certificate'],ideal_certificate=cert,certificate_passed=bool(certified),final_optimization=last,optimization_check=optcheck,operator_checks=check,source_mean_relative_error=source_error,numerical_validation_passed=numerical,frozen_parameters_verified=True,decomposition=decomposition,ideal_evaluation=evaluate_cube(truth,x0,source_config['evaluation']['spectrum_norm_threshold']),noisy_evaluation=item['record']['evaluation'],ideal_measurement_negative_roundoff_count=int(np.count_nonzero(ideal_measurement<0)))
            values={k:v[k] for k in ['evaluation_truth','kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','quantum_efficiency','reference_wavelength_m','gain_e_per_relative_power','read_sigma_e','background_e','output_x_m','output_y_m','scene_x_m','scene_y_m','seed','weights','variance_estimate_e2','alpha','L_data_estimate','effective_electron_response','method','measurement_unit']}
            values.update(noisy_measurement_e=v['measurement_e'],ideal_measurement_e=ideal_measurement,noisy_reconstruction=xn,ideal_reconstruction=x0,noisy_prediction_e=v['prediction_e'],noisy_residual_e=v['residual_e'],ideal_prediction_e=base.forward(x0)+background,ideal_residual_e=base.forward(x0)+background-ideal_measurement,ideal_weighted_target=fit['target'],source_identity=np.asarray(item['identity']),**errors)
            saved[ident]=values;histories[ident]=h;records.append(rec);valid=valid and numerical
            if dest:
                np.savez_compressed(dest/'arrays'/(ident+'.npz'),**values);write_json(dest/'metrics'/(ident+'_optimization.json'),dict(alpha=rec['alpha'],status=result['status'],execution='computed_ideal_control',history=h));write_json(dest/'metrics'/(ident+'_decomposition.json'),decomposition)
                with (dest/'metrics'/(ident+'_optimization.csv')).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(last));w.writeheader();w.writerows(h)
                write_json(dest/'progress.json',dict(status='running',completed_jobs=len(records),planned_jobs=planned))
            print('%s: %s, %d轮, 基线=%.6g, 噪声扰动=%.6g, 交叉项=%.6g'%(ident,result['status'],result['iterations'],decomposition['norms_relative']['baseline'],decomposition['norms_relative']['noise_perturbation'],decomposition['cross_term_relative_squared']),flush=True)
            fig,axes=plt.subplots(3,6,figsize=(20,8));figs.append(fig);vmax=max(float(truth.max()),float(xn.max()),float(x0.max()));signed_limit=max(float(abs(xn-x0).max()),1e-30)
            extent=[v['scene_x_m'][0]*1e6,v['scene_x_m'][-1]*1e6,v['scene_y_m'][0]*1e6,v['scene_y_m'][-1]*1e6]
            for b in range(3):
                for col,(title,a) in enumerate([('评价真值',truth[b]),('含噪恢复（复用）',xn[b]),('理想测量控制',x0[b]),('含噪绝对误差',abs(xn[b]-truth[b])),('理想绝对误差',abs(x0[b]-truth[b])),('含噪－理想',xn[b]-x0[b])]):
                    ax=axes[b,col];kw=dict(cmap='coolwarm',vmin=-signed_limit,vmax=signed_limit) if col==5 else dict(vmin=0,vmax=vmax)
                    im=ax.imshow(a,origin='lower',extent=extent,**kw);ax.set_title(title+' %.0f nm'%(v['wavelengths_m'][b]*1e9));ax.set_xlabel('x / μm');ax.set_ylabel('y / μm')
                    if col==0:power_im=im
                    if col==5:signed_im=im
            fig.suptitle(ident+'：固定W与alpha的理想测量控制');fig.subplots_adjust(right=.88,wspace=.6,hspace=.6)
            # 色条放在整个网格右侧，避免共享色条与末列坐标标签重叠。
            power_bar=fig.add_axes([.905,.18,.009,.62]);signed_bar=fig.add_axes([.958,.18,.009,.62])
            fig.colorbar(power_im,cax=power_bar,label='相对波段积分功率');fig.colorbar(signed_im,cax=signed_bar,label='有符号功率差')
            if dest:fig.savefig(dest/'figures'/(ident+'.png'),dpi=c['plots']['dpi'])
        for source in evidence['sources']:
            if tree_sha(Path(source['path']))!=source['sha256']:raise RuntimeError('运行期间来源改变')
        if dest:
            for ident,values in saved.items():
                with np.load(dest/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                    if set(z.files)!=set(values) or any(not np.array_equal(z[k],a) for k,a in values.items()):raise RuntimeError('控制数组落盘不一致')
                h=json.loads((dest/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))['history'];rec=next(r for r in records if r['identity']==ident)
                if h!=histories[ident] or json.loads((dest/'metrics'/(ident+'_decomposition.json')).read_text(encoding='utf-8'))!=rec['decomposition'] or not (dest/'figures'/(ident+'.png')).exists():raise RuntimeError('控制轨迹/分解/图落盘错误')
                with (dest/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:
                    if len(list(csv.DictReader(f)))!=len(h):raise RuntimeError('控制CSV长度错误')
            paths=['main_stage02f3_1.py','config_stage02f3_1.json','optics/fixed_target_control.py','optics/stage02f2_source.py','optics/electron_reconstruction.py','optics/reconstruction_accelerated.py','optics/ridge_certificate.py','main_stage02f2.py']
            write_json(dest/'source_manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
            with (dest/'metrics/decomposition.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f);w.writerow(['identity','status','iterations','alpha','total_relative','baseline_relative','noise_perturbation_relative','cross_relative_squared','baseline_lower','baseline_upper','noise_lower','noise_upper'])
                for r in records:
                    d=r['decomposition'];w.writerow([r['identity'],r['solver_status'],r['iterations'],r['alpha'],d['norms_relative']['total'],d['norms_relative']['baseline'],d['norms_relative']['noise_perturbation'],d['cross_term_relative_squared'],d['baseline_interval']['lower_relative'],d['baseline_interval']['upper_relative'],d['noise_perturbation_interval']['lower_relative'],d['noise_perturbation_interval']['upper_relative']])
        if runtime['show_plots']:plt.show()
        complete=len(records)==planned;converged=complete and all(r['solver_status']=='converged' for r in records);certified=complete and all(r['certificate_passed'] for r in records)
        passed=valid and complete and converged and certified
        report=dict(status='completed' if passed else 'incomplete',validation_passed=bool(passed),numerical_validation_passed=bool(valid),all_controls_converged=converged,certificate_validation_passed=certified,budget_exhausted=budget,planned_jobs=planned,completed_jobs=len(records),noisy_verified_reused_count=len(records),noisy_new_iterations=0,records=records,config=c,runtime=runtime,backend=backend,environment=environment_info(),source_unchanged=True,elapsed_s=time.perf_counter()-start,scope='Jeon光学编码复现＋理想测量控制诊断；非论文网络、非双孔径',assumptions=['理想测量使用模拟期望的特权信息，仅用于诊断，不用于实际恢复或参数选择。','固定W来自含噪观测，理想基线包括编码/非负约束/正则化/权重影响，不是纯岭偏差。','同一次实现的确定性误差分解，交叉项可负；不是独立方差相加。','范数区间为优化距离证书传播，非统计置信或区间舍入包络。','论文旋转方向对应、相机标定、真实GUI/PyCharm手动点击尚未验证。'])
        if dest:write_json(dest/'metrics/validation.json',report)
        print('结果目录：'+str(dest) if dest else '无保存模式：未写结果文件',flush=True);return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figs:plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_stage02f3_1.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'));return 0 if run(c,a.no_save,a.show_plots)['validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
