# -*- coding: utf-8 -*-
"""02F-3.2：仅用探测器留出观测选择alpha的小批CPU验证。"""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from optics.detector_holdout import prepare_problem,fit_candidate,choose_candidate,refit_selected
from optics.stage02f3_1_source import load_holdout_source
from optics.electron_reconstruction import make_electron_operator
from optics.reconstruction import estimate_data_lipschitz,evaluate_cube,squared_norm
from optics.reconstruction_accelerated import validate_accelerated
from optics.imaging import operator_checks
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,configure_plotting,effective_runtime
from optics.runutil import environment_info
from main_stage02f2 import optimization_check
from main_stage02b import allocate_unique_run_dir
ROOT=Path(__file__).resolve().parent

def write_json(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def validate_config(c):
    if (c['device'],c['scene'],c['method'],c['gain_e_per_relative_power'])!=('continuous','coincident_points','unweighted',10000.):raise ValueError('本批限定规范点源/连续器件/G10000/非加权')
    f=c['alpha_factors']
    if not f or any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in f) or any(b>=a for a,b in zip(f,f[1:])):raise ValueError('alpha网格须正有限且严格强到弱')
    r=c['reconstruction'];validate_accelerated(r['solver'])
    for v in [r['initial_L_factor'],r['certificate_relative_threshold'],r['power']['power_tolerance'],c['budget_seconds'],*c['thresholds'].values()]:
        if isinstance(v,bool) or not np.isfinite(v) or v<=0:raise ValueError('数值标度/容差须正有限')
    for v,minimum in [(c['split']['seed'],0),(c['check_seed'],0),(r['power']['seed'],0),(r['power']['power_iterations'],1),(c['plots']['dpi'],1)]:
        if isinstance(v,bool) or not isinstance(v,int) or v<minimum:raise ValueError('种子/次数/dpi须合法整数')
    if not isinstance(c['split']['identity'],str) or not c['split']['identity'] or isinstance(c['split']['validation_fraction'],bool) or not 0<c['split']['validation_fraction']<1:raise ValueError('拆分身份或比例非法')

def numerical_record(op,target,result,alpha,c):
    last=result['history'][-1];x=result['reconstruction'];checks=operator_checks(op,c['check_seed'],target);ridge=optimization_check(op,target,x,alpha,last,c['check_seed'])
    valid=bool(checks['inner_product_relative_error']<=c['thresholds']['adjoint_relative'] and checks['gradient_relative_error']<=c['thresholds']['gradient_relative'] and ridge['ridge_gradient_relative_error']<=c['thresholds']['gradient_relative'] and np.isclose(ridge['objective_recomputed'],last['objective'],rtol=1e-12,atol=0) and np.isclose(ridge['projected_gradient_recomputed'],last['projected_gradient_relative'],rtol=1e-10,atol=1e-14))
    if np.any(x<0) or not np.all(np.isfinite(x)) or not all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(result['history'],result['history'][1:])):raise RuntimeError('恢复非负性/轨迹检查失败')
    return dict(operator_checks=checks,optimization_check=ridge,numerical_validation_passed=valid)

def run(c,no_save=False,show_plots=False):
    validate_config(c);runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots'])
    dest=allocate_unique_run_dir('stage02f3_2') if runtime['save_results'] else None
    start=time.perf_counter();records=[];saved={};histories={};figs=[];budget=False;selection=None;final_record=None;evaluation=None
    try:
        items,evidence=load_holdout_source(resolve_project_path(c['source_run'],ROOT),ROOT)
        item=next(i for i in items if i['device']==c['device'] and i['scene']==c['scene'] and i['method']==c['method']);v=item['values']
        # 参数选择问题只包含器件与含噪观测；评价真值/理想控制不传入。
        base=make_electron_operator(v['kernels'],[len(v['scene_y_m']),len(v['scene_x_m'])],float(v['pitch_m']),v['response'],v['wavelengths_m'],v['quantum_efficiency'],float(v['reference_wavelength_m']),float(v['gain_e_per_relative_power']),v['crop_indices'].tolist())
        problem=prepare_problem(base,v['noisy_measurement_e'],float(v['background_e']),c['split']);train=problem['train'];validation=problem['validation'];cfg=c['reconstruction']
        train_estimate=estimate_data_lipschitz(train,cfg['power'])
        validation_checks=operator_checks(validation,c['check_seed'],problem['target_validation'])
        if validation_checks['inner_product_relative_error']>c['thresholds']['adjoint_relative'] or validation_checks['gradient_relative_error']>c['thresholds']['gradient_relative']:raise RuntimeError('验证采样算子检查失败')
        split_values=dict(train_indices=train.indices,validation_indices=validation.indices,detector_shape=np.asarray(base.output_shape),target_train_signal_e=problem['target_train'],target_validation_signal_e=problem['target_validation'],seed=np.asarray(c['split']['seed']))
        physics={k:v[k] for k in ['kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','quantum_efficiency','reference_wavelength_m','gain_e_per_relative_power','read_sigma_e','background_e','output_x_m','output_y_m','scene_x_m','scene_y_m','measurement_unit']};physics.update(measurement_e=problem['measurement'],effective_electron_response=base.response)
        saved.update(split=split_values,measurement_problem=physics)
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'source_evidence.json',evidence)
            np.savez_compressed(dest/'arrays/split.npz',**split_values);np.savez_compressed(dest/'arrays/measurement_problem.npz',**physics)
        for index,factor in enumerate(c['alpha_factors']):
            if time.perf_counter()-start>c['budget_seconds']:budget=True;break
            fit=fit_candidate(problem,factor,cfg,train_estimate);result=fit['result'];ident='candidate_%02d'%index
            rec=dict(identity=ident,alpha_factor=factor,alpha=fit['alpha'],L_data_estimate=train_estimate['value'],solver_status=result['status'],iterations=result['iterations'],elapsed_s=result['elapsed_s'],initialization=result['initialization'],validation_mse_e2=fit['validation_mse_e2'],certificate=fit['certificate'],certificate_passed=fit['certificate_passed'],final_optimization=result['history'][-1],**numerical_record(train,problem['target_train'],result,fit['alpha'],c))
            records.append(rec);histories[ident]=result['history'];values=dict(reconstruction=result['reconstruction'],alpha=np.asarray(fit['alpha']),alpha_factor=np.asarray(factor),L_train_estimate=np.asarray(train_estimate['value']),prediction_train_signal_e=fit['prediction_train_signal_e'],residual_train_e=fit['residual_train_e'],prediction_validation_signal_e=fit['prediction_validation_signal_e'],residual_validation_e=fit['residual_validation_e']);saved[ident]=values
            if dest:
                np.savez_compressed(dest/'arrays'/(ident+'.npz'),**values);write_json(dest/'metrics'/(ident+'_optimization.json'),dict(alpha=fit['alpha'],status=result['status'],history=result['history']))
                with (dest/'metrics'/(ident+'_optimization.csv')).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result['history'][-1]));w.writeheader();w.writerows(result['history'])
                write_json(dest/'progress.json',dict(status='running',planned_jobs=len(c['alpha_factors'])+1,completed_jobs=len(records),last_record=rec))
            print('%s factor=%.1g: %s, %d轮, 验证MSE=%.8g, 距离相对上界=%.6g'%(ident,factor,result['status'],result['iterations'],fit['validation_mse_e2'],fit['certificate']['distance_upper_bound_relative']),flush=True)
            if result['status']!='converged' or not fit['certificate_passed'] or not rec['numerical_validation_passed']:break
        ready=len(records)==len(c['alpha_factors']) and all(r['solver_status']=='converged' and r['certificate_passed'] and r['numerical_validation_passed'] for r in records)
        if ready and time.perf_counter()-start>c['budget_seconds']:budget=True
        if ready and not budget:
            chosen=choose_candidate(records,c['alpha_factors']);factor=records[chosen]['alpha_factor']
            selection=dict(selected_index=chosen,selected_identity=records[chosen]['identity'],selected_factor=factor,criterion='留出像元电子数MSE严格最小；并列按网格顺序',scores_e2=[r['validation_mse_e2'] for r in records],score_gaps_to_minimum_e2=[r['validation_mse_e2']-records[chosen]['validation_mse_e2'] for r in records],truth_used=False,ideal_measurement_used=False)
            # 先落盘选择证据，再进行全数据重拟合与最终真值评价。
            if dest:write_json(dest/'metrics/selection.json',selection)
            fit=refit_selected(problem,factor,cfg);result=fit['result'];x=result['reconstruction'];alpha=fit['alpha'];cert=fit['certificate']
            certified=cert['distance_upper_bound_relative'] is not None and cert['distance_upper_bound_relative']<=cfg['certificate_relative_threshold']
            final_record=dict(identity='full_refit',selected_factor=factor,alpha=alpha,L_data_estimate=fit['spectral_estimate']['value'],spectral_estimate=fit['spectral_estimate'],solver_status=result['status'],iterations=result['iterations'],elapsed_s=result['elapsed_s'],initialization=result['initialization'],certificate=cert,certificate_passed=bool(certified),final_optimization=result['history'][-1],**numerical_record(base,problem['measurement']-problem['background'],result,alpha,c))
            prediction=base.forward(x)+problem['background'];residual=prediction-problem['measurement'];values=dict(reconstruction=x,prediction_e=prediction,residual_e=residual,alpha=np.asarray(alpha),selected_factor=np.asarray(factor),L_full_estimate=np.asarray(fit['spectral_estimate']['value']));saved['full_refit']=values;histories['full_refit']=result['history']
            if dest:
                np.savez_compressed(dest/'arrays/full_refit.npz',**values);write_json(dest/'metrics/full_refit_optimization.json',dict(alpha=alpha,status=result['status'],history=result['history']))
                with (dest/'metrics/full_refit_optimization.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result['history'][-1]));w.writeheader();w.writerows(result['history'])
            # 只有选定方案和固定旧基线接受评价；不生成候选真值误差表。
            truth=v['evaluation_truth'];baseline=v['noisy_reconstruction'];threshold=evidence['evaluation_threshold']
            evaluation=dict(selected=evaluate_cube(truth,x,threshold),baseline=item['parent']['record']['evaluation'],baseline_execution='reused_verified',baseline_new_iterations=0,baseline_source_elapsed_s=item['parent']['record']['elapsed_s'],baseline_certificate=item['parent']['record']['certificate'],baseline_source_identity=item['parent']['identity'],selected_certificate=cert)
            eval_values=dict(evaluation_truth=truth,baseline_reconstruction=baseline,selected_reconstruction=x);saved['final_evaluation']=eval_values
            if dest:np.savez_compressed(dest/'arrays/final_evaluation.npz',**eval_values);write_json(dest/'metrics/final_evaluation.json',evaluation)
            print('选择factor=%.1g；全数据%s，%d轮；最终cube=%.6g（仅作评价）'%(factor,result['status'],result['iterations'],evaluation['selected']['cube_relative_l2']),flush=True)
            fig,axes=plt.subplots(3,5,figsize=(16,8));figs.append(fig);vmax=max(float(truth.max()),float(x.max()),float(baseline.max()))
            extent=[v['scene_x_m'][0]*1e6,v['scene_x_m'][-1]*1e6,v['scene_y_m'][0]*1e6,v['scene_y_m'][-1]*1e6]
            for b in range(3):
                for col,(title,a) in enumerate([('评价真值',truth[b]),('固定旧基线',baseline[b]),('留出选参后重拟合',x[b]),('旧基线绝对误差',abs(baseline[b]-truth[b])),('选参后绝对误差',abs(x[b]-truth[b]))]):
                    im=axes[b,col].imshow(a,vmin=0,vmax=vmax,origin='lower',extent=extent);axes[b,col].set_title(title+' %.0f nm'%(v['wavelengths_m'][b]*1e9));axes[b,col].set_xlabel('x / μm');axes[b,col].set_ylabel('y / μm')
            fig.suptitle('仅按探测器留出观测选择factor=%.1g；原始功率评价'%factor);fig.subplots_adjust(right=.9,wspace=.6,hspace=.6);bar=fig.add_axes([.925,.18,.012,.62]);fig.colorbar(im,cax=bar,label='相对波段积分功率')
            if dest:fig.savefig(dest/'figures/final_comparison.png',dpi=c['plots']['dpi'])
        scorefig,axs=plt.subplots(1,2,figsize=(10,4));figs.append(scorefig)
        if records:axs[0].semilogx([r['alpha_factor'] for r in records],[r['validation_mse_e2'] for r in records],'o-')
        axs[0].set_xlabel('预声明alpha相对因子');axs[0].set_ylabel('验证MSE / 电子数平方');axs[0].set_title('仅观测留出分数')
        mask=np.zeros(base.output_shape);mask.ravel()[validation.indices]=1;axs[1].imshow(mask,origin='lower',vmin=0,vmax=1,cmap='gray');axs[1].set_title('白：验证；黑：训练');axs[1].set_xlabel('x / 像元');axs[1].set_ylabel('y / 像元');scorefig.tight_layout()
        if dest:scorefig.savefig(dest/'figures/holdout_protocol.png',dpi=c['plots']['dpi'])
        for source in evidence['sources']:
            if tree_sha(Path(source['path']))!=source['sha256']:raise RuntimeError('运行期间来源改变')
        if dest:
            for ident,values in saved.items():
                with np.load(dest/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
                    if set(z.files)!=set(values) or any(not np.array_equal(z[k],a) for k,a in values.items()):raise RuntimeError('落盘数组不一致')
            for ident,h in histories.items():
                if json.loads((dest/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))['history']!=h:raise RuntimeError('落盘轨迹错误')
                with (dest/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as f:
                    if len(list(csv.DictReader(f)))!=len(h):raise RuntimeError('落盘CSV长度错误')
            if selection is not None and json.loads((dest/'metrics/selection.json').read_text(encoding='utf-8'))!=selection:raise RuntimeError('选择证据落盘错误')
            paths=['main_stage02f3_2.py','config_stage02f3_2.json','optics/detector_holdout.py','optics/stage02f3_1_source.py','optics/reconstruction.py','optics/reconstruction_accelerated.py','optics/ridge_certificate.py','main_stage02f2.py']
            write_json(dest/'source_manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
        if runtime['show_plots']:plt.show()
        passed=bool(ready and not budget and final_record is not None and final_record['solver_status']=='converged' and final_record['certificate_passed'] and final_record['numerical_validation_passed'])
        report=dict(status='completed' if passed else 'incomplete',validation_passed=passed,budget_exhausted=budget,planned_jobs=len(c['alpha_factors'])+1,completed_jobs=len(records)+(final_record is not None),candidate_records=records,full_refit=final_record,evaluation=evaluation,train_spectral_estimate=train_estimate,train_pixel_count=len(train.indices),validation_pixel_count=len(validation.indices),validation_operator_checks=validation_checks,config=c,runtime=runtime,backend=backend,environment=environment_info(),source_unchanged=True,elapsed_s=time.perf_counter()-start,scope='Jeon光学编码复现＋仅观测留出选参基础验证；非网络、非双孔径',assumptions=['同一次合成电子观测；训练/验证只按预声明身份与seed拆分。','候选求解/选择无真值、理想测量、理论方差信息；只评价最终方案及固定旧基线。','电子预测留出MSE不等于光谱cube误差；单次拆分不证明总体最优或显著性。','全数据重拟合转移相对因子，不转移实际alpha；选择过的验证像元不作为独立测试。','论文旋转方向对应、真实GUI/PyCharm点击和相机标定尚未验证。'])
        if selection is not None:report['selection']=selection
        if dest:write_json(dest/'metrics/validation.json',report)
        print('结果目录：'+str(dest) if dest else '无保存模式：未写结果文件',flush=True);return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for fig in figs:plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_stage02f3_2.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'));return 0 if run(c,a.no_save,a.show_plots)['validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
