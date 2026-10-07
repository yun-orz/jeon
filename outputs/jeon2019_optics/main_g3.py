# -*- coding: utf-8 -*-
"""G3：HQS展开结构与两步合成样本梯度验证；不是真实HSI训练。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from main_g1 import write_json
from main_g2 import load_source as load_g1_source
from main_stage02b import allocate_unique_run_dir
from optics.g2_rgb import RGBForward,synthetic_response,demo_scene
from optics.g3_hqs import TorchRGB,HQSDecoder
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def tensor(array,dtype=torch.float32):
    return torch.tensor(np.asarray(array).transpose(2,0,1)[None].copy(),dtype=dtype)


def array(value):return value.detach().cpu()[0].permute(1,2,0).numpy().copy()


def validate(c):
    for name,lo,hi in [('stages',1,6),('features',2,64),('levels',2,4)]:
        value=c['network'][name]
        if type(value) is not int or not lo<=value<=hi:raise ValueError('本批网络规模越界：'+name)
    for name,lo,hi in [('steps',1,5),('patch_size',16,64)]:
        value=c['smoke'][name]
        if type(value) is not int or not lo<=value<=hi:raise ValueError('本批仅允许小型梯度验算：'+name)
    positive=[c['network'][k] for k in ['epsilon_init','rho_init','threshold_init']]
    positive.extend(c['smoke'][k] for k in ['learning_rate','normalization_photons'])
    positive.extend(c['thresholds'].values())
    if any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in positive):raise ValueError('数值参数须有限正值')
    if type(c['cpu_threads']) is not int or not 1<=c['cpu_threads']<=16:raise ValueError('CPU线程数越界')
    if type(c['seed']) is not int or c['seed']<0:raise ValueError('随机种子非法')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('dpi非法')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']):raise ValueError('运行开关须为布尔值')


def load_source(c):
    source=resolve_project_path(c['source_run'],ROOT);before=tree_sha(source)
    status=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    audit=json.loads(resolve_project_path(c['source_audit'],ROOT).read_text(encoding='utf-8'))
    if status['status']!='completed' or not status['numerical_validation_passed'] or not audit['independent_audit']['passed'] or (source/'failed.json').exists():raise ValueError('G2来源未完成审核')
    # 当前状态索引可追加；只核验审核清单中的工程文件，不把历史绝对路径当运行依赖。
    for name,sha in audit['sha256'].items():
        if name.startswith('outputs/jeon2019_optics/') and hashlib.sha256((ROOT.parents[1]/name).read_bytes()).hexdigest()!=sha:raise ValueError('G2已审核工程文件改变')
    codes=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    for name,sha in codes.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('G2源码指纹改变')
    cfg=json.loads((source/'config_effective.json').read_text(encoding='utf-8'))
    _,_,_,bank,waves,pitch,fp=load_g1_source(cfg)
    response=synthetic_response(waves,cfg['response']);npop=RGBForward(bank,response,cfg['input_unit'],cfg['bin_width_nm'])
    with np.load(source/'arrays/g2_measurement.npz',allow_pickle=False) as data:
        cube=data['cube'];rgb=data['rgb']
        expected=dict(kernels=bank,response=response,wavelengths_m=waves,pitch_m=pitch,fingerprint=fp,
                      input_unit=cfg['input_unit'],bin_width_nm=cfg['bin_width_nm'],response_status='synthetic_not_calibrated')
        if any(not np.array_equal(data[k],v) for k,v in expected.items()):raise ValueError('G2物理核或单位字段不一致')
    expected_cube=demo_scene(cfg['scene']['height'],cfg['scene']['width'],waves,cfg['scene']['amplitude_photons_per_bin'])
    if cfg['input_unit']=='photons_per_nm':expected_cube/=cfg['bin_width_nm']
    if not np.array_equal(cube,expected_cube) or not np.allclose(rgb,npop.forward(cube),rtol=1e-12,atol=1e-10):raise ValueError('G2场景或强度观测不能复算')
    if c['smoke']['patch_size']>min(cube.shape[:2]):raise ValueError('验算patch超出场景')
    return source,before,codes,npop,cube,rgb,waves,fp


def optical_checks(npop,c):
    rng=np.random.default_rng(c['seed']);x=rng.normal(size=(12,13,25));y=rng.normal(size=(12,13,3))
    op=TorchRGB(npop.kernels,npop.response,npop.spectral_factor).double()
    tx=tensor(x,torch.float64).requires_grad_();ty=tensor(y,torch.float64)
    ax=op(tx);aty=op.adjoint(ty)
    ferr=float(np.linalg.norm(array(ax)-npop.forward(x))/max(np.linalg.norm(npop.forward(x)),1e-30))
    aerr=float(np.linalg.norm(array(aty)-npop.adjoint(y))/max(np.linalg.norm(npop.adjoint(y)),1e-30))
    left=(ax*ty).sum();right=(tx*aty).sum()
    dot=float(abs(left.detach()-right.detach())/max(float(torch.linalg.vector_norm(ax.detach())*torch.linalg.vector_norm(ty)),1e-30))
    (.5*(ax-ty).square().sum()).backward();expected=op.adjoint(ax.detach()-ty)
    grad=float(torch.linalg.vector_norm(tx.grad-expected)/torch.linalg.vector_norm(expected))
    passed=max(ferr,aerr)<=c['thresholds']['numpy_operator_relative'] and max(dot,grad)<=c['thresholds']['torch_adjoint_relative']
    return dict(passed=passed,numpy_forward_relative=ferr,numpy_adjoint_relative=aerr,adjoint_inner_product_relative=dot,autograd_adjoint_relative=grad)


def stage_rows(trace,model,y):
    rows=[]
    with torch.no_grad():
        for i,row in enumerate(trace):
            residual=model.operator(row['updated'])-y
            expected=(1-row['epsilon']*row['rho'])*row['previous']-row['epsilon']*model.operator.adjoint(model.operator(row['previous']))+row['epsilon']*model.operator.adjoint(y)+row['epsilon']*row['rho']*row['prior']
            error=float(torch.linalg.vector_norm(expected-row['updated'])/max(float(torch.linalg.vector_norm(row['updated'])),1e-30))
            rows.append(dict(stage=i,epsilon=float(row['epsilon']),rho=float(row['rho']),threshold=float(row['threshold']),
                equation21_relative=error,measurement_relative=float(torch.linalg.vector_norm(residual)/torch.linalg.vector_norm(y)),
                finite=bool(all(torch.isfinite(row[k]).all() for k in ['previous','prior','data_gradient','updated']))))
    return rows


def run(c,no_save=False,show_plots=False):
    validate(c);source,before,codes,npop,cube,rgb,waves,fp=load_source(c)
    runtime=effective_runtime(c,no_save,show_plots);torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed'])
    plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g3') if runtime['save_results'] else None;start=time.perf_counter()
    try:
        checks=optical_checks(npop,c);scale=c['smoke']['normalization_photons']
        model=HQSDecoder(TorchRGB(npop.kernels,npop.response,npop.spectral_factor),c['network']).float()
        model.eval();y=tensor(rgb/scale)
        with torch.no_grad():untrained,initial,initial_trace=model(y,True)
        untrained_rows=stage_rows(initial_trace,model,y)
        p=c['smoke']['patch_size'];h,w=cube.shape[:2];sy=(h-p)//2;sx=(w-p)//2
        # 裁剪后重新编码，不能用大图RGB裁剪冒充零延拓小域的配对观测。
        target=cube[sy:sy+p,sx:sx+p]/scale;patch_rgb=npop.forward(target)
        tx=tensor(target);ty=tensor(patch_rgb)
        optimizer=torch.optim.Adam(model.parameters(),lr=c['smoke']['learning_rate']);history=[]
        for step in range(c['smoke']['steps']):
            model.train();optimizer.zero_grad(set_to_none=True);pred=model(ty);loss=(pred-tx).abs().mean();loss.backward()
            grads=[]
            for i,stage in enumerate(model.stages):
                prior=[p.grad for p in stage.prior.parameters()]
                finite=all(g is not None and torch.isfinite(g).all() for g in prior)
                row=dict(stage=i,prior_gradient_l1=sum(float(g.abs().sum()) for g in prior if g is not None),
                    epsilon_gradient=float(stage.raw_epsilon.grad),rho_gradient=float(stage.raw_rho.grad),threshold_gradient=float(stage.prior.raw_threshold.grad),all_prior_gradients_finite=bool(finite))
                if not finite or not all(np.isfinite(row[k]) for k in ['prior_gradient_l1','epsilon_gradient','rho_gradient','threshold_gradient']):raise RuntimeError('学习参数梯度非有限')
                grads.append(row)
            optimizer.step()
            with torch.no_grad():after=float((model(ty)-tx).abs().mean())
            history.append(dict(step=step,loss_before=float(loss.detach()),loss_after=after,gradients=grads))
            print(f'合成单patch梯度验证 {step+1}/{c["smoke"]["steps"]}，L1={after:.6g}')
        model.eval()
        with torch.no_grad():smoke,_,trace=model(y,True);smoke_patch=model(ty)
        rows=stage_rows(trace,model,y)
        passed=checks['passed'] and all(v['finite'] and v['equation21_relative']<2e-6 for v in rows+untrained_rows)
        passed=passed and all(g['prior_gradient_l1']>0 and abs(g['epsilon_gradient'])>0 and abs(g['rho_gradient'])>0 for s in history for g in s['gradients'])
        data=dict(cube=cube,rgb=rgb,wavelengths_m=waves,initial_full=array(initial),untrained_full=array(untrained),smoke_full=array(smoke),
            stage_previous=np.stack([array(t['previous']) for t in trace]),stage_prior=np.stack([array(t['prior']) for t in trace]),
            stage_updated=np.stack([array(t['updated']) for t in trace]),stage_data_gradient=np.stack([array(t['data_gradient']) for t in trace]),
            epsilon=np.array([v['epsilon'] for v in rows]),rho=np.array([v['rho'] for v in rows]),
            smoke_target=target,smoke_rgb=patch_rgb,smoke_prediction=array(smoke_patch),normalization_photons=scale,fingerprint=fp)
        fig,axes=plt.subplots(2,3,figsize=(12,8));maximum=float(np.percentile(cube/scale,99.9))
        for j,b in enumerate([4,12,19]):
            axes[0,j].imshow(cube[:,:,b]/scale,origin='lower',cmap='inferno',vmin=0,vmax=maximum)
            axes[0,j].set_title(f'{round(waves[b]*1e9)}nm 合成真值')
            axes[1,j].imshow(data['smoke_full'][:,:,b],origin='lower',cmap='inferno',vmin=0,vmax=maximum)
            axes[1,j].set_title(f'仅{c["smoke"]["steps"]}步梯度验算输出；非训练完成')
        fig.suptitle(f'共同显示上限{maximum:.3g}；少量亮点截断仅用于显示，原数组不变')
        fig.tight_layout();figures.append(('untrained_smoke_comparison.png',fig))
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot([v['measurement_relative'] for v in untrained_rows],'.-',label='随机初始化')
        axes[0].plot([v['measurement_relative'] for v in rows],'.-',label='合成梯度验算后');axes[0].legend();axes[0].set_xlabel('阶段索引');axes[0].set_ylabel('观测残差相对范数')
        axes[1].plot([s['loss_before'] for s in history],'.-',label='更新前');axes[1].plot([s['loss_after'] for s in history],'.-',label='更新后')
        axes[1].set_xlabel('单patch验算步数');axes[1].set_ylabel('L1；不代表泛化精度');axes[1].legend();fig.tight_layout();figures.append(('stage_and_gradient_checks.png',fig))
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));np.savez_compressed(dest/'arrays/g3_diagnostics.npz',**data)
            # 只保存本程序生成的状态字典；不读取未知来源的模型对象。
            torch.save(dict(state_dict=model.state_dict(),network_config=c['network'],response_status='synthetic_not_calibrated',training_status='two_step_single_patch_smoke'),dest/'arrays/smoke_checkpoint.pt')
            payload=torch.load(dest/'arrays/smoke_checkpoint.pt',map_location='cpu',weights_only=True)
            restored=HQSDecoder(TorchRGB(npop.kernels,npop.response,npop.spectral_factor),c['network']).float()
            restored.load_state_dict(payload['state_dict']);restored.eval()
            with torch.no_grad():torch.testing.assert_close(restored(ty),smoke_patch,rtol=0,atol=0)
            with np.load(dest/'arrays/g3_diagnostics.npz',allow_pickle=False) as saved:
                if any(not np.array_equal(saved[k],v) for k,v in data.items()):raise RuntimeError('G3保存数组不一致')
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            write_json(dest/'metrics/optical_checks.json',checks);write_json(dest/'metrics/stage_checks.json',dict(untrained=untrained_rows,smoke=rows))
            write_json(dest/'metrics/gradient_smoke.json',dict(history=history,patch_origin_yx=[sy,sx],measurement_reencoded=True,genuine_hsi_training=False))
            write_json(dest/'source_evidence.json',dict(path=str(source),sha256=before))
            names=sorted(set(['main_g3.py','config_g3.json','optics/g3_hqs.py','requirements_g3.txt',*codes]))
            write_json(dest/'source_manifest.json',{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names})
            with (dest/'report_g3.md').open('x',encoding='utf-8') as stream:
                stream.write(f'# G3 HQS结构与梯度验证\n\nJeon光学编码公式实现＋合成响应＋HQS结构验证。使用PyTorch，原文TensorFlow；无作者权重或真实HSI训练。{c["network"]["stages"]}阶段、{c["network"]["features"]}特征、{c["network"]["levels"]}层，各阶段先验及参数独立。未公开细节为实施假设。\n\n随机初始化输出及{c["smoke"]["steps"]}步合成单patch输出均不是正式重建基线，不能评价论文PSNR/SSIM/SAM。详细中文说明见G3_README.md。\n')
            required=['arrays/g3_diagnostics.npz','arrays/smoke_checkpoint.pt','metrics/optical_checks.json','metrics/stage_checks.json','metrics/gradient_smoke.json',
                'source_evidence.json','source_manifest.json','config_effective.json','report_g3.md',*['figures/'+n for n,_ in figures]]
            if any(not (dest/n).is_file() or (dest/n).stat().st_size==0 for n in required):raise RuntimeError('G3必需结果缺失')
        if tree_sha(source)!=before:raise RuntimeError('G2来源改变')
        if runtime['show_plots']:plt.show()
        report=dict(status='completed' if passed else 'diagnostic_checks_failed',structural_validation_passed=bool(passed),paper_alignment_passed=False,
            genuine_hsi_training=False,real_response_verified=False,torch_version=torch.__version__,device='cpu',model_dtype='float32',optical_reference_dtype='float64',
            parameter_count=sum(p.numel() for p in model.parameters()),stages=len(model.stages),features=c['network']['features'],levels=c['network']['levels'],
            smoke_steps=c['smoke']['steps'],source_unchanged=True,height_fingerprint=fp,environment=environment_info(),runtime=runtime,backend=backend,elapsed_s=time.perf_counter()-start)
        if dest:write_json(dest/'metrics/validation.json',report)
        print('G3结果：'+str(dest) if dest else 'G3无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g3.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['structural_validation_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
