# -*- coding: utf-8 -*-
"""G4扩充：保留物理模型与尺度，权重热启动、扩大训练/验证，不评价测试集。"""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from main_g1 import write_json
from main_g3 import tensor,array
from main_g4 import evaluate,core
from main_g5 import load_source
from main_stage02b import allocate_unique_run_dir
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB,HQSDecoder
from optics.g4_data import file_sha
from optics.g4_expand_data import prepare_expanded
from optics.g5_metrics import assess
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def validate(c):
    for value,lo,hi in [(c['dataset']['train_patches'],4,32),(c['dataset']['validation_patches'],2,16),
                       (c['training']['epochs'],1,12),(c['training']['half_lr_after_epochs'],1,12),(c['cpu_threads'],1,16)]:
        if type(value) is not int or not lo<=value<=hi:raise ValueError('本批规模越界')
    for v in [c['training']['learning_rate'],c['training']['gradient_clip_norm'],c['training']['max_seconds']]:
        if isinstance(v,bool) or not np.isfinite(v) or v<=0:raise ValueError('训练参数必须有限正值')
    if c['training']['max_seconds']>1200:raise ValueError('本阶段CPU训练预算最多1200秒')
    if type(c['seed']) is not int or c['seed']<0:raise ValueError('随机种子非法')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('dpi非法')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']):raise ValueError('开关须布尔值')


def scores(targets,predictions,c,count=None):
    h=c['dataset']['halo'];s=c['dataset']['core_size'];rows=[]
    for a,b in zip(targets[:count],predictions[:count]):
        row,_=assess(a[h:h+s,h:h+s],b[h:h+s,h:h+s],c['data_range'],c['zero_norm_threshold']);rows.append(row)
    mse=float(np.mean([row['mse'] for row in rows]));valid=sum(row['sam_valid_pixels'] for row in rows)
    summary=dict(mse_pooled=mse,psnr_pooled_db=float(10*np.log10(c['data_range']**2/mse)) if mse>0 else None,
        ssim_mean=float(np.mean([row['ssim_mean'] for row in rows])),
        sam_mean_rad=sum(row['sam_mean_rad']*row['sam_valid_pixels'] for row in rows if row['sam_valid_pixels'])/valid if valid else None,
        sam_valid_pixels=valid,sam_undefined_pixels=sum(row['sam_undefined_pixels'] for row in rows))
    return dict(summary=summary,per_patch=rows)


def run(c,no_save=False,show_plots=False,checkpoint=None):
    validate(c);torch.set_num_threads(c['cpu_threads'])
    print('正在核对冻结模型、指标源码与数据来源……',flush=True)
    g5=resolve_project_path(c['source_g5'],ROOT);g5_before=tree_sha(g5)
    audit=json.loads(resolve_project_path(c['g5_audit'],ROOT).read_text(encoding='utf-8'))
    if not audit['independent_audit']['passed'] or g5_before!=audit['run_sha256']:raise ValueError('G5指标来源改变或未审核')
    for name,sha in audit['sha256'].items():
        if file_sha(Path(name))!=sha:raise ValueError('G5冻结源码改变')
    protocol=json.loads((g5/'metrics/protocol.json').read_text(encoding='utf-8'))
    if c['data_range']!=protocol['data_range'] or c['zero_norm_threshold']!=protocol['zero_norm_threshold']:raise ValueError('本批保持原指标约定')
    source,before,codes,old_data,payload,model=load_source(c)
    folder=resolve_project_path(c['dataset']['directory'],ROOT);dataset_before=tree_sha(folder)
    subsets,metadata=prepare_expanded(folder,payload['dataset'],c['dataset'],c['seed'])
    print(f'数据已准备：训练{len(subsets["train"])}块，验证{len(subsets["validation"])}块；旧尺度保持不变',flush=True)
    for split in ['train','validation']:
        n=metadata[split]['original_patches']
        if not np.array_equal(subsets[split][:n],old_data[split+'_targets']):raise RuntimeError('原块或原训练尺度未保留')
    g3=Path(json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    g2=Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(g2/'arrays/g2_measurement.npz',allow_pickle=False) as z:
        npop=RGBForward(z['kernels'],z['response']);waves=z['wavelengths_m'].copy();fingerprint=str(z['fingerprint'])
    measurements={split:np.stack([npop.forward(p) for p in values]).astype(np.float32) for split,values in subsets.items()}
    print('测量重编码完成，正在计算原模型的训练/验证参考损失……',flush=True)
    initial_train,_=evaluate(model,subsets['train'],measurements['train'],c)
    initial_val,initial_predictions=evaluate(model,subsets['validation'],measurements['validation'],c,True)
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g4_expand') if runtime['save_results'] else None;start=time.perf_counter()
    try:
        torch.manual_seed(c['seed']);rng=np.random.default_rng(c['seed'])
        best=initial_val;best_epoch=0;best_state={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
        history=[];steps=0;training_seconds=0.;checkpoint_source=None
        if checkpoint is not None:
            checkpoint_source=resolve_project_path(checkpoint,ROOT)
            saved=torch.load(checkpoint_source,map_location='cpu',weights_only=True)
            if saved['dataset']!=metadata or saved['network_config']!=payload['network_config']:raise ValueError('评价模型的数据协议或结构不一致')
            model.load_state_dict(saved['state_dict']);best_epoch=saved['best_epoch'];best=saved['validation_l1']
        else:
            # 原G4没有保存Adam状态，因此新建优化器；不是恢复原优化器的续训。
            optimizer=torch.optim.Adam(model.parameters(),lr=c['training']['learning_rate'])
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=c['training']['half_lr_after_epochs'],gamma=.5)
            for epoch in range(1,c['training']['epochs']+1):
                order=rng.permutation(len(subsets['train']));online=[];norms=[];lr=optimizer.param_groups[0]['lr']
                for index in order:
                    if time.perf_counter()-start>c['training']['max_seconds']:raise RuntimeError('训练达到CPU预算，保留失败证据；不伪报完整训练')
                    model.train();optimizer.zero_grad(set_to_none=True)
                    prediction=model(tensor(measurements['train'][index]));target=tensor(subsets['train'][index])
                    loss=(core(prediction,c)-core(target,c)).abs().mean();loss.backward()
                    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('损失或梯度非有限')
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip_norm'])
                    if not torch.isfinite(norm):raise RuntimeError('梯度范数非有限')
                    optimizer.step();steps+=1;online.append(float(loss.detach()));norms.append(float(norm))
                train_l1,_=evaluate(model,subsets['train'],measurements['train'],c)
                val_l1,_=evaluate(model,subsets['validation'],measurements['validation'],c)
                if not np.isfinite(train_l1) or not np.isfinite(val_l1):raise RuntimeError('评价损失非有限')
                selected=val_l1<best
                if selected:best=val_l1;best_epoch=epoch;best_state={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
                scheduler.step()
                history.append(dict(epoch=epoch,training_order=order.tolist(),learning_rate=lr,online_l1=float(np.mean(online)),
                    train_l1=train_l1,validation_l1=val_l1,selected_best=selected,gradient_norm_max=max(norms)))
                print(f'扩充训练 {epoch}/{c["training"]["epochs"]}：train L1={train_l1:.6g}，validation L1={val_l1:.6g}，lr={lr:.6g}',flush=True)
            training_seconds=time.perf_counter()-start;model.load_state_dict(best_state)
        final_train,_=evaluate(model,subsets['train'],measurements['train'],c)
        final_val,predictions=evaluate(model,subsets['validation'],measurements['validation'],c,True)
        if abs(final_val-best)>1e-8:raise RuntimeError('最佳模型与选择损失不一致')
        if not torch.equal(model.operator.kernels,torch.tensor(npop.kernels,dtype=torch.float32)) or not torch.equal(model.operator.response,torch.tensor(npop.response,dtype=torch.float32)):
            raise RuntimeError('光学核或响应改变')
        scored=dict(expanded_validation_before=scores(subsets['validation'],initial_predictions,c),expanded_validation_after=scores(subsets['validation'],predictions,c),
            original_validation_before=scores(subsets['validation'],initial_predictions,c,metadata['validation']['original_patches']),
            original_validation_after=scores(subsets['validation'],predictions,c,metadata['validation']['original_patches']))
        fig,ax=plt.subplots(figsize=(8,4));epochs=[0]+[r['epoch'] for r in history]
        ax.plot(epochs,[initial_train]+[r['train_l1'] for r in history],'.-',label=f'img3训练{len(subsets["train"])}块')
        ax.plot(epochs,[initial_val]+[r['validation_l1'] for r in history],'.-',label=f'img4验证{len(subsets["validation"])}块')
        ax.set_xlabel('热启动后epoch');ax.set_ylabel('中心区域L1');ax.legend();ax.grid(alpha=.3)
        fig.suptitle('同一固定尺度、原光学核；img5未参与本次实验');fig.tight_layout();figures.append(('expanded_training_curve.png',fig))
        h=metadata['halo'];s=metadata['core_size'];a=subsets['validation'][0,h:h+s,h:h+s];b=predictions[0,h:h+s,h:h+s]
        fig,axes=plt.subplots(2,3,figsize=(11,7))
        for column,k in enumerate([4,12,19]):
            vmax=max(float(a[:,:,k].max()),float(b[:,:,k].max()),1e-12);vmin=min(0.,float(b[:,:,k].min()))
            for row,value in enumerate([a[:,:,k],b[:,:,k]]):
                im=axes[row,column].imshow(value,origin='lower',vmin=vmin,vmax=vmax,cmap='inferno')
                axes[row,column].set_title(('目标' if row==0 else '扩充训练最佳模型')+f' {round(waves[k]*1e9)}nm');fig.colorbar(im,ax=axes[row,column])
        fig.suptitle('验证集中心ROI；同列共用显示范围，数值不裁剪');fig.tight_layout();figures.append(('expanded_validation.png',fig))
        training=dict(mode='checkpoint_evaluation' if checkpoint is not None else 'G4 best weights warm start, NEW Adam state',
            optimizer_steps=steps,epochs_completed=len(history),history=history,best_epoch=best_epoch,
            epoch0_is_selection_candidate=True,initial_train_l1=initial_train,initial_validation_l1=initial_val,
            best_train_l1=final_train,best_validation_l1=final_val,training_seconds=training_seconds,
            checkpoint_source=str(checkpoint_source) if checkpoint_source else None,checkpoint_sha256=file_sha(checkpoint_source) if checkpoint_source else None)
        report=dict(status='completed',expanded_training_passed=True,training=training,scores={name:value['summary'] for name,value in scored.items()},
            test_data_loaded=False,test_evaluated=False,independent_test_result=False,paper_alignment_passed=False,
            response_status='synthetic_not_calibrated',optical_kernels_unchanged=True,height_fingerprint=fingerprint,
            target_unit='relative_photons_original_train_scaled',prediction_clipped=False,environment=environment_info(),
            torch_version=torch.__version__,device='cpu',runtime=runtime,backend=backend)
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'dataset_manifest.json',metadata)
            write_json(dest/'metrics/training.json',training);write_json(dest/'metrics/scores.json',scored)
            np.savez_compressed(dest/'arrays/expanded_subset.npz',train_targets=subsets['train'],validation_targets=subsets['validation'],
                train_measurements=measurements['train'],validation_measurements=measurements['validation'],
                before_validation_predictions=initial_predictions,best_validation_predictions=predictions,wavelengths_m=waves,
                normalization_scale=metadata['normalization_scale'])
            torch.save(dict(state_dict=model.state_dict(),network_config=payload['network_config'],best_epoch=best_epoch,
                dataset=metadata,validation_l1=final_val,source_mode=training['mode'],optimizer_state_saved=False),dest/'arrays/best_checkpoint.pt')
            saved=torch.load(dest/'arrays/best_checkpoint.pt',map_location='cpu',weights_only=True)
            restored=HQSDecoder(TorchRGB(npop.kernels,npop.response),payload['network_config']).float();restored.load_state_dict(saved['state_dict'])
            restored_l1,restored_predictions=evaluate(restored,subsets['validation'],measurements['validation'],c,True)
            if restored_l1!=final_val or not np.array_equal(restored_predictions,predictions):raise RuntimeError('扩充模型保存重载不一致')
            write_json(dest/'source_evidence.json',dict(g4=dict(path=str(source),sha256=before),g5=dict(path=str(g5),sha256=g5_before),dataset=dict(path=str(folder),sha256=dataset_before)))
            names=sorted(set(['main_g4_expand.py','config_g4_expand.json','optics/g4_expand_data.py','main_g5.py','optics/g5_metrics.py',*codes]))
            write_json(dest/'source_manifest.json',{name:file_sha(ROOT/name) for name in names})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
        if tree_sha(source)!=before or tree_sha(g5)!=g5_before or tree_sha(folder)!=dataset_before:raise RuntimeError('冻结来源或数据改变')
        if dest:write_json(dest/'metrics/validation.json',report)
        if runtime['show_plots']:plt.show()
        print(json.dumps(report['scores'],ensure_ascii=False,allow_nan=False,indent=2));print('扩充训练结果：'+str(dest) if dest else '扩充训练无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g4_expand.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true')
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--checkpoint');a=p.parse_args()
    if a.evaluate_only!=(a.checkpoint is not None):p.error('--evaluate-only与--checkpoint必须一起使用')
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots,a.checkpoint)['expanded_training_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
