# -*- coding: utf-8 -*-
"""G4：Harvard两场景真实HSI小规模训练验证，CPU/PyCharm直接运行。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from main_g1 import write_json
from main_g3 import tensor,array
from main_stage02b import allocate_unique_run_dir
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB,HQSDecoder
from optics.g4_data import prepare_subset,file_sha
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def validate(c):
    d=c['dataset'];t=c['training'];n=c['network']
    for value,lo,hi in [(d['train_patches'],1,16),(d['validation_patches'],1,8),(d['core_size'],16,64),
                       (t['epochs'],1,20),(n['stages'],1,3),(n['features'],2,64),(n['levels'],2,4),(c['cpu_threads'],1,16)]:
        if type(value) is not int or not lo<=value<=hi:raise ValueError('本批CPU数据或网络规模越界')
    if d['halo']!=48:raise ValueError('本批固定48像元halo，与97像元核对应')
    if d['train_scene']==d['validation_scene']:raise ValueError('训练验证场景重复')
    if not 0<d['train_percentile']<=100:raise ValueError('训练百分位越界')
    for value in [t['learning_rate'],t['gradient_clip_norm'],n['epsilon_init'],n['rho_init'],n['threshold_init']]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0:raise ValueError('学习数值须有限正值')
    if type(c['seed']) is not int or c['seed']<0:raise ValueError('随机种子非法')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('dpi非法')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']):raise ValueError('运行开关须为布尔值')


def load_optics(c):
    source=resolve_project_path(c['source_g3'],ROOT);before=tree_sha(source)
    status=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    audit=json.loads(resolve_project_path(c['source_audit'],ROOT).read_text(encoding='utf-8'))
    if status['status']!='completed' or not status['structural_validation_passed'] or not audit['independent_audit']['passed'] or (source/'failed.json').exists():raise ValueError('G3来源未审核通过')
    codes=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    for name,sha in codes.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or file_sha(p)!=sha:raise ValueError('冻结G3源码改变')
    evidence=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    g2=resolve_project_path(c['source_g2'],ROOT)
    if tree_sha(g2)!=evidence['sha256']:raise ValueError('G2来源与G3审核来源不一致')
    with np.load(g2/'arrays/g2_measurement.npz',allow_pickle=False) as data:
        bank=data['kernels'];response=data['response'];waves=data['wavelengths_m'];fp=str(data['fingerprint'])
        if str(data['input_unit'])!='photons_per_bin' or str(data['response_status'])!='synthetic_not_calibrated':raise ValueError('本批要求原G2的波段光子模型和明确合成响应')
    if bank.shape!=(25,97,97) or not np.array_equal(waves,np.array([v/1e9 for v in range(420,661,10)])):raise ValueError('物理核波段或尺寸不匹配')
    return source,before,codes,RGBForward(bank,response),waves,fp


def core(value,config):
    h=config['dataset']['halo'];s=config['dataset']['core_size']
    return value[...,h:h+s,h:h+s]


def evaluate(model,targets,measurements,c,keep=False):
    losses=[];predictions=[];model.eval()
    with torch.no_grad():
        for target,measurement in zip(targets,measurements):
            pred=model(tensor(measurement));loss=(core(pred,c)-core(tensor(target),c)).abs().mean()
            losses.append(float(loss))
            if keep:predictions.append(array(pred))
    return float(np.mean(losses)),np.stack(predictions) if keep else None


def run(c,no_save=False,show_plots=False):
    validate(c);source,before,codes,npop,waves,fp=load_optics(c)
    folder=resolve_project_path(c['dataset']['directory'],ROOT);dataset_before=tree_sha(folder)
    subsets,metadata=prepare_subset(folder,c['dataset'],c['seed'])
    measurements={split:np.stack([npop.forward(p) for p in patches]).astype(np.float32) for split,patches in subsets.items()}
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g4') if runtime['save_results'] else None;start=time.perf_counter()
    try:
        torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']);rng=np.random.default_rng(c['seed'])
        model=HQSDecoder(TorchRGB(npop.kernels,npop.response),c['network']).float()
        optimizer=torch.optim.Adam(model.parameters(),lr=c['training']['learning_rate'])
        baseline_train,_=evaluate(model,subsets['train'],measurements['train'],c)
        baseline_validation,baseline_predictions=evaluate(model,subsets['validation'],measurements['validation'],c,True)
        history=[];best=float('inf');best_state=None;best_epoch=None;steps=0
        for epoch in range(1,c['training']['epochs']+1):
            order=rng.permutation(len(subsets['train']));online_losses=[];gradient_norms=[]
            for index in order:
                model.train();optimizer.zero_grad(set_to_none=True)
                prediction=model(tensor(measurements['train'][index]));target=tensor(subsets['train'][index])
                loss=(core(prediction,c)-core(target,c)).abs().mean();loss.backward()
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('训练损失或参数梯度非有限')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip_norm'])
                if not torch.isfinite(norm):raise RuntimeError('梯度范数非有限')
                optimizer.step();steps+=1;online_losses.append(float(loss.detach()));gradient_norms.append(float(norm))
            train_loss,_=evaluate(model,subsets['train'],measurements['train'],c)
            validation_loss,_=evaluate(model,subsets['validation'],measurements['validation'],c)
            if not np.isfinite(train_loss) or not np.isfinite(validation_loss):raise RuntimeError('验证损失非有限')
            selected=validation_loss<best
            if selected:best=validation_loss;best_epoch=epoch;best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            history.append(dict(epoch=epoch,training_order=order.tolist(),online_loss=float(np.mean(online_losses)),
                train_l1=train_loss,validation_l1=validation_loss,gradient_norm_max=max(gradient_norms),selected_best=selected))
            print(f'真实HSI小规模训练 {epoch}/{c["training"]["epochs"]}：train L1={train_loss:.6g}，validation L1={validation_loss:.6g}',flush=True)
        model.load_state_dict(best_state)
        final_train,_=evaluate(model,subsets['train'],measurements['train'],c)
        final_validation,predictions=evaluate(model,subsets['validation'],measurements['validation'],c,True)
        if abs(final_validation-best)>1e-8:raise RuntimeError('最优模型与验证记录不符')
        # 相对光子目标采用同一固定模型，并未学习光学核或相机响应。
        assert torch.equal(model.operator.kernels,torch.tensor(npop.kernels,dtype=torch.float32))
        assert torch.equal(model.operator.response,torch.tensor(npop.response,dtype=torch.float32))
        fig,axes=plt.subplots(2,3,figsize=(12,8));halo=c['dataset']['halo'];s=c['dataset']['core_size']
        actual=subsets['validation'][0,halo:halo+s,halo:halo+s];output=predictions[0,halo:halo+s,halo:halo+s]
        vmax=max(float(np.percentile(actual,99)),1e-6)
        for j,b in enumerate([4,12,19]):
            axes[0,j].imshow(actual[:,:,b],origin='lower',vmin=0,vmax=vmax,cmap='inferno');axes[0,j].set_title(f'{round(waves[b]*1e9)}nm 验证场景真实HSI目标')
            axes[1,j].imshow(output[:,:,b],origin='lower',vmin=0,vmax=vmax,cmap='inferno');axes[1,j].set_title('小规模训练输出；非论文性能基线')
        fig.suptitle(f'img4独立场景，中心{s}×{s}；共同显示尺度，单位为训练尺度归一化的相对光子表示')
        fig.tight_layout();figures.append(('validation_real_hsi.png',fig))
        fig,ax=plt.subplots(figsize=(7,4));epochs=[0]+[v['epoch'] for v in history]
        ax.plot(epochs,[baseline_train]+[v['train_l1'] for v in history],'.-',label='img3训练场景')
        ax.plot(epochs,[baseline_validation]+[v['validation_l1'] for v in history],'.-',label='img4验证场景')
        ax.set_xlabel('epoch');ax.set_ylabel('中心区域相对HSI L1');ax.legend();ax.grid(alpha=.3);fig.tight_layout();figures.append(('training_validation_curve.png',fig))
        metrics=dict(baseline_train_l1=baseline_train,baseline_validation_l1=baseline_validation,final_train_l1=final_train,
            best_validation_l1=final_validation,best_epoch=best_epoch,history=history,optimizer_steps=steps,
            metric_scope='two_scene_small_subset_center_ROI; not paper PSNR/SSIM/SAM',independent_test_set=False)
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'dataset_manifest.json',metadata)
            np.savez_compressed(dest/'arrays/g4_subset.npz',train_targets=subsets['train'],validation_targets=subsets['validation'],
                train_measurements=measurements['train'],validation_measurements=measurements['validation'],
                baseline_validation_predictions=baseline_predictions,best_validation_predictions=predictions,
                wavelengths_m=waves,normalization_scale=metadata['normalization_scale'],fingerprint=fp,
                sensitivity=np.array(metadata['sensitivity']),target_unit='relative_photons_train_scaled',measurement_unit='relative_linear_response')
            torch.save(dict(state_dict=best_state,network_config=c['network'],best_epoch=best_epoch,dataset=metadata,
                response_status='synthetic_not_calibrated',training_status='harvard_two_scene_small_subset',validation_l1=best),dest/'arrays/best_checkpoint.pt')
            payload=torch.load(dest/'arrays/best_checkpoint.pt',map_location='cpu',weights_only=True)
            restored=HQSDecoder(TorchRGB(npop.kernels,npop.response),c['network']).float();restored.load_state_dict(payload['state_dict'])
            restored_loss,restored_predictions=evaluate(restored,subsets['validation'],measurements['validation'],c,True)
            if not np.array_equal(restored_predictions,predictions) or restored_loss!=final_validation:raise RuntimeError('保存模型重载不一致')
            with np.load(dest/'arrays/g4_subset.npz',allow_pickle=False) as z:
                for key,value in [('train_targets',subsets['train']),('validation_targets',subsets['validation']),('train_measurements',measurements['train']),('validation_measurements',measurements['validation']),('best_validation_predictions',predictions)]:
                    if not np.array_equal(z[key],value):raise RuntimeError('G4保存数组不一致')
            write_json(dest/'metrics/training.json',metrics)
            write_json(dest/'source_evidence.json',dict(g3=dict(path=str(source),sha256=before),dataset=dict(path=str(folder),sha256=dataset_before)))
            names=sorted(set(['main_g4.py','config_g4.json','optics/g4_data.py',*codes]))
            write_json(dest/'source_manifest.json',{name:file_sha(ROOT/name) for name in names})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            with (dest/'report_g4.md').open('x',encoding='utf-8') as stream:
                stream.write(f'# G4真实HSI小规模训练\n\nHarvard官方img3训练、img4验证，校正相机灵敏度并换算相对光子表示，420:10:660nm。训练尺度仅用训练块拟合；没有绝对辐射/光子标定。\n\n{steps}次Adam更新，最优epoch={best_epoch}，验证中心L1={best:.6g}。光学核固定，相机RGB响应仍为合成设置。仅两场景小样本，非官方网络、非原论文指标或泛化结论。详细说明见G4_README.md。\n')
            required=['arrays/g4_subset.npz','arrays/best_checkpoint.pt','config_effective.json','dataset_manifest.json','metrics/training.json',
                'source_evidence.json','source_manifest.json','report_g4.md',*['figures/'+n for n,_ in figures]]
            if any(not (dest/name).is_file() or (dest/name).stat().st_size==0 for name in required):raise RuntimeError('G4必需产物缺失')
        if tree_sha(source)!=before or tree_sha(folder)!=dataset_before:raise RuntimeError('G3来源或原始数据被改变')
        if runtime['show_plots']:plt.show()
        report=dict(status='completed',small_real_hsi_training_passed=True,genuine_hsi_data=True,absolute_photon_calibration=False,
            paper_alignment_passed=False,real_rgb_response_verified=False,full_paper_training_reproduced=False,
            train_scenes=[c['dataset']['train_scene']],validation_scenes=[c['dataset']['validation_scene']],scene_split_disjoint=True,
            train_patches=len(subsets['train']),validation_patches=len(subsets['validation']),context_size=metadata['context_size'],
            core_size=s,epochs=c['training']['epochs'],optimizer_steps=steps,best_epoch=best_epoch,best_validation_l1=best,
            optical_kernels_unchanged=True,source_and_dataset_unchanged=True,height_fingerprint=fp,torch_version=torch.__version__,
            device='cpu',model_dtype='float32',environment=environment_info(),runtime=runtime,backend=backend,elapsed_s=time.perf_counter()-start)
        if dest:write_json(dest/'metrics/validation.json',report)
        print('G4结果：'+str(dest) if dest else 'G4无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g4.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['small_real_hsi_training_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
