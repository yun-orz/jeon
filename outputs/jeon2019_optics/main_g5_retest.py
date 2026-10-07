# -*- coding: utf-8 -*-
"""G5固定场景复测：原G4模型与扩充模型，同一img5输入，0次训练。"""
import argparse
import csv
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from main_g1 import write_json
from main_g5_test import predict,aggregate
from main_stage02b import allocate_unique_run_dir
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB,HQSDecoder
from optics.g4_data import file_sha
from optics.g5_metrics import assess
from optics.g5_retest_metrics import measurement_fit,summary_difference
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def checked_source(path,audit_path):
    audit=json.loads(audit_path.read_text(encoding='utf-8'));before=tree_sha(path)
    if not audit['independent_audit']['passed'] or before!=audit['run_sha256'] or (path/'failed.json').exists():raise ValueError('模型或测试来源改变/未审核')
    for name,sha in audit['sha256'].items():
        if file_sha(Path(name))!=sha:raise ValueError('来源审核锁定的源码或说明改变')
    codes=json.loads((path/'source_manifest.json').read_text(encoding='utf-8'))
    for name,sha in codes.items():
        local=(ROOT/name).resolve()
        if not local.is_relative_to(ROOT) or file_sha(local)!=sha:raise ValueError('来源源码清单改变')
    return before,codes


def run(c,no_save=False,show_plots=False):
    if type(c['cpu_threads']) is not int or not 1<=c['cpu_threads']<=16:raise ValueError('线程数非法')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('dpi非法')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']):raise ValueError('开关须为布尔值')
    torch.set_num_threads(c['cpu_threads']);start=time.perf_counter()
    old=resolve_project_path(c['source_test'],ROOT);new=resolve_project_path(c['source_expanded'],ROOT)
    old_sha,old_codes=checked_source(old,resolve_project_path(c['test_audit'],ROOT))
    new_sha,new_codes=checked_source(new,resolve_project_path(c['expanded_audit'],ROOT))
    print('来源已核对；复用同一img5测量，冻结扩充模型推理……',flush=True)
    old_status=json.loads((old/'metrics/validation.json').read_text(encoding='utf-8'))
    new_status=json.loads((new/'metrics/validation.json').read_text(encoding='utf-8'))
    if new_status['test_evaluated'] or new_status['test_data_loaded']:raise ValueError('扩充训练声明不是未加载测试数据')
    metadata=json.loads((old/'dataset_manifest.json').read_text(encoding='utf-8'))
    payload=torch.load(new/'arrays/best_checkpoint.pt',map_location='cpu',weights_only=True)
    if payload['dataset']['normalization_scale']!=metadata['normalization_scale']:raise ValueError('新旧模型训练尺度不同')
    for split in ['train','validation']:
        if metadata['scene']==payload['dataset'][split]['scene'] or metadata['sha256']==payload['dataset'][split]['sha256']:raise ValueError('测试与训练/验证内容重复')
    with np.load(old/'arrays/heldout_test.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    g4=Path(json.loads((new/'source_evidence.json').read_text(encoding='utf-8'))['g4']['path'])
    g3=Path(json.loads((g4/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    g2=Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(g2/'arrays/g2_measurement.npz',allow_pickle=False) as z:
        op=RGBForward(z['kernels'],z['response']);waves=z['wavelengths_m'].copy();fingerprint=str(z['fingerprint'])
    if not np.array_equal(waves,data['wavelengths_m']):raise ValueError('新旧波段不同')
    model=HQSDecoder(TorchRGB(op.kernels,op.response),payload['network_config']).float();model.load_state_dict(payload['state_dict'])
    if not torch.equal(model.operator.kernels,torch.tensor(op.kernels,dtype=torch.float32)) or not torch.equal(model.operator.response,torch.tensor(op.response,dtype=torch.float32)):raise ValueError('光学核或响应改变')
    outputs=predict(model,data['measurements'])
    h=metadata['halo'];s=metadata['core_size'];roi=lambda a:a[:,h:h+s,h:h+s,:]
    target=roi(data['targets_context']);protocol=dict(old_status['metric_protocol'],dataset_scope='previously examined img5; fixed-scene repeated evaluation',
        independent_test_set=False,fresh_blind_test=False,held_out_from_gradient_updates=True,optimizer_steps=0)
    records={};summary={};fits={};recoded={}
    for label,predictions in [('original_g4',data['predictions_context']),('expanded_g4',outputs)]:
        records[label]=[assess(a,b,protocol['data_range'],protocol['zero_norm_threshold'])[0] for a,b in zip(target,roi(predictions))]
        summary[label]=aggregate(records[label],protocol['data_range'])
        fits[label],recoded[label]=measurement_fit(op,predictions,data['measurements'],h,s)
    old_difference=summary_difference(summary['original_g4'],old_status['summary']['frozen_epoch2'])
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g5_retest') if runtime['save_results'] else None
    try:
        fig,axes=plt.subplots(3,3,figsize=(12,10))
        for column,k in enumerate([4,12,19]):
            values=[target[0,:,:,k],roi(data['predictions_context'])[0,:,:,k],roi(outputs)[0,:,:,k]]
            vmin=min(0.,min(float(a.min()) for a in values));vmax=max(1e-12,max(float(a.max()) for a in values))
            for row,(value,label) in enumerate(zip(values,['目标','原G4','扩充G4'])):
                im=axes[row,column].imshow(value,origin='lower',vmin=vmin,vmax=vmax,cmap='inferno');axes[row,column].set_title(f'{label} {round(waves[k]*1e9)}nm');fig.colorbar(im,ax=axes[row,column])
        fig.suptitle('已查看过的img5固定区域复测：同列共用数值范围，不是新盲测');fig.tight_layout();figures.append(('fixed_scene_comparison.png',fig))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for label in ['original_g4','expanded_g4']:
            mse=np.mean([r['mse_per_band'] for r in records[label]],axis=0);p=np.full(mse.shape,np.nan);valid=mse>0;p[valid]=10*np.log10(protocol['data_range']**2/mse[valid])
            axes[0].plot(waves*1e9,p,label=label);axes[1].plot(waves*1e9,np.mean([r['ssim_per_band'] for r in records[label]],axis=0),label=label)
        axes[0].set_ylabel('PSNR / dB');axes[1].set_ylabel('SSIM')
        for ax in axes:ax.set_xlabel('波长 / nm');ax.legend();ax.grid(alpha=.3)
        fig.tight_layout();figures.append(('fixed_scene_per_band.png',fig))
        report=dict(status='completed',fixed_scene_retest_completed=True,fresh_blind_test=False,held_out_from_gradient_updates=True,
            independent_scene_count=1,roi_count=len(target),optimizer_steps=0,checkpoint_frozen=True,new_best_epoch=payload['best_epoch'],
            old_summary_exact=old_difference==0,old_summary_numerically_reproduced=True,old_summary_max_absolute_difference=old_difference,
            summary=summary,measurement_fit=fits,protocol=protocol,paper_alignment_passed=False,
            response_status='synthetic_not_calibrated',height_fingerprint=fingerprint,device='cpu',torch_version=torch.__version__,
            environment=environment_info(),runtime=runtime,backend=backend,elapsed_s=time.perf_counter()-start)
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'dataset_manifest.json',metadata)
            write_json(dest/'metrics/scores.json',records);write_json(dest/'metrics/protocol.json',protocol)
            np.savez_compressed(dest/'arrays/fixed_scene_retest.npz',targets_context=data['targets_context'],measurements=data['measurements'],
                original_predictions_context=data['predictions_context'],expanded_predictions_context=outputs,wavelengths_m=waves,
                original_recoded_rgb=recoded['original_g4'],expanded_recoded_rgb=recoded['expanded_g4'])
            with (dest/'metrics/per_band.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['model','patch','wavelength_nm','mse','psnr_db','ssim'])
                for label,rows in records.items():
                    for i,row in enumerate(rows):
                        for k,wave in enumerate(waves):writer.writerow([label,i,round(wave*1e9),row['mse_per_band'][k],row['psnr_per_band_db'][k],row['ssim_per_band'][k]])
            write_json(dest/'source_evidence.json',dict(original_test=dict(path=str(old),sha256=old_sha),expanded_training=dict(path=str(new),sha256=new_sha)))
            names=sorted(set(['main_g5_retest.py','config_g5_retest.json','optics/g5_retest_metrics.py',*old_codes,*new_codes]))
            write_json(dest/'source_manifest.json',{name:file_sha(ROOT/name) for name in names})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
            with np.load(dest/'arrays/fixed_scene_retest.npz',allow_pickle=False) as z:
                if not np.array_equal(z['expanded_predictions_context'],outputs):raise RuntimeError('保存预测重载不一致')
        if tree_sha(old)!=old_sha or tree_sha(new)!=new_sha:raise RuntimeError('冻结来源被改变')
        if dest:write_json(dest/'metrics/validation.json',report)
        if runtime['show_plots']:plt.show()
        print(json.dumps(dict(summary=summary,measurement_fit=fits),ensure_ascii=False,allow_nan=False,indent=2));print('固定场景复测结果：'+str(dest) if dest else '固定场景复测无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g5_retest.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['fixed_scene_retest_completed'] else 2


if __name__=='__main__':raise SystemExit(main())
