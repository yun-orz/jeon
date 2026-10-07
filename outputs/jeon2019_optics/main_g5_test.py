# -*- coding: utf-8 -*-
"""G5补充：一幅未参与训练/选模型的Harvard场景，冻结模型CPU测试。"""
import argparse
import csv
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from main_g1 import write_json
from main_g3 import tensor,array
from main_g5 import load_source
from main_stage02b import allocate_unique_run_dir
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB,HQSDecoder
from optics.g4_data import file_sha
from optics.g5_metrics import assess
from optics.g5_test_data import prepare_test
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def predict(model,measurements):
    model.eval()
    with torch.no_grad():return np.stack([array(model(tensor(m))) for m in measurements])


def aggregate(rows,L):
    mse=float(np.mean([r['mse'] for r in rows]));valid=sum(r['sam_valid_pixels'] for r in rows)
    return dict(mse_pooled=mse,psnr_pooled_db=float(10*np.log10(L*L/mse)) if mse>0 else None,
        psnr_is_positive_infinity=mse==0,ssim_mean=float(np.mean([r['ssim_mean'] for r in rows])),
        sam_mean_rad=sum(r['sam_mean_rad']*r['sam_valid_pixels'] for r in rows if r['sam_valid_pixels'])/valid if valid else None,
        sam_valid_pixels=valid,sam_undefined_pixels=sum(r['sam_undefined_pixels'] for r in rows))


def run(c,no_save=False,show_plots=False):
    if type(c['cpu_threads']) is not int or not 1<=c['cpu_threads']<=16:raise ValueError('线程数非法')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('dpi非法')
    if any(type(c['runtime'][key]) is not bool for key in ['save_results','show_plots']):raise ValueError('开关必须布尔值')
    torch.set_num_threads(c['cpu_threads'])
    audit=json.loads(resolve_project_path(c['g5_audit'],ROOT).read_text(encoding='utf-8'))
    g5=resolve_project_path(c['source_g5'],ROOT);g5_before=tree_sha(g5)
    if not audit['independent_audit']['passed'] or g5_before!=audit['run_sha256']:raise ValueError('G5来源未审核或改变')
    for name,sha in audit['sha256'].items():
        if file_sha(Path(name))!=sha:raise ValueError('G5冻结源码或说明改变')
    source,before,codes,old_data,payload,model=load_source(c)
    protocol=json.loads((g5/'metrics/protocol.json').read_text(encoding='utf-8'))
    if c['data_range']!=protocol['data_range'] or c['zero_norm_threshold']!=protocol['zero_norm_threshold']:
        raise ValueError('本批固定G5已审核的指标尺度与零谱阈值')
    protocol=dict(protocol,dataset_scope='one held-out Harvard img5 capture; four center ROIs, not validation or paper ten-image test',independent_test_set=True)
    folder=resolve_project_path(c['test_dataset']['directory'],ROOT);dataset_before=tree_sha(folder)
    targets,metadata=prepare_test(folder,payload['dataset'],c['test_dataset'])
    npop=RGBForward(model.operator.kernels.double().numpy(),model.operator.response.double().numpy())
    # 上式仅将冻结float32缓存转成float64，不改变缓存值；精确G2核用于生成测量。
    source_g3=Path(json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    source_g2=Path(json.loads((source_g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(source_g2/'arrays/g2_measurement.npz',allow_pickle=False) as z:
        npop=RGBForward(z['kernels'],z['response']);waves=z['wavelengths_m'].copy();fingerprint=str(z['fingerprint'])
    measurements=np.stack([npop.forward(p) for p in targets]).astype(np.float32)
    outputs=predict(model,measurements)
    # 用原G4种子重建初始化，并在旧验证图上先核对其确为同一个比较模型。
    source_config=json.loads((source/'config_effective.json').read_text(encoding='utf-8'))
    torch.manual_seed(source_config['seed'])
    initial_model=HQSDecoder(TorchRGB(npop.kernels,npop.response),payload['network_config']).float()
    if not np.array_equal(predict(initial_model,old_data['validation_measurements']),old_data['baseline_validation_predictions']):
        raise RuntimeError('未训练比较模型与G4初始化不同')
    initial_outputs=predict(initial_model,measurements)
    h=metadata['halo'];s=metadata['core_size'];roi=lambda a:a[:,h:h+s,h:h+s,:]
    target=roi(targets);records={};maps={};summary={}
    for label,predictions in [('untrained',roi(initial_outputs)),('frozen_epoch2',roi(outputs))]:
        records[label]=[];maps[label]=[]
        for a,b in zip(target,predictions):
            row,m=assess(a,b,c['data_range'],c['zero_norm_threshold']);records[label].append(row);maps[label].append(m)
        summary[label]=aggregate(records[label],c['data_range'])
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g5_test') if runtime['save_results'] else None
    try:
        fig,axes=plt.subplots(2,3,figsize=(11,7))
        for k,(a,b) in enumerate(zip(target[0].transpose(2,0,1)[[4,12,19]],roi(outputs)[0].transpose(2,0,1)[[4,12,19]])):
            maximum=max(float(a.max()),float(b.max()),1e-12);minimum=min(0.,float(b.min()))
            for j,value in enumerate([a,b]):
                im=axes[j,k].imshow(value,origin='lower',vmin=minimum,vmax=maximum,cmap='inferno');axes[j,k].set_title(('目标' if j==0 else '冻结模型')+f' {round(waves[[4,12,19][k]]*1e9)}nm')
                fig.colorbar(im,ax=axes[j,k])
        fig.suptitle('独立场景中央ROI：每列共用完整目标/预测范围，相对光子训练单位');fig.tight_layout();figures.append(('heldout_reconstruction.png',fig))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for label,color in [('untrained','gray'),('frozen_epoch2','tab:blue')]:
            mse=np.mean([r['mse_per_band'] for r in records[label]],axis=0)
            psnr=np.full(mse.shape,np.nan);valid=mse>0;psnr[valid]=10*np.log10(c['data_range']**2/mse[valid])
            axes[0].plot(waves*1e9,psnr,color=color,label=label)
            axes[1].plot(waves*1e9,np.mean([r['ssim_per_band'] for r in records[label]],axis=0),color=color,label=label)
        axes[0].set_ylabel('PSNR / dB');axes[1].set_ylabel('SSIM')
        for ax in axes:ax.set_xlabel('波长 / nm');ax.legend();ax.grid(alpha=.3)
        fig.suptitle('一幅独立场景、四个ROI；非论文十图基线');fig.tight_layout();figures.append(('heldout_per_band.png',fig))
        report=dict(status='completed',heldout_test_passed=True,independent_test_scene=True,test_scene=metadata['scene'],
            independent_scene_count=1,roi_count=len(targets),whole_image_evaluated=False,optimizer_steps=0,
            checkpoint_frozen=True,best_epoch=payload['best_epoch'],initial_model_matches_g4=True,
            paper_alignment_passed=False,response_status='synthetic_not_calibrated',summary=summary,
            metric_protocol=protocol,scope='heldout scene replaces validation scope only; metric formula unchanged',
            device='cpu',torch_version=torch.__version__,environment=environment_info(),runtime=runtime,backend=backend,
            height_fingerprint=fingerprint,prediction_clipped=False,source_and_dataset_unchanged=True)
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'dataset_manifest.json',metadata)
            write_json(dest/'metrics/scores.json',records)
            np.savez_compressed(dest/'arrays/heldout_test.npz',targets_context=targets,measurements=measurements,
                predictions_context=outputs,untrained_predictions_context=initial_outputs,wavelengths_m=waves,
                sam_rad=np.stack([m['sam_rad'] for m in maps['frozen_epoch2']]),ssim=np.stack([m['ssim'] for m in maps['frozen_epoch2']]),
                sam_valid=np.stack([m['sam_valid'] for m in maps['frozen_epoch2']]),normalization_scale=metadata['normalization_scale'])
            with (dest/'metrics/per_band.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['model','patch','wavelength_nm','mse','psnr_db','ssim'])
                for label,rows in records.items():
                    for i,row in enumerate(rows):
                        for k,wave in enumerate(waves):writer.writerow([label,i,round(wave*1e9),row['mse_per_band'][k],row['psnr_per_band_db'][k],row['ssim_per_band'][k]])
            write_json(dest/'source_evidence.json',dict(g4=dict(path=str(source),sha256=before),g5=dict(path=str(g5),sha256=g5_before),dataset=dict(path=str(folder),sha256=dataset_before)))
            names=sorted(set(['main_g5_test.py','config_g5_test.json','optics/g5_test_data.py','main_g5.py','optics/g5_metrics.py',*codes]))
            write_json(dest/'source_manifest.json',{name:file_sha(ROOT/name) for name in names})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
        if tree_sha(source)!=before or tree_sha(g5)!=g5_before or tree_sha(folder)!=dataset_before:raise RuntimeError('冻结来源或测试数据改变')
        if dest:write_json(dest/'metrics/validation.json',report)
        if runtime['show_plots']:plt.show()
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2));print('独立测试结果：'+str(dest) if dest else '独立测试无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g5_test.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['heldout_test_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
