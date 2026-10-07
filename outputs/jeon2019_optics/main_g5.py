# -*- coding: utf-8 -*-
"""G5：冻结G4模型，重载推理与验证集PSNR/SSIM/SAM诊断。"""
import argparse
import csv
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from main_g1 import write_json
from main_g3 import tensor,array
from main_stage02b import allocate_unique_run_dir
from optics.g3_hqs import TorchRGB,HQSDecoder
from optics.g4_data import file_sha
from optics.g5_metrics import assess
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path,effective_runtime,configure_plotting
from optics.runutil import environment_info

ROOT=Path(__file__).resolve().parent


def load_source(c):
    source=resolve_project_path(c['source_g4'],ROOT);before=tree_sha(source)
    status=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    audit=json.loads(resolve_project_path(c['source_audit'],ROOT).read_text(encoding='utf-8'))
    if status['status']!='completed' or not status['small_real_hsi_training_passed'] or not audit['independent_audit']['passed'] or (source/'failed.json').exists():
        raise ValueError('G4来源未通过审核')
    codes=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    for name,sha in codes.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or file_sha(path)!=sha:raise ValueError('G4冻结源码改变')
    # 审核文件还锁定了模型来源及说明；CURRENT_STATUS是可追加的状态索引。
    for name,sha in audit['sha256'].items():
        path=Path(name)
        if not path.is_absolute():path=(ROOT.parents[1]/path).resolve()
        if path.name!='CURRENT_STATUS.md' and file_sha(path)!=sha:raise ValueError('G4审核文件改变')
    evidence=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    for item in evidence.values():
        if tree_sha(Path(item['path']))!=item['sha256']:raise ValueError('G4数据或G3来源改变')
    with np.load(source/'arrays/g4_subset.npz',allow_pickle=False) as z:
        data={key:z[key].copy() for key in z.files}
    g3=Path(evidence['g3']['path'])
    g2=Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(g2/'arrays/g2_measurement.npz',allow_pickle=False) as z:
        kernels=z['kernels'].copy();response=z['response'].copy()
    payload=torch.load(source/'arrays/best_checkpoint.pt',map_location='cpu',weights_only=True)
    model=HQSDecoder(TorchRGB(kernels,response),payload['network_config']).float()
    model.load_state_dict(payload['state_dict']);model.eval()
    if not torch.equal(model.operator.kernels,torch.tensor(kernels,dtype=torch.float32)) or not torch.equal(model.operator.response,torch.tensor(response,dtype=torch.float32)):
        raise ValueError('模型中的光学核或响应改变')
    return source,before,codes,data,payload,model


def run(c,no_save=False,show_plots=False):
    if type(c['cpu_threads']) is not int or not 1<=c['cpu_threads']<=16:raise ValueError('CPU线程数非法')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi']<50:raise ValueError('dpi非法')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results','show_plots']):raise ValueError('开关必须布尔值')
    torch.set_num_threads(c['cpu_threads'])
    source,before,codes,data,payload,model=load_source(c)
    metadata=payload['dataset'];h=metadata['halo'];s=metadata['core_size']
    predictions=[]
    with torch.no_grad():
        for measurement in data['validation_measurements']:predictions.append(array(model(tensor(measurement))))
    predictions=np.stack(predictions)
    if not np.array_equal(predictions,data['best_validation_predictions']):raise RuntimeError('重载模型与G4保存预测不一致')
    target=data['validation_targets'][:,h:h+s,h:h+s,:]
    predicted=predictions[:,h:h+s,h:h+s,:]
    initial=data['baseline_validation_predictions'][:,h:h+s,h:h+s,:]
    records={};maps={}
    for label,values in [('untrained',initial),('best_epoch',predicted)]:
        records[label]=[];maps[label]=[]
        for a,b in zip(target,values):
            r,m=assess(a,b,c['data_range'],c['zero_norm_threshold']);records[label].append(r);maps[label].append(m)
    summary={}
    for label,rows in records.items():
        mse=float(np.mean([r['mse'] for r in rows]));valid=sum(r['sam_valid_pixels'] for r in rows)
        summary[label]=dict(mse_pooled=mse,psnr_pooled_db=float(10*np.log10(c['data_range']**2/mse)) if mse>0 else None,
            psnr_pooled_is_positive_infinity=mse==0,ssim_mean=float(np.mean([r['ssim_mean'] for r in rows])),
            sam_mean_rad=sum(r['sam_mean_rad']*r['sam_valid_pixels'] for r in rows if r['sam_valid_pixels'])/valid if valid else None,
            sam_valid_pixels=valid,sam_undefined_pixels=sum(r['sam_undefined_pixels'] for r in rows))
    protocol=dict(dataset_scope='G4 img4 validation: used to select best epoch, NOT independent test',
        target_unit=str(data['target_unit']),normalization_scale=float(data['normalization_scale']),
        spectral_representation='relative photons, sensitivity corrected; NOT absolute calibration or radiance',
        core_size=s,halo=h,psnr_formula='10 log10(L^2 / pooled MSE), L fixed in training-scaled units',
        data_range=c['data_range'],ssim='each band: Gaussian 11x11 sigma1.5, population moments, valid convolution, K1=.01 K2=.03; equal band/patch means; no downsampling',
        sam='arccos(clipped normalized dot), radians; mean over valid pixel spectra; signed raw prediction',
        zero_norm_threshold=c['zero_norm_threshold'],prediction_clipped=False,per_image_normalization=False,
        independent_test_set=False,checkpoint_frozen=True,optimizer_steps=0,
        paper_table1=dict(pdf_page=8,images=10,ours_psnr_db=35.88,ours_ssim=.93,ours_sam=.12,sam_unit_confirmed=False),
        directly_comparable_to_paper=False,
        blockers=['not the ten KAIST test images','validation-selected checkpoint and small training subset',
                  'synthetic RGB response','optical size alignment unresolved','network implementation assumptions',
                  'paper PSNR range/aggregation and SSIM settings not fully specified','photon vs energy spectral representation not aligned'])
    runtime=effective_runtime(c,no_save,show_plots);plt,backend=configure_plotting(runtime['show_plots']);figures=[]
    dest=allocate_unique_run_dir('g5') if runtime['save_results'] else None
    try:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for label,color in [('untrained','gray'),('best_epoch','tab:blue')]:
            band_mse=np.mean([r['mse_per_band'] for r in records[label]],axis=0)
            scores=np.full(band_mse.shape,np.nan);nonzero=band_mse>0
            scores[nonzero]=10*np.log10(c['data_range']**2/band_mse[nonzero])
            axes[0].plot(data['wavelengths_m']*1e9,scores,label=label,color=color)
            axes[1].plot(data['wavelengths_m']*1e9,np.mean([r['ssim_per_band'] for r in records[label]],axis=0),label=label,color=color)
        axes[0].set_ylabel('PSNR / dB');axes[1].set_ylabel('SSIM')
        for ax in axes:ax.set_xlabel('波长 / nm');ax.legend();ax.grid(alpha=.3)
        fig.suptitle('G4验证集指标诊断；未裁剪预测，非论文测试基线');fig.tight_layout();figures.append(('per_band_metrics.png',fig))
        fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        values=[maps['best_epoch'][0]['mse_spatial'],maps['best_epoch'][0]['sam_rad'],maps['best_epoch'][0]['ssim'].mean(axis=-1)]
        for ax,value,title in zip(axes,values,['空间MSE，32×32','SAM / rad，32×32','SSIM有效域，22×22']):
            im=ax.imshow(value,origin='lower');ax.set_title(title);fig.colorbar(im,ax=ax)
        fig.tight_layout();figures.append(('metric_maps.png',fig))
        report=dict(status='completed',metric_implementation_passed=True,model_reload_exact=True,
            validation_diagnostic_only=True,paper_alignment_passed=False,independent_test_set=False,
            best_epoch=payload['best_epoch'],summary=summary,protocol=protocol,environment=environment_info(),
            torch_version=torch.__version__,device='cpu',runtime=runtime,backend=backend)
        if dest:
            write_json(dest/'config_effective.json',dict(c,runtime=runtime));write_json(dest/'metrics/scores.json',records)
            write_json(dest/'metrics/protocol.json',protocol)
            np.savez_compressed(dest/'arrays/g5_evaluation.npz',targets=target,predictions=predicted,untrained_predictions=initial,
                wavelengths_m=data['wavelengths_m'],sam_rad=np.stack([m['sam_rad'] for m in maps['best_epoch']]),
                sam_valid=np.stack([m['sam_valid'] for m in maps['best_epoch']]),
                ssim=np.stack([m['ssim'] for m in maps['best_epoch']]))
            with (dest/'metrics/per_band.csv').open('x',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['model','patch','wavelength_nm','mse','psnr_db','ssim'])
                for label,rows in records.items():
                    for index,row in enumerate(rows):
                        for k,wave in enumerate(data['wavelengths_m']):writer.writerow([label,index,round(wave*1e9),row['mse_per_band'][k],row['psnr_per_band_db'][k],row['ssim_per_band'][k]])
            write_json(dest/'source_evidence.json',dict(path=str(source),sha256=before))
            names=sorted(set(['main_g5.py','config_g5.json','optics/g5_metrics.py',*codes]))
            write_json(dest/'source_manifest.json',{name:file_sha(ROOT/name) for name in names})
            for name,fig in figures:fig.savefig(dest/'figures'/name,dpi=c['plots']['dpi'])
        if tree_sha(source)!=before:raise RuntimeError('G4冻结来源被改变')
        if dest:write_json(dest/'metrics/validation.json',report)
        if runtime['show_plots']:plt.show()
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2))
        print('G5结果：'+str(dest) if dest else 'G5无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest:write_json(dest/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures:plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config_g5.json');p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true');a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['metric_implementation_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
