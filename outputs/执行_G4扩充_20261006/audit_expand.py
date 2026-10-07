# -*- coding: utf-8 -*-
"""扩充训练独立审核：原始MAT、固定尺度、原块比较、只读评价及来源。"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve
ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE));sys.path.insert(0,str(ENGINE/'tests'))
from main_g1 import write_json
from optics.stage02c_source import tree_sha
from optics.g4_data import file_sha
from test_g5 import reference_ssim


def main():
    run=ENGINE/'results/g4_expand/run_20261006_090723'
    status=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    metadata=json.loads((run/'dataset_manifest.json').read_text(encoding='utf-8'))
    c=json.loads((run/'config_effective.json').read_text(encoding='utf-8'))
    evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    source=Path(evidence['g4']['path']);folder=Path(evidence['dataset']['path'])
    old=json.loads((source/'dataset_manifest.json').read_text(encoding='utf-8'))
    with np.load(run/'arrays/expanded_subset.npz',allow_pickle=False) as z:data={key:z[key].copy() for key in z.files}
    with np.load(source/'arrays/g4_subset.npz',allow_pickle=False) as z:old_data={key:z[key].copy() for key in z.files}
    sensitivity=np.loadtxt(folder/'calib.txt').reshape(-1)
    target_exact=True;origin_exact=True;original_blocks_exact=True
    for offset,split in enumerate(['train','validation']):
        row=metadata[split];raw_data=loadmat(folder/row['scene'],variable_names=['ref','lbl']);raw=raw_data['ref'];lbl=raw_data['lbl']
        mask=(lbl!=0)&np.isfinite(raw[:,:,:25]).all(axis=-1)&(raw[:,:,:25]>=0).all(axis=-1);size=metadata['context_size']
        eligible=[(y,x) for y in range(0,raw.shape[0]-size+1,size) for x in range(0,raw.shape[1]-size+1,size) if mask[y:y+size,x:x+size].all()]
        original=[tuple(p) for p in old[split]['origins_yx']];available=[p for p in eligible if p not in original]
        selected=original+[available[i] for i in np.random.default_rng(c['seed']+offset).choice(len(available),size=row['patches']-len(original),replace=False)]
        origin_exact=origin_exact and [list(p) for p in selected]==row['origins_yx']
        direct=np.stack([raw[y:y+size,x:x+size,:25]/sensitivity[:25]*np.arange(420,661,10)/540/old['normalization_scale'] for y,x in selected]).astype(np.float32)
        target_exact=target_exact and np.array_equal(direct,data[split+'_targets'])
        original_blocks_exact=original_blocks_exact and np.array_equal(data[split+'_targets'][:len(original)],old_data[split+'_targets'])
        del raw_data,raw,lbl,mask,direct
    original_prediction_exact=np.array_equal(data['before_validation_predictions'][:old['validation']['patches']],old_data['best_validation_predictions'])
    g3=Path(json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    g2=Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(g2/'arrays/g2_measurement.npz',allow_pickle=False) as z:kernels=z['kernels'].copy();response=z['response'].copy()
    measure_error=0.
    for split in ['train','validation']:
        # 所有上下文逐波段交叉卷积，而非只核对单点或重用成像类。
        for i,target in enumerate(data[split+'_targets']):
            bands=np.stack([fftconvolve(target[:,:,k].astype(float),kernels[k],mode='same') for k in range(25)],axis=-1)
            expected=(bands@response.T).astype(np.float32)
            measure_error=max(measure_error,float(np.max(np.abs(expected-data[split+'_measurements'][i]))))
    score=json.loads((run/'metrics/scores.json').read_text(encoding='utf-8'));ssim_error=0.;sam_error=0.;psnr_error=0.
    h=metadata['halo'];s=metadata['core_size']
    for label,key in [('expanded_validation_before','before_validation_predictions'),('expanded_validation_after','best_validation_predictions')]:
        for i,(a,b) in enumerate(zip(data['validation_targets'][:,h:h+s,h:h+s],data[key][:,h:h+s,h:h+s])):
            a=a.astype(float);b=b.astype(float);row=score[label]['per_patch'][i]
            local=[reference_ssim(a[:,:,k],b[:,:,k]).mean() for k in range(25)]
            ssim_error=max(ssim_error,float(np.max(np.abs(np.array(local)-row['ssim_per_band']))))
            psnr_error=max(psnr_error,abs(float(20*np.log10(1/np.sqrt(np.mean((a-b)**2))))-row['psnr_cube_db']))
            na=np.linalg.norm(a,axis=-1,keepdims=True);nb=np.linalg.norm(b,axis=-1,keepdims=True)
            valid=(na[:,:,0]>c['zero_norm_threshold'])&(nb[:,:,0]>c['zero_norm_threshold'])
            u=a[valid]/na[valid];v=b[valid]/nb[valid]
            angle=2*np.arctan2(np.linalg.norm(u-v,axis=-1),np.linalg.norm(u+v,axis=-1))
            sam_error=max(sam_error,abs(float(angle.mean())-row['sam_mean_rad']))
    training=status['training'];history=training['history']
    selected_losses=[training['initial_validation_l1']]+[row['validation_l1'] for row in history]
    selection_valid=abs(min(selected_losses)-training['best_validation_l1'])<1e-12 and training['best_epoch']==int(np.argmin(selected_losses))
    training_complete=training['optimizer_steps']==c['training']['epochs']*c['dataset']['train_patches'] and training['epochs_completed']==c['training']['epochs']
    orders_valid=all(sorted(row['training_order'])==list(range(c['dataset']['train_patches'])) for row in history)
    before=tree_sha(ENGINE)
    child=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g4_expand.py'),'--evaluate-only','--checkpoint',str(run/'arrays/best_checkpoint.pt'),'--no-save'],
        cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE)
    inference_scores=None
    if child.returncode==0:
        position=child.stdout.find('{');inference_scores=json.JSONDecoder().raw_decode(child.stdout[position:])[0]
    source_unchanged=all(tree_sha(Path(item['path']))==item['sha256'] for item in evidence.values())
    g0=subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    # 仅核对旧测试run的字节，没有解码测试数组或再次计算测试成绩。
    test_audit=json.loads((ROOT/'outputs/执行_G5独立测试_20261006/最终审核.json').read_text(encoding='utf-8'))
    test_unchanged=tree_sha(Path(test_audit['official_run']))==test_audit['run_sha256']
    result=dict(passed=target_exact and origin_exact and original_blocks_exact and original_prediction_exact and measure_error<1e-6
        and ssim_error<1e-11 and sam_error<1e-12 and psnr_error<1e-12 and selection_valid and training_complete and orders_valid
        and before==after and child.returncode==0 and inference_scores==status['scores'] and source_unchanged and test_unchanged and g0.returncode==0,
        original_mat_targets_exact=target_exact,selected_origins_exact=origin_exact,original_blocks_exact=original_blocks_exact,
        original_model_validation_predictions_exact=original_prediction_exact,measurement_max_error=measure_error,
        ssim_max_error=ssim_error,sam_max_error_rad=sam_error,psnr_max_error_db=psnr_error,
        best_epoch_selection_valid=selection_valid,full_config_training_completed=training_complete,epoch_orders_valid=orders_valid,
        no_save_evaluation_returncode=child.returncode,no_save_scores_exact=inference_scores==status['scores'],
        no_save_sha_unchanged=before==after,files_checked=len(before),full_training_repeated=False,
        evaluation_stdout=child.stdout,evaluation_stderr=child.stderr,source_and_dataset_unchanged=source_unchanged,
        previous_test_run_sha_unchanged=test_unchanged,test_evaluated=False,g0_returncode=g0.returncode,g0_stdout=g0.stdout)
    write_json(Path(__file__).with_name('independent_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise RuntimeError('扩充训练审核未通过')
    names=['main_g4_expand.py','config_g4_expand.json','optics/g4_expand_data.py','tests/test_g4_expand.py','G4_EXPAND_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='expanded_train_validation_passed',official_run=str(run),
        independent_audit=result,tests_run=3,paper_alignment_passed=False,test_evaluated=False,
        sha256={str(ENGINE/name):file_sha(ENGINE/name) for name in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
