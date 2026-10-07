# -*- coding: utf-8 -*-
"""独立测试审核：原始数据公式、另一卷积实现、指标及只读运行核对。"""
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
    run=ENGINE/'results/g5_test/run_20261006_081215'
    metadata=json.loads((run/'dataset_manifest.json').read_text(encoding='utf-8'))
    report=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    scores=json.loads((run/'metrics/scores.json').read_text(encoding='utf-8'))
    evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    source=Path(evidence['g4']['path']);folder=Path(evidence['dataset']['path'])
    old=json.loads((source/'dataset_manifest.json').read_text(encoding='utf-8'))
    raw_data=loadmat(folder/metadata['scene'],variable_names=['ref','lbl']);raw=raw_data['ref'];lbl=raw_data['lbl']
    sensitivity=np.loadtxt(folder/'calib.txt').reshape(-1);size=metadata['context_size']
    mask=(lbl!=0)&np.isfinite(raw[:,:,:25]).all(axis=-1)&(raw[:,:,:25]>=0).all(axis=-1)
    eligible=[(y,x) for y in range(0,raw.shape[0]-size+1,size) for x in range(0,raw.shape[1]-size+1,size) if np.all(mask[y:y+size,x:x+size])]
    chosen=[eligible[i] for i in np.random.default_rng(metadata['selection_seed']).choice(len(eligible),size=metadata['patches'],replace=False)]
    raw_patches=np.stack([raw[y:y+size,x:x+size,:25] for y,x in chosen])
    direct_targets=(raw_patches/sensitivity[:25]*np.arange(420,661,10)/540/old['normalization_scale']).astype(np.float32)
    with np.load(run/'arrays/heldout_test.npz',allow_pickle=False) as z:
        targets=z['targets_context'].copy();measurements=z['measurements'].copy();predictions=z['predictions_context'].copy();initial=z['untrained_predictions_context'].copy()
    target_exact=np.array_equal(targets,direct_targets)
    source_g3=Path(json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    source_g2=Path(json.loads((source_g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(source_g2/'arrays/g2_measurement.npz',allow_pickle=False) as z:kernels=z['kernels'].copy();response=z['response'].copy()
    measurement_error=0.
    for i,target in enumerate(targets):
        blurred=np.stack([fftconvolve(target[:,:,k].astype(float),kernels[k],mode='same') for k in range(25)],axis=-1)
        independent=(blurred@response.T).astype(np.float32)
        measurement_error=max(measurement_error,float(np.max(np.abs(independent-measurements[i]))))
    h=metadata['halo'];s=metadata['core_size'];ssim_errors=[];sam_errors=[];psnr_errors=[]
    for label,values in [('untrained',initial),('frozen_epoch2',predictions)]:
        for i,(a,b) in enumerate(zip(targets[:,h:h+s,h:h+s],values[:,h:h+s,h:h+s])):
            a=a.astype(float);b=b.astype(float);row=scores[label][i]
            ssim=[reference_ssim(a[:,:,k],b[:,:,k]).mean() for k in range(25)]
            ssim_errors.append(float(np.max(np.abs(np.array(ssim)-row['ssim_per_band']))))
            rmse=np.sqrt(np.sum((a-b)**2)/a.size);psnr_errors.append(abs(float(20*np.log10(1./rmse))-row['psnr_cube_db']))
            u=a/np.linalg.norm(a,axis=-1,keepdims=True);v=b/np.linalg.norm(b,axis=-1,keepdims=True)
            angle=2*np.arctan2(np.linalg.norm(u-v,axis=-1),np.linalg.norm(u+v,axis=-1))
            sam_errors.append(abs(float(angle.mean())-row['sam_mean_rad']))
    before=tree_sha(ENGINE)
    command=[sys.executable,'-B',str(ENGINE/'main_g5_test.py'),'--no-save']
    child=subprocess.run(command,cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE)
    summary=json.JSONDecoder().raw_decode(child.stdout)[0] if child.returncode==0 else None
    g0=subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    source_unchanged=all(tree_sha(Path(item['path']))==item['sha256'] for item in evidence.values())
    result=dict(passed=target_exact and measurement_error<1e-6 and max(ssim_errors)<1e-11 and max(sam_errors)<1e-12
                and max(psnr_errors)<1e-12 and child.returncode==0 and before==after and summary==report['summary'] and g0.returncode==0 and source_unchanged,
        original_mat_targets_exact=target_exact,selected_origins_exact=[list(p) for p in chosen]==metadata['origins_yx'],
        train_scale_reused=metadata['normalization_scale']==old['normalization_scale'],
        independent_measurement_max_error=measurement_error,independent_ssim_max_error=max(ssim_errors),
        independent_sam_max_error_rad=max(sam_errors),independent_psnr_max_error_db=max(psnr_errors),
        source_and_dataset_unchanged=source_unchanged,no_save_returncode=child.returncode,no_save_summary_exact=summary==report['summary'],
        no_save_sha_unchanged=before==after,files_checked=len(before),foreign_working_directory=str(ROOT/'work'),
        no_save_stdout=child.stdout,no_save_stderr=child.stderr,g0_returncode=g0.returncode,g0_stdout=g0.stdout)
    write_json(Path(__file__).with_name('independent_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise RuntimeError('独立测试审核未通过，保留记录')
    names=['main_g5_test.py','config_g5_test.json','optics/g5_test_data.py','tests/test_g5_test_data.py','G5_TEST_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='one_heldout_scene_small_subset_passed',official_run=str(run),
        independent_audit=result,tests_run=9,independent_scene_count=1,paper_alignment_passed=False,
        sha256={str(ENGINE/name):file_sha(ENGINE/name) for name in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
