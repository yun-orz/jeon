# -*- coding: utf-8 -*-
"""独立核对原始支持域、共享观测、指标及只读复算，不训练。"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve

ROOT=Path(__file__).resolve().parents[2]; ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE));sys.path.insert(0,str(ENGINE/'tests'))
from optics.stage02c_source import tree_sha
from optics.g4_data import file_sha
from main_g1 import write_json
from test_g5 import reference_ssim


def main():
    run=ENGINE/'results/g5_boundary/run_20261006_130532'
    report=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    records=json.loads((run/'metrics/scores.json').read_text(encoding='utf-8'))
    meta=json.loads((run/'dataset_manifest.json').read_text(encoding='utf-8'))['original_metadata']
    evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    with np.load(run/'arrays/boundary.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    folder=Path(evidence['dataset']['path']);raw=loadmat(folder/meta['scene'],variable_names=['ref','lbl'])
    cal=np.loadtxt(folder/'calib.txt').reshape(-1);chosen=[];selection=[];support=report['support_size']
    for index,(y,x) in enumerate(meta['origins_yx']):
        sy=y+meta['context_size']//2-support//2;sx=x+meta['context_size']//2-support//2
        bounds=sy>=0 and sx>=0 and sy+support<=raw['ref'].shape[0] and sx+support<=raw['ref'].shape[1]
        invalid=None
        if bounds:
            block=raw['ref'][sy:sy+support,sx:sx+support,:25]
            valid=(raw['lbl'][sy:sy+support,sx:sx+support]!=0)&np.isfinite(block).all(-1)&(block>=0).all(-1)
            invalid=int(np.count_nonzero(~valid))
            if not invalid:
                # 独立按声明的物理换算计算，不调用支持域选择/换算函数。
                chosen.append((block/cal[:25]*np.arange(420,661,10)/540./meta['normalization_scale']).astype(np.float32))
        selection.append((index,bounds,invalid,bounds and invalid==0))
    selection_exact=all((r['original_index'],r['in_bounds'],r['invalid_pixels'],r['accepted'])==s for r,s in zip(report['selection'],selection)) and len(selection)==len(report['selection'])
    raw_exact=np.array_equal(np.stack(chosen),data['support_targets'])
    del raw
    shared=[]
    for target in chosen:
        bands=np.stack([fftconvolve(target[:,:,k].astype(float),data['kernels'][k],mode='same') for k in range(25)],axis=-1)
        shared.append(bands@data['response'].T)
    shared=np.stack(shared).astype(np.float32)
    measurement_max=float(np.max(np.abs(shared.astype(float)-data['shared_support_rgb'])))
    crops_exact=True;ssim_error=0.;sam_error=0.;psnr_error=0.;change_error=0.
    core=report['core_size'];a=data['target_core'].astype(float)
    for size in report['context_sizes']:
        offset=(support-size)//2
        crops_exact=crops_exact and np.array_equal(data[f'input_{size}'],shared[:,offset:offset+size,offset:offset+size])
        b=data[f'prediction_core_{size}'].astype(float)
        reference=data[f'prediction_core_{report["context_sizes"][-1]}'].astype(float)
        difference=b-reference
        expected={'rmse':float(np.sqrt(np.mean(difference**2))),'max_absolute':float(np.max(np.abs(difference))),
                  'relative_l2':float(np.sqrt(np.sum(difference**2)/np.sum(reference**2)))}
        change_error=max(change_error,max(abs(v-report['change_vs_largest'][str(size)][k]) for k,v in expected.items()))
        for i,(truth,pred) in enumerate(zip(a,b)):
            row=records[str(size)][i]
            psnr_error=max(psnr_error,abs(float(20*np.log10(1/np.sqrt(np.mean((truth-pred)**2))))-row['psnr_cube_db']))
            ssim_error=max(ssim_error,float(np.max(np.abs(np.array([reference_ssim(truth[:,:,k],pred[:,:,k]).mean() for k in range(25)])-row['ssim_per_band']))))
            u=truth/np.linalg.norm(truth,axis=-1,keepdims=True);v=pred/np.linalg.norm(pred,axis=-1,keepdims=True)
            angles=2*np.arctan2(np.linalg.norm(u-v,axis=-1),np.linalg.norm(u+v,axis=-1))
            sam_error=max(sam_error,abs(float(angles.mean())-row['sam_mean_rad']))
    # 中心32的RGB必须与旧人工截断128支持下的记录一致。
    h=(128-core)//2
    center_exact=np.array_equal(data['input_128'][:,h:h+core,h:h+core],data['legacy_128_rgb'][:,h:h+core,h:h+core])
    manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    code_exact=all(file_sha(ENGINE/name)==sha for name,sha in manifest.items())
    print('独立原始数据、卷积和指标计算完成；执行完整只读复算及SHA核对',flush=True)
    before=tree_sha(ENGINE)
    child=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g5_boundary.py'),'--no-save'],cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE);repeat_exact=False
    if child.returncode==0:
        value=json.JSONDecoder().raw_decode(child.stdout[child.stdout.find('{'):])[0]
        keys=['summary','change_vs_largest','adjacent_change','legacy_128_prediction_change','last_pair_below_threshold']
        repeat_exact=value=={key:report[key] for key in keys}
    sources_unchanged=all(tree_sha(Path(i['path']))==i['sha256'] for i in evidence.values())
    g0=subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    result=dict(passed=selection_exact and raw_exact and measurement_max==0 and crops_exact and center_exact and code_exact
                and ssim_error<1e-11 and sam_error<1e-12 and psnr_error<1e-12 and change_error<1e-12
                and before==after and child.returncode==0 and repeat_exact and sources_unchanged and g0.returncode==0,
                selection_and_exclusion_exact=selection_exact,raw_conversion_exact=raw_exact,
                independent_rgb_float32_max_error=measurement_max,shared_observation_crops_exact=crops_exact,
                legacy_center_rgb_exact=center_exact,source_code_sha_exact=code_exact,
                independent_ssim_max_error=ssim_error,independent_sam_max_error_rad=sam_error,
                independent_psnr_max_error_db=psnr_error,independent_change_max_error=change_error,
                no_save_returncode=child.returncode,no_save_scores_exact=repeat_exact,no_save_engine_sha_unchanged=before==after,
                files_checked=len(before),source_runs_and_dataset_unchanged=sources_unchanged,
                g0_returncode=g0.returncode,g0_stdout=g0.stdout,stdout=child.stdout,stderr=child.stderr,optimizer_steps=0)
    output=Path(__file__).with_name('independent_audit.json');index=1
    while output.exists():
        output=Path(__file__).with_name(f'independent_audit_repeat{index}.json');index+=1
    write_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise RuntimeError('边界诊断独立审核未通过')
    names=['main_g5_boundary.py','config_g5_boundary.json','optics/g5_boundary.py','tests/test_g5_boundary.py','G5_BOUNDARY_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='boundary_diagnostic_passed',official_run=str(run),
               independent_audit=result,tests_run=4,paper_alignment_passed=False,whole_image_convergence_proved=False,
               sha256={str(ENGINE/n):file_sha(ENGINE/n) for n in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
