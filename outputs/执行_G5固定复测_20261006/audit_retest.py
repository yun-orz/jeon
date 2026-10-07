# -*- coding: utf-8 -*-
"""固定复测独立审核：测量再编码、另一指标表达式、只读运行与来源。"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from scipy.signal import fftconvolve
ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE));sys.path.insert(0,str(ENGINE/'tests'))
from main_g1 import write_json
from optics.stage02c_source import tree_sha
from optics.g4_data import file_sha
from test_g5 import reference_ssim


def main():
    run=ENGINE/'results/g5_retest/run_20261006_093137'
    status=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    rows=json.loads((run/'metrics/scores.json').read_text(encoding='utf-8'))
    metadata=json.loads((run/'dataset_manifest.json').read_text(encoding='utf-8'))
    evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    old=Path(evidence['original_test']['path']);new=Path(evidence['expanded_training']['path'])
    with np.load(run/'arrays/fixed_scene_retest.npz',allow_pickle=False) as z:data={key:z[key].copy() for key in z.files}
    with np.load(old/'arrays/heldout_test.npz',allow_pickle=False) as z:
        fixed_input_exact=np.array_equal(data['targets_context'],z['targets_context']) and np.array_equal(data['measurements'],z['measurements']) and np.array_equal(data['original_predictions_context'],z['predictions_context'])
    g4=Path(json.loads((new/'source_evidence.json').read_text(encoding='utf-8'))['g4']['path'])
    g3=Path(json.loads((g4/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    g2=Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(g2/'arrays/g2_measurement.npz',allow_pickle=False) as z:kernels=z['kernels'].copy();response=z['response'].copy()
    h=metadata['halo'];s=metadata['core_size'];conv_error=0.;fit_error=0.;ssim_error=0.;sam_error=0.;psnr_error=0.
    for label,key,rgbkey in [('original_g4','original_predictions_context','original_recoded_rgb'),('expanded_g4','expanded_predictions_context','expanded_recoded_rgb')]:
        independent_rgb=[]
        for p in data[key]:
            bands=np.stack([fftconvolve(p[:,:,k].astype(float),kernels[k],mode='same') for k in range(25)],axis=-1)
            independent_rgb.append(bands@response.T)
        independent_rgb=np.stack(independent_rgb)
        conv_error=max(conv_error,float(np.max(np.abs(independent_rgb-data[rgbkey]))))
        y=data['measurements'][:,h:h+s,h:h+s].astype(float);difference=independent_rgb[:,h:h+s,h:h+s]-y
        relative=float(np.linalg.norm(difference.ravel())/np.linalg.norm(y.ravel()))
        fit_error=max(fit_error,abs(relative-status['measurement_fit'][label]['center_relative_l2']))
        for i,(a,b) in enumerate(zip(data['targets_context'][:,h:h+s,h:h+s],data[key][:,h:h+s,h:h+s])):
            a=a.astype(float);b=b.astype(float);row=rows[label][i]
            local=[reference_ssim(a[:,:,k],b[:,:,k]).mean() for k in range(25)]
            ssim_error=max(ssim_error,float(np.max(np.abs(np.array(local)-row['ssim_per_band']))))
            psnr_error=max(psnr_error,abs(float(20*np.log10(1/np.sqrt(np.mean((a-b)**2))))-row['psnr_cube_db']))
            na=np.linalg.norm(a,axis=-1,keepdims=True);nb=np.linalg.norm(b,axis=-1,keepdims=True)
            valid=(na[:,:,0]>status['protocol']['zero_norm_threshold'])&(nb[:,:,0]>status['protocol']['zero_norm_threshold'])
            u=a[valid]/na[valid];v=b[valid]/nb[valid]
            angle=2*np.arctan2(np.linalg.norm(u-v,axis=-1),np.linalg.norm(u+v,axis=-1))
            sam_error=max(sam_error,abs(float(angle.mean())-row['sam_mean_rad']))
    before=tree_sha(ENGINE)
    child=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g5_retest.py'),'--no-save'],cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE);recomputed=None
    if child.returncode==0:recomputed=json.JSONDecoder().raw_decode(child.stdout[child.stdout.find('{'):])[0]
    repeat_exact=recomputed==dict(summary=status['summary'],measurement_fit=status['measurement_fit'])
    unchanged=all(tree_sha(Path(item['path']))==item['sha256'] for item in evidence.values())
    g0=subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    result=dict(passed=fixed_input_exact and conv_error<1e-12 and fit_error<1e-12 and ssim_error<1e-11 and sam_error<1e-12 and psnr_error<1e-12
        and before==after and child.returncode==0 and repeat_exact and unchanged and g0.returncode==0,
        fixed_inputs_and_original_predictions_exact=fixed_input_exact,independent_rgb_recoding_max_error=conv_error,
        independent_measurement_relative_l2_error=fit_error,independent_ssim_max_error=ssim_error,
        independent_sam_max_error_rad=sam_error,independent_psnr_max_error_db=psnr_error,
        no_save_returncode=child.returncode,no_save_scores_exact=repeat_exact,no_save_sha_unchanged=before==after,files_checked=len(before),
        source_runs_unchanged=unchanged,optimizer_steps=0,fresh_blind_test=False,stdout=child.stdout,stderr=child.stderr,
        g0_returncode=g0.returncode,g0_stdout=g0.stdout)
    output=Path(__file__).with_name('independent_audit.json');number=1
    while output.exists():
        output=Path(__file__).with_name(f'independent_audit_repeat{number}.json');number+=1
    write_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise RuntimeError('固定复测审核未通过')
    names=['main_g5_retest.py','config_g5_retest.json','optics/g5_retest_metrics.py','tests/test_g5_retest.py','G5_RETEST_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='fixed_scene_retest_passed',official_run=str(run),independent_audit=result,
        tests_run=5,paper_alignment_passed=False,fresh_blind_test=False,sha256={str(ENGINE/n):file_sha(ENGINE/n) for n in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
