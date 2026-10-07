# -*- coding: utf-8 -*-
"""用独立Gauss核积分、标准FFT及另一探测器积分验证拆分结果。"""
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE))
from main_g1 import write_json
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha


def main():
    run=ENGINE/'results/g1_lp_sampling/run_20261006_133000'
    summary=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))['summary']
    evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    with np.load(run/'arrays/sampling.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    points=[(0,0),(25,-12),(300,-300)];nodes,weights=np.polynomial.legendre.leggauss(24)
    gauss_errors=[];fft_errors=[];kernel_error=0.
    qnodes=(np.arange(776)-(776-1)/2)*6.22e-6/8
    yq,xq=np.meshgrid(qnodes,qnodes,indexing='ij');query=np.array([(yq/1e-6+304).ravel(),(xq/1e-6+304).ravel()])
    for wave in [420,540,660]:
        wavelength=wave*1e-9;u1=data[f'u1_{wave}'];axis=data['input_axis_m'];own=[];cell=[];lp=[]
        for y,x in points:
            vectors=[]
            for coordinate in [y*1e-6,x*1e-6]:
                shift=coordinate-axis[:,None]+1e-6*nodes
                vectors.append(1e-6*np.sum(weights*np.exp(1j*np.pi*shift**2/(wavelength*.05)),axis=1)/2)
            own.append(vectors[0]@u1@vectors[1]/(1j*wavelength*.05))
            cell.append(data[f'cell_averaged_native_{wave}'][304+y,304+x])
            lp.append(data[f'Fresnel_adapted_2048_native_{wave}'][304+y,304+x])
        own=np.array(own);cell=np.array(cell);lp=np.array(lp);overlap=np.vdot(own,lp)
        gauss_errors.append(dict(wavelength_nm=wave,cell_relative_l2=float(np.linalg.norm(own-cell)/np.linalg.norm(cell)),
                                library_unit_phase_relative_l2=float(np.linalg.norm(own*overlap/abs(overlap)-lp)/np.linalg.norm(lp))))
        n=4096;field=np.zeros((n,n),complex);offset=n//2-550;field[offset:offset+1101,offset:offset+1101]=u1
        frequency=np.fft.fftfreq(n,d=1e-6)
        transfer=np.exp(-1j*np.pi*wavelength*.05*(frequency[:,None]**2+frequency[None,:]**2))
        output=np.fft.ifft2(np.fft.fft2(field)*transfer);crop=output[n//2-304:n//2+305,n//2-304:n//2+305]
        saved=data[f'Forvard_4096_native_{wave}'];overlap=np.vdot(crop,saved)
        fft_errors.append(dict(wavelength_nm=wave,relative_l2=float(np.linalg.norm(crop*overlap/abs(overlap)-saved)/np.linalg.norm(saved))))
        del field,transfer,output,crop
        pin=float(np.sum(abs(u1)**2)*1e-12)
        for label in ['Forvard_3072','Forvard_4096','Fresnel_adapted_1536','Fresnel_adapted_2048','cell_averaged']:
            intensity=abs(data[f'{label}_native_{wave}'])**2
            samples=map_coordinates(intensity,query,order=1,prefilter=False).reshape(776,776)
            kernel=samples.reshape(97,8,97,8).sum((1,3))*(6.22e-6/8)**2/pin
            kernel_error=max(kernel_error,float(np.max(abs(kernel-data[f'{label}_kernel_{wave}']))))
    manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    code_exact=all(file_sha(ENGINE/n)==sha for n,sha in manifest.items())
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_g1_lp_sampling.py','-v'],cwd=str(ENGINE),capture_output=True,text=True,encoding='utf-8')
    print('独立Gauss积分、FFT和像元核对完成；执行无保存复算与SHA核对',flush=True)
    before=tree_sha(ENGINE)
    child=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g1_lp_sampling.py'),'--no-save'],cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE);repeat=False
    if child.returncode==0:repeat=json.JSONDecoder().raw_decode(child.stdout[child.stdout.find('{'):])[0]==summary
    unchanged=all(tree_sha(Path(i['path']))==i['sha256'] for i in evidence.values())
    g0=subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    passed=summary['numerical_health_passed'] and summary['cell_model_verified'] and max(r['cell_relative_l2'] for r in gauss_errors)<1e-10 and max(r['library_unit_phase_relative_l2'] for r in gauss_errors)<1e-10
    passed=passed and max(r['relative_l2'] for r in fft_errors)<1e-5 and kernel_error<1e-12 and code_exact and tests.returncode==0 and child.returncode==0 and repeat and before==after and unchanged and g0.returncode==0
    result=dict(passed=bool(passed),independent_gauss_point_errors=gauss_errors,independent_fft_phase_aligned_errors=fft_errors,
                independent_detector_kernel_max_error=kernel_error,source_code_sha_exact=code_exact,tests_returncode=tests.returncode,tests_stderr=tests.stderr,
                no_save_returncode=child.returncode,no_save_summary_exact=repeat,no_save_engine_sha_unchanged=before==after,files_checked=len(before),
                source_runs_and_package_unchanged=unchanged,g0_returncode=g0.returncode,g0_stdout=g0.stdout,stderr=child.stderr,
                last_padding_pair_below_threshold=summary['last_padding_pair_below_threshold'],paper_alignment_passed=False)
    path=Path(__file__).with_name('independent_audit.json');i=1
    while path.exists():path=Path(__file__).with_name(f'independent_audit_repeat{i}.json');i+=1
    write_json(path,result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise RuntimeError('采样拆分独立审核未通过')
    names=['main_g1_lp_sampling.py','config_g1_lp_sampling.json','optics/g1_lp_sampling.py','tests/test_g1_lp_sampling.py','G1_LP_SAMPLING_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='sampling_decomposition_audited_padding_convergence_pending',official_run=str(run),independent_audit=result,
              tests_run=3,paper_alignment_passed=False,sha256={str(ENGINE/n):file_sha(ENGINE/n) for n in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
