# -*- coding: utf-8 -*-
"""独立谱域传播与探测器重采样核对；一致性未通过不等于审核未执行。"""
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE))
from main_g1 import write_json
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha


def main():
    run=ENGINE/'results/g1_lightpipes/run_20261006_131933'
    report=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))['summary']
    evidence=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    with np.load(run/'arrays/crosscheck.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    phase_aligned=[];kernel_error=0.;interpolation=[]
    pitch=6.22e-6;q=8
    # 独立构造每个探测器像元的8×8中点，不调用原网格或积分函数。
    nodes=(np.arange(97*q)-(97*q-1)/2)*pitch/q
    yy,xx=np.meshgrid(nodes,nodes,indexing='ij')
    query=np.array([(yy/1e-6+304).ravel(),(xx/1e-6+304).ravel()])
    for wave in [420,540,660]:
        u1=data[f'u1_{wave}'];pin=float(np.sum(np.abs(u1)**2)*1e-12)
        for n in [1536,2048]:
            field=np.zeros((n,n),dtype=complex);offset=n//2-550;field[offset:offset+1101,offset:offset+1101]=u1
            frequency=np.fft.fftfreq(n,d=1e-6)
            # 使用精确π的标准近轴传递函数；LightPipes使用旧近似π，单位相位另对齐。
            transfer=np.exp(-1j*np.pi*(wave*1e-9)*.05*(frequency[:,None]**2+frequency[None,:]**2))
            output=np.fft.ifft2(np.fft.fft2(field)*transfer)
            own=output[n//2-304:n//2+305,n//2-304:n//2+305]
            saved=data[f'Forvard_{n}_native_{wave}'];overlap=np.vdot(own,saved)
            error=float(np.linalg.norm(own*overlap/abs(overlap)-saved)/np.linalg.norm(saved))
            phase_aligned.append(dict(wavelength_nm=wave,n=n,relative_l2=error,pi_constant_difference_included=True))
            del field,transfer,output
        for label in ['Forvard_1536','Forvard_2048','Fresnel_1536','reference']:
            value=data[f'{label}_native_{wave}'];intensity=np.abs(value)**2
            samples=map_coordinates(intensity,query,order=1,mode='constant',prefilter=False).reshape(97*q,97*q)
            power=samples.reshape(97,q,97,q).sum(axis=(1,3))*(pitch/q)**2
            kernel=power/pin
            reference=data[f'reference_kernel_{wave}'] if label=='reference' else data[f'{label}_kernel_{wave}']
            if label=='reference':
                interpolation.append(dict(wavelength_nm=wave,relative_l1=float(np.sum(np.abs(kernel-reference))/np.sum(reference)),
                                          relative_l2=float(np.linalg.norm(kernel-reference)/np.linalg.norm(reference)),
                                          status='empirical_reference_interpolation_difference_not_rigorous_bound'))
            else:kernel_error=max(kernel_error,float(np.max(np.abs(kernel-reference))))
    manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    code_exact=all(file_sha(ENGINE/n)==sha for n,sha in manifest.items())
    print('独立FFT/探测器重采样核对完成；执行无保存复算及完整SHA核对',flush=True)
    before=tree_sha(ENGINE)
    child=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g1_lightpipes.py'),'--no-save'],cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE);repeat=False
    if child.returncode==0:repeat=json.JSONDecoder().raw_decode(child.stdout[child.stdout.find('{'):])[0]==report
    unchanged=all(tree_sha(Path(i['path']))==i['sha256'] for i in evidence.values())
    g0=subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    result=dict(passed=report['numerical_health_passed'] and max(r['relative_l2'] for r in phase_aligned)<1e-5 and kernel_error<1e-12
                and code_exact and child.returncode==0 and repeat and before==after and unchanged and g0.returncode==0,
                independent_fft_unit_phase_aligned_errors=phase_aligned,independent_detector_kernel_max_error=kernel_error,
                empirical_reference_interpolation_difference=interpolation,source_code_sha_exact=code_exact,
                no_save_returncode=child.returncode,no_save_summary_exact=repeat,no_save_engine_sha_unchanged=before==after,files_checked=len(before),
                source_runs_and_package_unchanged=unchanged,g0_returncode=g0.returncode,g0_stdout=g0.stdout,stderr=child.stderr,
                all_cross_algorithm_comparisons_below_threshold=all(r['below_threshold'] for r in report['agreement']),paper_alignment_passed=False)
    output=Path(__file__).with_name('independent_audit.json');i=1
    while output.exists():output=Path(__file__).with_name(f'independent_audit_repeat{i}.json');i+=1
    write_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise RuntimeError('LightPipes执行结果独立审核未通过')
    names=['main_g1_lightpipes.py','config_g1_lightpipes.json','requirements_lightpipes.txt','G1_LIGHTPIPES_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='crosscheck_execution_audited_with_algorithm_discrepancies',official_run=str(run),
               independent_audit=result,paper_alignment_passed=False,algorithm_agreement_passed=False,
               sha256={str(ENGINE/n):file_sha(ENGINE/n) for n in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
