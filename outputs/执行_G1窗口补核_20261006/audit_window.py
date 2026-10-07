# -*- coding: utf-8 -*-
"""独立二维Fresnel求和、像元积分、径向累计与只读复算审核。"""
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE))
from main_g1 import write_json
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha


def main():
    run=ENGINE/'results/g1_window/run_20261006_131121'
    report=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    source=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    with np.load(run/'arrays/control_540nm.npz',allow_pickle=False) as z:field={k:z[k].copy() for k in z.files}
    with np.load(run/'arrays/window_bank.npz',allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    alignment=Path(next(p for p in source if 'g1_alignment' in p))
    with np.load(alignment/'arrays/height_cw_fixed.npz',allow_pickle=False) as z:axis=z['x_m'].copy();mask=z['mask'].copy()
    xx,yy=np.meshgrid(axis,axis);dx=float(axis[1]-axis[0]);wave=float(field['wavelength_m']);distance=.05
    peak=float(np.max(np.abs(field['u2_complex'])));errors=[]
    nodes=field['node_x_m'];last=len(nodes)-1
    for row,col in [(len(nodes)//2,len(nodes)//2),(len(nodes)//2+100,len(nodes)//2-50),(last,last)]:
        x,y=nodes[col],nodes[row]
        kernel=np.exp(1j*np.pi*((x-xx)**2+(y-yy)**2)/(wave*distance))
        # 全局相位按kz求值，避免大相位乘除顺序引入无物理意义的舍入差。
        direct=np.exp(1j*(2*np.pi/wave)*distance)/(1j*wave*distance)*np.sum(field['u1_complex']*kernel)*dx**2
        errors.append(float(abs(direct-field['u2_complex'][row,col])/peak))
    intensity=np.abs(field['u2_complex'])**2
    # 明确逐像元累加，避免调用原积分函数。
    power=np.array([[np.sum(intensity[y*8:(y+1)*8,x*8:(x+1)*8])*(float(bank['pitch_m'])/8)**2 for x in range(161)] for y in range(161)])
    power_error=float(np.max(np.abs(power-field['pixel_power']))/np.max(field['pixel_power']))
    pin=float(np.sum(np.abs(field['u1_complex'])**2)*dx**2)
    pin_error=abs(pin/float(field['Pin'])-1)
    index=report['wavelengths_nm'].index(540)
    kernel_error=float(np.max(np.abs(power/pin-bank['expanded_kernels'][index])))
    x=bank['pixel_x_m'];rad=np.hypot(x[:,None],x[None,:]);inside=rad<=min(abs(x[0]),x[-1]);r=rad[inside]
    order=np.argsort(r);ordered_r=r[order];radius_error=0.
    for image,sizes in zip(bank['expanded_kernels'],report['expanded_sizes']):
        accumulated=np.cumsum(image[inside][order])
        for denominator,label in [(1.,'input'),(float(image.sum()),'window')]:
            for fraction in [.5,.8]:
                hits=np.flatnonzero(accumulated>=denominator*fraction)
                expected=float(ordered_r[hits[0]]*1e6) if len(hits) else None
                stored=sizes[f'R{int(fraction*100)}_{label}_um']
                if expected is None or stored is None:
                    if expected!=stored:raise RuntimeError('径向阈值无定义状态不一致')
                else:radius_error=max(radius_error,abs(expected-stored))
    manifest=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    code_exact=all(file_sha(ENGINE/n)==sha for n,sha in manifest.items())
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_g1*.py','-v'],cwd=str(ENGINE),capture_output=True,text=True,encoding='utf-8')
    print('独立复场/积分/半径核对完成；进行无保存复算与完整SHA核对',flush=True)
    before=tree_sha(ENGINE)
    child=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g1_window.py'),'--no-save'],cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE);repeat=False
    if child.returncode==0:
        result=json.JSONDecoder().raw_decode(child.stdout[child.stdout.find('{'):])[0]
        repeat=result==dict(numerical_validation_passed=report['numerical_validation_passed'],design_phase_error=report['design_phase_integer_cycle_max_error'],
                           checks=report['checks'],old_sizes=report['old_sizes'],expanded_sizes=report['expanded_sizes'])
    unchanged=all(tree_sha(Path(p))==sha for p,sha in source.items())
    g0=subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    audit=dict(passed=report['numerical_validation_passed'] and max(errors)<1e-12 and power_error<1e-12 and pin_error<1e-12 and kernel_error<1e-12
               and radius_error<1e-9 and code_exact and tests.returncode==0 and child.returncode==0 and repeat and before==after and unchanged and g0.returncode==0,
               direct_fresnel_relative_to_peak_errors=errors,independent_pixel_power_relative_max_error=power_error,
               independent_pin_relative_error=pin_error,independent_kernel_max_error=kernel_error,independent_radius_max_error_um=radius_error,
               source_code_sha_exact=code_exact,tests_returncode=tests.returncode,tests_stdout=tests.stdout,tests_stderr=tests.stderr,
               no_save_returncode=child.returncode,no_save_values_exact=repeat,no_save_engine_sha_unchanged=before==after,files_checked=len(before),
               source_runs_unchanged=unchanged,g0_returncode=g0.returncode,g0_stdout=g0.stdout,stderr=child.stderr)
    output=Path(__file__).with_name('independent_audit.json');counter=1
    while output.exists():output=Path(__file__).with_name(f'independent_audit_repeat{counter}.json');counter+=1
    write_json(output,audit);print(json.dumps(audit,ensure_ascii=False,indent=2))
    if not audit['passed']:raise RuntimeError('窗口诊断独立审核未通过')
    names=['main_g1_window.py','config_g1_window.json','G1_WINDOW_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='window_diagnostic_passed',official_run=str(run),independent_audit=audit,
        paper_alignment_passed=False,sha256={str(ENGINE/n):file_sha(ENGINE/n) for n in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
