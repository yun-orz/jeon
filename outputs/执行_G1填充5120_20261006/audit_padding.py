# -*- coding: utf-8 -*-
"""用另一FFT后端、可分离传递函数及像元积分独立审核5120结果。"""
from pathlib import Path
import json
import sys
import subprocess
import time
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / 'outputs/jeon2019_optics'
sys.path.insert(0, str(ENGINE))
from main_g1 import write_json
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha


def unique_json(name, value):
    """审核重复执行时新增记录，保留已有审核及失败记录。"""
    path = Path(__file__).with_name(name)
    index = 1
    while path.exists():
        path = Path(__file__).with_name(f'{Path(name).stem}_repeat{index}.json')
        index += 1
    write_json(path, value)
    return path


def independent_difference(a, b):
    delta = np.abs(a - b)
    return dict(relative_l1=float(delta.sum() / np.abs(b).sum()),
                relative_l2=float(np.sqrt(np.sum(delta**2) / np.sum(np.abs(b)**2))),
                max_absolute=float(delta.max()))


def main():
    start = time.perf_counter()
    run = ENGINE / 'results/g1_padding/run_20261006_133832'
    summary = json.loads((run / 'metrics/validation.json').read_text(encoding='utf-8'))['summary']
    evidence = json.loads((run / 'source_evidence.json').read_text(encoding='utf-8'))
    manifest = json.loads((run / 'source_manifest.json').read_text(encoding='utf-8'))
    with np.load(run / 'arrays/padding.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    fft_errors = []
    kernel_error = 0.0
    metric_error = 0.0
    input_exact = True
    axis_exact = np.allclose(data['native_axis_m'], np.arange(-304, 305)*1e-6, rtol=0, atol=1e-18)
    qnodes = (np.arange(776) - 775/2) * (6.22e-6 / 8)
    yq, xq = np.meshgrid(qnodes, qnodes, indexing='ij')
    query = np.array([(yq/1e-6 + 304).ravel(), (xq/1e-6 + 304).ravel()])
    source = Path(evidence['sampling']['path'])
    with np.load(source / 'arrays/sampling.npz', allow_pickle=False) as old:
        input_exact = str(old['height_fingerprint']) == summary['height_fingerprint']
        for wave, row in zip([420, 540, 660], summary['rows']):
            print(f'独立NumPy FFT核对{wave}nm的5120域', flush=True)
            wavelength = wave*1e-9
            u1 = data[f'u1_{wave}']
            input_exact = input_exact and np.array_equal(u1, old[f'u1_{wave}'])
            input_exact = input_exact and np.array_equal(data[f'reference_kernel_{wave}'], old[f'reference_kernel_{wave}'])
            input_exact = input_exact and np.array_equal(data[f'LightPipes_4096_native_{wave}'], old[f'Forvard_4096_native_{wave}'])
            input_exact = input_exact and np.array_equal(data[f'LightPipes_4096_kernel_{wave}'], old[f'Forvard_4096_kernel_{wave}'])
            pin = float(np.sum(np.abs(u1)**2)*1e-12)
            n = 5120
            field = np.zeros((n, n), dtype=np.complex128)
            offset = n//2 - 550
            field[offset:offset+1101, offset:offset+1101] = u1
            spectrum = np.fft.fft2(field)
            del field
            frequency = np.fft.fftfreq(n, d=1e-6)
            # 独立用两次一维因子相乘；不调用受审核的传播函数。
            factor = np.exp(-1j*np.pi*wavelength*.05*frequency**2)
            spectrum *= factor[:, None]
            spectrum *= factor[None, :]
            output = np.fft.ifft2(spectrum)
            del spectrum
            power = sum(float(np.sum(np.abs(output[y:y+32])**2)) for y in range(0, n, 32))*1e-12
            crop = output[n//2-304:n//2+305, n//2-304:n//2+305].copy()
            del output
            saved = data[f'equivalent_5120_native_{wave}']
            overlap = np.vdot(crop, saved)
            phase = overlap / abs(overlap)
            error = float(np.linalg.norm(crop*phase-saved) / np.linalg.norm(saved))
            fft_errors.append(dict(wavelength_nm=wave,unit_phase_aligned_field_relative_l2=error,
                                   independent_full_grid_power_over_input=power/pin))
            for label in ['LightPipes_4096', 'equivalent_4096', 'equivalent_5120']:
                intensity = np.abs(data[f'{label}_native_{wave}'])**2
                samples = map_coordinates(intensity, query, order=1, prefilter=False).reshape(776, 776)
                kernel = samples.reshape(97, 8, 97, 8).sum((1, 3))*(6.22e-6/8)**2/pin
                kernel_error = max(kernel_error, float(np.max(np.abs(kernel-data[f'{label}_kernel_{wave}']))))
            new = data[f'equivalent_5120_kernel_{wave}']
            for target, key in [(data[f'LightPipes_4096_kernel_{wave}'], 'change_4096_to_5120'),
                                (data[f'reference_kernel_{wave}'], 'reference_difference')]:
                independently = independent_difference(new, target)
                metric_error = max(metric_error, max(abs(independently[k]-row[key][k]) for k in independently))
            metric_error = max(metric_error, abs(float(new.sum())-row['detector_efficiency']))
    code_exact = all(file_sha(ENGINE/name) == digest for name, digest in manifest.items())
    tests = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_g1_padding.py','-v'],
                           cwd=str(ENGINE),capture_output=True,text=True,encoding='utf-8')
    print('独立传播及像元检查完成；进行异启动目录无保存复算和基线核对',flush=True)
    before = tree_sha(ENGINE)
    child = subprocess.run([sys.executable,'-B',str(ENGINE/'main_g1_padding.py'),'--no-save'],
                           cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after = tree_sha(ENGINE)
    repeat = False
    if child.returncode == 0 and '{' in child.stdout:
        repeat = json.JSONDecoder().raw_decode(child.stdout[child.stdout.find('{'):])[0] == summary
    unchanged = all(tree_sha(Path(item['path'])) == item['sha256'] for item in evidence.values())
    g0 = subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),
                        capture_output=True,text=True,encoding='utf-8')
    field_passed = max(row['unit_phase_aligned_field_relative_l2'] for row in fft_errors) < 1e-12
    power_passed = max(abs(row['independent_full_grid_power_over_input']-1) for row in fft_errors) < 1e-12
    gates_passed = summary['last_padding_pair_below_threshold'] and summary['reference_below_threshold']
    passed = (summary['numerical_health_passed'] and gates_passed and field_passed and power_passed
              and input_exact and axis_exact and kernel_error < 1e-12 and metric_error < 1e-12
              and code_exact and tests.returncode == 0 and child.returncode == 0
              and repeat and before == after and unchanged and g0.returncode == 0)
    result = dict(passed=bool(passed), independent_fft_checks=fft_errors,
                  independent_detector_kernel_max_error=kernel_error,independent_summary_metric_max_error=metric_error,
                  fixed_input_and_reference_exact=bool(input_exact),native_coordinate_axis_verified=bool(axis_exact),
                  source_code_sha_exact=code_exact,tests_run=3,tests_returncode=tests.returncode,tests_stderr=tests.stderr,
                  no_save_returncode=child.returncode,no_save_summary_exact=repeat,no_save_engine_sha_unchanged=before==after,
                  files_checked=len(before),source_runs_and_package_unchanged=unchanged,
                  g0_returncode=g0.returncode,g0_stdout=g0.stdout,no_save_stderr=child.stderr,
                  last_padding_pair_below_threshold=summary['last_padding_pair_below_threshold'],
                  reference_below_threshold=summary['reference_below_threshold'],direct_lightpipes_5120_executed=False,
                  gpu_execution_verified=False,paper_alignment_passed=False,infinite_domain_convergence_proved=False,
                  elapsed_seconds=time.perf_counter()-start)
    unique_json('independent_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not passed:
        raise RuntimeError('5120补核独立审核未通过，请保留记录并核查')
    names = ['main_g1_padding.py','config_g1_padding.json','optics/g1_padding.py','tests/test_g1_padding.py','G1_PADDING_README.md']
    unique_json('最终审核.json',dict(status='finite_padding_pair_and_reference_audited',official_run=str(run),
                independent_audit=result,tests_run=3,paper_alignment_passed=False,infinite_domain_convergence_proved=False,
                direct_lightpipes_5120_executed=False,gpu_execution_verified=False,
                sha256={str(ENGINE/name):file_sha(ENGINE/name) for name in names},run_sha256=tree_sha(run)))


if __name__ == '__main__':
    main()
