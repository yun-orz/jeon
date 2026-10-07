# -*- coding: utf-8 -*-
"""安装和正式GPU运行完成后独立验收；缺少GPU结果时拒绝通过。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.signal import fftconvolve
import torch

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT/'outputs/jeon2019_optics'
sys.path.insert(0, str(ENGINE))
from main_g1 import write_json
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha


def norm_error(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='正式GPU结果目录，按工程位置解析')
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute(): run = ENGINE/run
    if not torch.cuda.is_available():
        raise RuntimeError('实际CUDA不可用，拒绝验收GPU')
    if (run/'failed.json').exists():
        raise RuntimeError('该run存在失败记录，拒绝验收')
    report = json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    summary = report['summary']
    evidence = json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    codes = json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    with np.load(run/'arrays/gpu_checks.npz', allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    kernels, response = data['kernels'], data['response']

    def forward(x):
        fields = np.stack([fftconvolve(x[:, :, k], kernels[k], mode='same') for k in range(25)], axis=-1)
        return np.einsum('hwb,cb->hwc', fields, response)

    def adjoint(y):
        mixed = np.einsum('hwc,cb->hwb', y, response)
        return np.stack([fftconvolve(mixed[:, :, k], kernels[k, ::-1, ::-1], mode='same') for k in range(25)], axis=-1)

    own_forward = forward(data['optical_x'])
    optical = dict(forward=norm_error(data['optical_forward'], own_forward),
                   adjoint=norm_error(data['optical_adjoint'], adjoint(data['optical_y'])),
                   gradient=norm_error(data['optical_gradient'], adjoint(own_forward-data['optical_y'])))
    equation_errors = []
    backprojection = adjoint(data['validation_measurements'][0].astype(np.float64))
    for index in range(3):
        previous = data['gpu_trace_previous'][index].astype(np.float64)
        prior = data['gpu_trace_prior'][index].astype(np.float64)
        epsilon = data['gpu_trace_epsilon'][index]
        rho = data['gpu_trace_rho'][index]
        own_gradient = adjoint(forward(previous))-backprojection
        expected = previous-epsilon*(own_gradient+rho*(previous-prior))
        equation_errors.append(norm_error(data['gpu_trace_updated'][index], expected))
    error = max(abs(norm_error(data['gpu_predictions'],data['new_cpu_predictions'])-
                    summary['prediction_checks']['gpu_vs_new_cpu']),
                abs(norm_error(data['new_cpu_predictions'],data['saved_cpu_predictions'])-
                    summary['prediction_checks']['new_cpu_vs_saved_cpu']))
    code_exact = all(file_sha(ENGINE/name) == sha for name, sha in codes.items())
    tests = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_g3.py','-v'],
                           cwd=str(ENGINE),capture_output=True,text=True,encoding='utf-8')
    before = tree_sha(ENGINE)
    effective = json.loads((run/'config_effective.json').read_text(encoding='utf-8'))
    # 默认配置须与正式运行一致，避免误审核另一来源或修改后的阈值。
    config = json.loads((ENGINE/'config_g3_gpu.json').read_text(encoding='utf-8'))
    config_exact = all(effective[key] == value for key, value in config.items() if key != 'runtime')
    child = subprocess.run([sys.executable,'-B',str(ENGINE/'main_g3_gpu.py'),'--no-save'],cwd=str(ROOT/'work'),
                           capture_output=True,text=True,encoding='utf-8')
    repeat = False
    if child.returncode == 0 and '{' in child.stdout:
        repeat = json.JSONDecoder().raw_decode(child.stdout[child.stdout.find('{'):])[0] == summary
    after = tree_sha(ENGINE)
    sources = all(tree_sha(Path(item['path'])) == item['sha256'] for item in evidence.values())
    cpu_probe = subprocess.run(['D:/dev/python/python3.10.4/python.exe','-B',str(Path(__file__).with_name('cpu_environment_snapshot.py'))],
                               capture_output=True,text=True,encoding='utf-8')
    old_cpu = json.loads(Path(__file__).with_name('cpu_environment_before.json').read_text(encoding='utf-8-sig'))
    cpu_unchanged = cpu_probe.returncode == 0 and json.loads(cpu_probe.stdout) == old_cpu
    g0 = subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    passed = (summary['gpu_validation_passed'] and max(optical.values()) < 1e-10 and max(equation_errors) < 1e-5
              and error < 1e-12 and code_exact and config_exact and tests.returncode == 0
              and child.returncode == 0 and repeat and before == after and sources and cpu_unchanged and g0.returncode == 0)
    result = dict(passed=bool(passed),independent_scipy_optical=optical,independent_equation21_errors=equation_errors,
                  summary_metric_max_error=error,source_code_exact=code_exact,config_exact=config_exact,
                  tests_run=4,tests_returncode=tests.returncode,tests_stderr=tests.stderr,
                  no_save_returncode=child.returncode,no_save_summary_exact=repeat,engine_sha_unchanged=before==after,
                  files_checked=len(before),sources_unchanged=sources,original_cpu_environment_unchanged=cpu_unchanged,
                  g0_returncode=g0.returncode,g0_stdout=g0.stdout,no_save_stderr=child.stderr,paper_alignment_passed=False)
    dest = Path(__file__).with_name('independent_audit.json')
    index = 1
    while dest.exists():
        dest = Path(__file__).with_name(f'independent_audit_repeat{index}.json')
        index += 1
    write_json(dest, result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not passed: raise RuntimeError('GPU独立审核未通过')
    final = Path(__file__).with_name('最终审核.json')
    index = 1
    while final.exists():
        final = Path(__file__).with_name(f'最终审核_repeat{index}.json')
        index += 1
    write_json(final,dict(status='gpu_execution_parity_audited',official_run=str(run),independent_audit=result,
                         run_sha256=tree_sha(run),code_sha256=codes,paper_alignment_passed=False))


if __name__ == '__main__':
    main()
