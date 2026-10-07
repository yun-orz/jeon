# -*- coding: utf-8 -*-
"""GPU安装前的独立CPU预审；不将该预审视为GPU验收。"""
from pathlib import Path
import json
import subprocess
import sys
import time
import numpy as np
from scipy.signal import fftconvolve
import torch

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT/'outputs/jeon2019_optics'
sys.path.insert(0, str(ENGINE))
from main_g1 import write_json
from main_g3_gpu import make_model, predict_trace, optical_check, smoke_step, relative
from main_g5_retest import checked_source
from optics.g2_rgb import RGBForward
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha
from cpu_environment_snapshot import snapshot


def main():
    start = time.perf_counter()
    config = json.loads((ENGINE/'config_g3_gpu.json').read_text(encoding='utf-8'))
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(config['seed'])
    source = ENGINE/config['source_run']
    before, _ = checked_source(source, ENGINE/config['source_audit'])
    payload = torch.load(source/'arrays/best_checkpoint.pt', map_location='cpu', weights_only=True)
    g4 = Path(json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['g4']['path'])
    g3 = Path(json.loads((g4/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    g2 = Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(g2/'arrays/g2_measurement.npz', allow_pickle=False) as z:
        op = RGBForward(z['kernels'], z['response'])
    with np.load(source/'arrays/expanded_subset.npz', allow_pickle=False) as z:
        measurements = z['validation_measurements'].copy()
        reference = z['best_validation_predictions'].copy()
        train_measurement = z['train_measurements'][0].copy()
        train_target = z['train_targets'][0].copy()
    optical, values = optical_check(op, 'cpu', config['seed'])
    x, y = values['optical_x'], values['optical_y']
    # 另一逐波段卷积实现，不导入受审核的FFT卷积函数。
    blurred = np.stack([fftconvolve(x[:, :, k], op.kernels[k], mode='same') for k in range(25)], axis=-1)
    forward = np.einsum('hwb,cb->hwc', blurred, op.response)
    mixed = np.einsum('hwc,cb->hwb', y, op.response)
    adjoint = np.stack([fftconvolve(mixed[:, :, k], op.kernels[k, ::-1, ::-1], mode='same') for k in range(25)], axis=-1)
    residual = np.einsum('hwc,cb->hwb', forward-y, op.response)
    gradient = np.stack([fftconvolve(residual[:, :, k], op.kernels[k, ::-1, ::-1], mode='same') for k in range(25)], axis=-1)
    independent = dict(forward=relative(values['optical_forward'], forward),
                       adjoint=relative(values['optical_adjoint'], adjoint),
                       gradient=relative(values['optical_gradient'], gradient))
    model = make_model(payload, op, 'cpu')
    prediction, trace = predict_trace(model, measurements)
    exact = np.array_equal(prediction, reference)
    prediction_error = relative(prediction, reference)
    initial_parameter = next(model.parameters()).detach().clone()
    smoke, _, _ = smoke_step(model, train_measurement, train_target, config['smoke'],
                             payload['dataset']['halo'], payload['dataset']['core_size'])
    changed = float(torch.linalg.vector_norm(next(model.parameters()).detach()-initial_parameter))
    tests = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_g3.py','-v'],
                           cwd=str(ENGINE),capture_output=True,text=True,encoding='utf-8')
    # 验证旧CPU解释器运行GPU入口时明确失败，且不会产生伪GPU结果。
    engine_before = tree_sha(ENGINE)
    rejection = subprocess.run([sys.executable,'-B',str(ENGINE/'main_g3_gpu.py')],cwd=str(ROOT/'work'),
                               capture_output=True,text=True,encoding='utf-8')
    cuda_rejected = rejection.returncode != 0 and 'CUDA不可用' in rejection.stderr
    no_fake_output = tree_sha(ENGINE) == engine_before
    environment_before = json.loads(Path(__file__).with_name('cpu_environment_before.json').read_text(encoding='utf-8-sig'))
    environment_after = snapshot()
    unchanged = environment_before == environment_after and tree_sha(source) == before
    g0 = subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8')
    passed = (max(optical.values()) < config['thresholds']['optical_float64'] and
              max(independent.values()) < config['thresholds']['optical_float64'] and
              prediction_error < config['thresholds']['prediction_float32'] and changed > 0 and
              tests.returncode == 0 and cuda_rejected and no_fake_output and unchanged and g0.returncode == 0)
    result = dict(cpu_preflight_passed=bool(passed),gpu_validation_passed=False,gpu_execution_verified=False,
                  status='cpu_preflight_only_gpu_installation_pending',optical_checks=optical,
                  independent_scipy_checks=independent,saved_checkpoint_predictions_exact=exact,
                  saved_checkpoint_predictions_relative=prediction_error,validation_patches=8,
                  cpu_one_step_smoke=smoke,first_parameter_change_norm=changed,
                  test_returncode=tests.returncode,tests_run=4,tests_stderr=tests.stderr,
                  missing_cuda_explicitly_rejected=cuda_rejected,no_fake_gpu_output_written=no_fake_output,
                  original_cpu_environment_and_source_unchanged=unchanged,cpu_environment=environment_after,
                  g0_returncode=g0.returncode,g0_stdout=g0.stdout,
                  paper_alignment_passed=False,elapsed_seconds=time.perf_counter()-start,
                  code_sha256={name:file_sha(ENGINE/name) for name in ['main_g3_gpu.py','config_g3_gpu.json','requirements_g3_gpu.txt','G3_GPU_README.md']})
    path = Path(__file__).with_name('cpu_preflight.json')
    index = 1
    while path.exists():
        path = Path(__file__).with_name(f'cpu_preflight_repeat{index}.json')
        index += 1
    write_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError('CPU预审未通过；结果保留')


if __name__ == '__main__':
    main()
