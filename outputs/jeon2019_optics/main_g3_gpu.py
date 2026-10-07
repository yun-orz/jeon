# -*- coding: utf-8 -*-
"""独立GPU环境验收：旧模型一致性、物理伴随及一次训练通路检查。"""
import os
# 在导入Torch、创建CUDA上下文前指定确定性矩阵工作区。
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from main_g1 import write_json
from main_g3 import tensor, array
from main_stage02b import allocate_unique_run_dir
from main_g5_retest import checked_source
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB, HQSDecoder
from optics.g4_data import file_sha
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path, effective_runtime, configure_plotting
from optics.runutil import environment_info

ROOT = Path(__file__).resolve().parent


def relative(a, b):
    """双精度汇总差异，基准范数为零时用正数保护。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a-b) / max(float(np.linalg.norm(b)), 1e-30))


def optical_check(op, device, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(12, 13, 25))
    y = rng.normal(size=(12, 13, 3))
    model = TorchRGB(op.kernels, op.response).double().to(device)
    tx = tensor(x, torch.float64).to(device).requires_grad_()
    ty = tensor(y, torch.float64).to(device)
    ax = model(tx)
    aty = model.adjoint(ty)
    loss = .5*(ax-ty).square().sum()
    loss.backward()
    forward = array(ax)
    adjoint = array(aty)
    gradient = array(tx.grad)
    dot = float(abs((ax*ty).sum().detach()-(tx*aty).sum().detach()) /
                (torch.linalg.vector_norm(ax.detach())*torch.linalg.vector_norm(ty)))
    checks = dict(numpy_forward_relative=relative(forward, op.forward(x)),
                  numpy_adjoint_relative=relative(adjoint, op.adjoint(y)),
                  autograd_relative=relative(gradient, op.adjoint(op.forward(x)-y)),
                  adjoint_dot_relative=dot)
    return checks, dict(optical_x=x, optical_y=y, optical_forward=forward,
                        optical_adjoint=adjoint, optical_gradient=gradient)


def make_model(payload, op, device):
    model = HQSDecoder(TorchRGB(op.kernels, op.response), payload['network_config']).float()
    model.load_state_dict(payload['state_dict'], strict=True)
    if not torch.equal(model.operator.kernels, torch.tensor(op.kernels, dtype=torch.float32)):
        raise ValueError('checkpoint光学核与原始G2不一致')
    if not torch.equal(model.operator.response, torch.tensor(op.response, dtype=torch.float32)):
        raise ValueError('checkpoint响应与原始G2不一致')
    return model.to(device)


def predict_trace(model, measurements):
    device = next(model.parameters()).device
    outputs = []
    trace_data = {}
    model.eval()
    with torch.no_grad():
        for index, value in enumerate(measurements):
            prediction, initial, trace = model(tensor(value).to(device), return_trace=True)
            outputs.append(array(prediction))
            if index == 0:
                trace_data['initial'] = array(initial)
                for name in ['previous', 'prior', 'data_gradient', 'updated']:
                    trace_data[name] = np.stack([array(row[name]) for row in trace])
                for name in ['epsilon', 'rho', 'threshold']:
                    trace_data[name] = np.array([float(row[name]) for row in trace])
    return np.stack(outputs), trace_data


def smoke_step(model, measurement, target, c, halo, size):
    """仅原训练集第一块，一次新Adam检查；不续训或选择模型。"""
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'], foreach=False)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(tensor(measurement).to(device))
    truth = tensor(target).to(device)
    core = lambda value: value[..., halo:halo+size, halo:halo+size]
    loss = (core(prediction)-core(truth)).abs().mean()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError('GPU/CPU训练损失或梯度非法')
    gradients = {name: p.grad.detach().cpu().numpy().copy() for name, p in model.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip_norm'], foreach=False)
    optimizer.step()
    parameters = {name: p.detach().cpu().numpy().copy() for name, p in model.named_parameters()}
    if any(not np.isfinite(value).all() for value in parameters.values()):
        raise RuntimeError('一步更新后参数非有限')
    return dict(loss=float(loss.detach()), unclipped_gradient_norm=float(norm)), gradients, parameters


def dictionary_relative(a, b):
    if a.keys() != b.keys():
        raise ValueError('参数键不一致')
    numerator = denominator = 0.0
    for key in a:
        av = a[key].astype(np.float64)
        bv = b[key].astype(np.float64)
        numerator += float(np.sum((av-bv)**2))
        denominator += float(np.sum(bv**2))
    return float(np.sqrt(numerator / max(denominator, 1e-60)))


def run(c, no_save=False, show_plots=False):
    start = time.perf_counter()
    if type(c['cpu_threads']) is not int or not 1 <= c['cpu_threads'] <= 16:
        raise ValueError('CPU线程数非法')
    if type(c['seed']) is not int or c['seed'] < 0:
        raise ValueError('种子非法')
    if any(isinstance(v, bool) or not np.isfinite(v) or not 0 < v < 1 for v in c['thresholds'].values()):
        raise ValueError('阈值须介于0与1')
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in c['smoke'].values()):
        raise ValueError('训练检查参数须为正')
    if any(type(c['runtime'][key]) is not bool for key in ['save_results', 'show_plots']):
        raise ValueError('运行开关须为布尔值')
    if type(c['plots']['dpi']) is not int or c['plots']['dpi'] < 50:
        raise ValueError('图像dpi非法')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用：请在PyCharm选用work/environments/jeon_gpu_cu121/Scripts/python.exe；不会自动以CPU冒充GPU')
    torch.set_num_threads(c['cpu_threads'])
    torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = resolve_project_path(c['source_run'], ROOT)
    before, codes = checked_source(source, resolve_project_path(c['source_audit'], ROOT))
    # 沿已保存来源链取G2，保留原CCW核和合成响应；不混入CW诊断核。
    g4 = Path(json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['g4']['path'])
    g3 = Path(json.loads((g4/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    g2 = Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    evidence = {label: dict(path=str(path), sha256=tree_sha(path)) for label, path in [('expanded', source), ('g4', g4), ('g3', g3), ('g2', g2)]}
    with np.load(g2/'arrays/g2_measurement.npz', allow_pickle=False) as z:
        op = RGBForward(z['kernels'], z['response'])
        fingerprint = str(z['fingerprint'])
        wavelengths = z['wavelengths_m'].copy()
    payload = torch.load(source/'arrays/best_checkpoint.pt', map_location='cpu', weights_only=True)
    with np.load(source/'arrays/expanded_subset.npz', allow_pickle=False) as z:
        measurements = z['validation_measurements'].copy()
        old_predictions = z['best_validation_predictions'].copy()
        train_measurement = z['train_measurements'][0].copy()
        train_target = z['train_targets'][0].copy()
    runtime = effective_runtime(c, no_save, show_plots)
    dest = allocate_unique_run_dir('g3_gpu') if runtime['save_results'] else None
    plt, backend = configure_plotting(runtime['show_plots'])
    fig = None
    try:
        optical, arrays = optical_check(op, 'cuda', c['seed'])
        arrays.update(kernels=op.kernels, response=op.response, wavelengths_m=wavelengths)
        print('GPU双精度光学算子、伴随及梯度完成；核对冻结模型的8个验证块', flush=True)
        cpu_model = make_model(payload, op, 'cpu')
        gpu_model = make_model(payload, op, 'cuda')
        torch.cuda.reset_peak_memory_stats()
        cpu_prediction, cpu_trace = predict_trace(cpu_model, measurements)
        gpu_prediction, gpu_trace = predict_trace(gpu_model, measurements)
        prediction_checks = dict(new_cpu_vs_saved_cpu=relative(cpu_prediction, old_predictions),
                                 gpu_vs_new_cpu=relative(gpu_prediction, cpu_prediction),
                                 gpu_vs_saved_cpu=relative(gpu_prediction, old_predictions))
        stage_checks = {key: relative(gpu_trace[key], cpu_trace[key]) for key in cpu_trace}
        print('冻结模型各阶段核对完成；执行原训练第一块的一次CPU/GPU Adam检查', flush=True)
        halo = payload['dataset']['halo']
        size = payload['dataset']['core_size']
        cpu_smoke, cpu_grad, cpu_params = smoke_step(cpu_model, train_measurement, train_target, c['smoke'], halo, size)
        gpu_smoke, gpu_grad, gpu_params = smoke_step(gpu_model, train_measurement, train_target, c['smoke'], halo, size)
        smoke = dict(cpu=cpu_smoke, gpu=gpu_smoke,
                     loss_relative=abs(gpu_smoke['loss']-cpu_smoke['loss'])/max(abs(cpu_smoke['loss']), 1e-30),
                     gradient_relative=dictionary_relative(gpu_grad, cpu_grad),
                     updated_parameter_relative=dictionary_relative(gpu_params, cpu_params))
        del cpu_grad, gpu_grad, cpu_params, gpu_params
        # 每次烟雾检查都在内存副本上，旧checkpoint完全不写入。
        arrays.update(validation_measurements=measurements, saved_cpu_predictions=old_predictions,
                      new_cpu_predictions=cpu_prediction, gpu_predictions=gpu_prediction)
        arrays.update({f'cpu_trace_{key}': value for key, value in cpu_trace.items()})
        arrays.update({f'gpu_trace_{key}': value for key, value in gpu_trace.items()})
        threshold = c['thresholds']
        passed = (max(optical.values()) < threshold['optical_float64']
                  and max(prediction_checks.values()) < threshold['prediction_float32']
                  and max(stage_checks.values()) < threshold['prediction_float32']
                  and smoke['loss_relative'] < threshold['prediction_float32']
                  and smoke['gradient_relative'] < threshold['gradient_float32']
                  and smoke['updated_parameter_relative'] < threshold['parameter_float32'])
        changed = float(torch.linalg.vector_norm(next(gpu_model.parameters()).detach().cpu()-payload['state_dict'][next(iter(dict(gpu_model.named_parameters())))]))
        passed = passed and changed > 0
        summary = dict(gpu_validation_passed=bool(passed), optical=optical, prediction_checks=prediction_checks,
                       stage_checks=stage_checks, smoke=smoke, optimizer_steps_per_device=1,
                       first_parameter_change_norm=changed, validation_patches=8, source_best_epoch=payload['best_epoch'],
                       height_fingerprint=fingerprint, response_status='synthetic_not_calibrated',
                       model_dtype='float32', optical_dtype='float64', amp_enabled=False, tf32_enabled=False,
                       deterministic_algorithms=True, training_purpose='one-step execution validation; no model selected',
                       test_data_loaded=False, paper_alignment_passed=False)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        k = 12
        cpu = cpu_prediction[0, halo:halo+size, halo:halo+size, k]
        gpu = gpu_prediction[0, halo:halo+size, halo:halo+size, k]
        for ax, value, title in zip(axes, [cpu, gpu, gpu-cpu], ['新环境CPU', 'GPU', 'GPU−CPU']):
            im = ax.imshow(value, origin='lower', cmap='inferno',
                           vmin=min(float(cpu.min()), float(gpu.min())) if title != 'GPU−CPU' else None,
                           vmax=max(float(cpu.max()), float(gpu.max())) if title != 'GPU−CPU' else None)
            ax.set_title(title); fig.colorbar(im, ax=ax)
        fig.suptitle('冻结模型540nm中心区域；比较执行一致性，不评价训练改进')
        fig.tight_layout()
        if dest:
            write_json(dest/'config_effective.json', dict(c, runtime=runtime))
            np.savez_compressed(dest/'arrays/gpu_checks.npz', **arrays)
            with np.load(dest/'arrays/gpu_checks.npz', allow_pickle=False) as z:
                if any(not np.array_equal(z[key], value) for key, value in arrays.items()):
                    raise RuntimeError('保存重载不一致')
            fig.savefig(dest/'figures/cpu_gpu_comparison.png', dpi=c['plots']['dpi'])
            write_json(dest/'source_evidence.json', evidence)
            names = sorted(set([*codes, 'main_g3_gpu.py', 'config_g3_gpu.json', 'G3_GPU_README.md', 'requirements_g3_gpu.txt']))
            write_json(dest/'source_manifest.json', {name: file_sha(ROOT/name) for name in names})
        if tree_sha(source) != before or any(tree_sha(Path(item['path'])) != item['sha256'] for item in evidence.values()):
            raise RuntimeError('旧模型或来源改变')
        device_info = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report = dict(summary=summary, environment=environment_info(), torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                      torch_file=torch.__file__, device=device_info, elapsed_seconds=time.perf_counter()-start, backend=backend)
        if dest: write_json(dest/'metrics/validation.json', report)
        if runtime['show_plots']: plt.show()
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        print('GPU验证结果：'+str(dest) if dest else 'GPU无保存：未写结果文件')
        if not passed: raise RuntimeError('GPU一致性门槛未通过；保留结果，不自动更改阈值')
        return report
    except Exception as exc:
        if dest: write_json(dest/'failed.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if fig is not None: plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config_g3_gpu.json')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    args = parser.parse_args()
    c = json.loads(resolve_project_path(args.config, ROOT).read_text(encoding='utf-8'))
    return 0 if run(c, args.no_save, args.show_plots)['summary']['gpu_validation_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
