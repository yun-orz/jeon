# -*- coding: utf-8 -*-
"""G5有限域诊断：固定模型，同一物理中心，共享RGB测量，不训练。"""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from scipy.io import loadmat
from main_g1 import write_json
from main_g5_retest import checked_source
from main_g5_test import predict, aggregate
from main_stage02b import allocate_unique_run_dir
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB, HQSDecoder
from optics.g4_data import file_sha
from optics.g5_boundary import center_crop, support_tiles, change_metrics
from optics.g5_metrics import assess
from optics.stage02c_source import tree_sha
from optics.stage02_runtime import resolve_project_path, effective_runtime, configure_plotting
from optics.runutil import environment_info

ROOT = Path(__file__).resolve().parent


def run(c, no_save=False, show_plots=False):
    start = time.perf_counter()
    sizes = c['context_sizes']; support = c['support_size']
    if not isinstance(sizes, list) or len(sizes) < 3 or any(type(s) is not int or s < 128 for s in sizes) or sizes != sorted(set(sizes)) or sizes[0] != 128:
        raise ValueError('至少三个严格递增上下文，首项必须128')
    if type(support) is not int or support > 512 or support < sizes[-1]+96 or any((support-s) % 2 for s in sizes):
        raise ValueError('支持域需覆盖最大上下文及97像素核半径；上限512')
    if type(c['cpu_threads']) is not int or not 1 <= c['cpu_threads'] <= 16:
        raise ValueError('CPU线程数非法')
    threshold = c['stability_relative_l2_threshold']
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('诊断阈值须介于0与1，不能作为论文标准')
    if any(type(c['runtime'][k]) is not bool for k in ['save_results', 'show_plots']) or type(c['plots']['dpi']) is not int or c['plots']['dpi'] < 50:
        raise ValueError('运行或绘图参数非法')
    torch.set_num_threads(c['cpu_threads'])
    source = resolve_project_path(c['source_retest'], ROOT)
    before, old_codes = checked_source(source, resolve_project_path(c['retest_audit'], ROOT))
    evidence = json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    original = Path(evidence['original_test']['path']); expanded = Path(evidence['expanded_training']['path'])
    for item in evidence.values():
        if tree_sha(Path(item['path'])) != item['sha256']:
            raise ValueError('冻结来源run改变')
    meta = json.loads((source/'dataset_manifest.json').read_text(encoding='utf-8'))
    protocol = json.loads((source/'metrics/protocol.json').read_text(encoding='utf-8'))
    payload = torch.load(expanded/'arrays/best_checkpoint.pt', map_location='cpu', weights_only=True)
    if payload['dataset']['normalization_scale'] != meta['normalization_scale']:
        raise ValueError('训练尺度改变')
    folder = resolve_project_path(c['dataset_directory'], ROOT)
    dataset_before = tree_sha(folder)
    if file_sha(folder/meta['scene']) != meta['sha256'] or file_sha(folder/'calib.txt') != meta['sensitivity_sha256']:
        raise ValueError('原始场景或校正改变')
    mat = loadmat(folder/meta['scene'], variable_names=['ref', 'lbl'])
    calibration = np.loadtxt(folder/'calib.txt').reshape(-1)
    tiles, selection = support_tiles(mat['ref'], mat['lbl'], meta['origins_yx'], meta['context_size'], support, calibration, meta['normalization_scale'])
    del mat
    indices = [r['original_index'] for r in selection if r['accepted']]
    with np.load(source/'arrays/fixed_scene_retest.npz', allow_pickle=False) as z:
        legacy_targets = z['targets_context'][indices].copy()
        legacy_rgb = z['measurements'][indices].copy()
        legacy_predictions = center_crop(z['expanded_predictions_context'][indices], meta['core_size']).copy()
    if not np.array_equal(center_crop(tiles, 128), legacy_targets):
        raise ValueError('同一目标中心/换算未精确复现')
    g4 = Path(json.loads((expanded/'source_evidence.json').read_text(encoding='utf-8'))['g4']['path'])
    g3 = Path(json.loads((g4/'source_evidence.json').read_text(encoding='utf-8'))['g3']['path'])
    g2 = Path(json.loads((g3/'source_evidence.json').read_text(encoding='utf-8'))['path'])
    with np.load(g2/'arrays/g2_measurement.npz', allow_pickle=False) as z:
        op = RGBForward(z['kernels'], z['response']); waves = z['wavelengths_m'].copy()
    if op.kernels.shape[-2:] != (97, 97):
        raise ValueError('本诊断限定已审核97像素核')
    model = HQSDecoder(TorchRGB(op.kernels, op.response), payload['network_config']).float()
    model.load_state_dict(payload['state_dict'])
    if not torch.equal(model.operator.kernels, torch.tensor(op.kernels, dtype=torch.float32)) or not torch.equal(model.operator.response, torch.tensor(op.response, dtype=torch.float32)):
        raise ValueError('模型光学缓存与冻结物理算子不同')
    multiple = 2**(payload['network_config']['levels']-1)
    if any(s % multiple or ((s-128)//2) % multiple for s in sizes):
        raise ValueError('上下文及裁剪偏移必须保持U-net池化网格相位')
    print(f'来源核对完成；接受原块{indices}，固定epoch {payload["best_epoch"]}，共享{support}像素支持域测量', flush=True)
    master_rgb = np.stack([op.forward(t) for t in tiles]).astype(np.float32)
    target = center_crop(tiles, meta['core_size']).copy()
    cores = {}; records = {}; summaries = {}; inputs = {}; timings = {}
    for size in sizes:
        print(f'CPU冻结推理：上下文{size}×{size}，{len(tiles)}块', flush=True)
        now = time.perf_counter(); inputs[str(size)] = center_crop(master_rgb, size).copy()
        predictions = predict(model, inputs[str(size)])
        cores[str(size)] = center_crop(predictions, meta['core_size']).copy()
        records[str(size)] = [assess(a, b, protocol['data_range'], protocol['zero_norm_threshold'])[0] for a, b in zip(target, cores[str(size)])]
        summaries[str(size)] = aggregate(records[str(size)], protocol['data_range'])
        timings[str(size)] = time.perf_counter()-now
    largest = str(sizes[-1])
    comparisons = {str(s): change_metrics(cores[str(s)], cores[largest]) for s in sizes}
    adjacent = {f'{a}_to_{b}': change_metrics(cores[str(a)], cores[str(b)]) for a, b in zip(sizes[:-1], sizes[1:])}
    legacy_comparison = change_metrics(cores['128'], legacy_predictions)
    old_measurement_difference = change_metrics(inputs['128'], legacy_rgb)
    central_measurement_difference = change_metrics(center_crop(inputs['128'], meta['core_size']), center_crop(legacy_rgb, meta['core_size']))
    # 最后两种尺寸接近，只能称有限范围稳定，不能证明任意上下文或整图收敛。
    last_change = adjacent[f'{sizes[-2]}_to_{sizes[-1]}']['relative_l2']
    report = dict(status='completed',boundary_diagnostic_completed=True,optimizer_steps=0,checkpoint_frozen=True,
                  fresh_blind_test=False,paper_alignment_passed=False,whole_image_convergence_proved=False,
                  context_sizes=sizes,support_size=support,core_size=meta['core_size'],selection=selection,
                  accepted_original_indices=indices,original_roi_count=len(selection),evaluated_roi_count=len(indices),
                  shared_measurement=True,forward_support_margin=(support-sizes[-1])//2,kernel_radius=48,
                  pooling_multiple=multiple,pooling_phase_aligned=True,summary=summaries,change_vs_largest=comparisons,
                  adjacent_change=adjacent,legacy_128_prediction_change=legacy_comparison,
                  legacy_128_measurement_change=old_measurement_difference,legacy_center_measurement_change=central_measurement_difference,
                  diagnostic_threshold=threshold,last_pair_below_threshold=last_change is not None and last_change < threshold,
                  threshold_status='engineering_diagnostic_only_not_paper_criterion',context_inference_seconds=timings,
                  response_status='synthetic_not_calibrated',environment=environment_info(),torch_version=torch.__version__,
                  elapsed_seconds=time.perf_counter()-start)
    runtime = effective_runtime(c, no_save, show_plots)
    plt, backend = configure_plotting(runtime['show_plots']); figures = []
    dest = allocate_unique_run_dir('g5_boundary') if runtime['save_results'] else None
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(sizes, [comparisons[str(s)]['relative_l2'] for s in sizes], 'o-')
        axes[0].axhline(threshold, color='gray', linestyle='--', label='诊断阈值，非论文标准')
        axes[0].set_ylabel('中心预测相对L2差（参考最大上下文）'); axes[0].legend()
        axes[1].plot(sizes, [summaries[str(s)]['psnr_pooled_db'] for s in sizes], 'o-'); axes[1].set_ylabel('固定中心PSNR / dB')
        for ax in axes: ax.set_xlabel('上下文边长 / 像素'); ax.grid(alpha=.3)
        fig.suptitle('冻结模型，共享RGB测量；只评价支持域有效的原区域'); fig.tight_layout(); figures.append(('boundary_summary.png', fig))
        fig, axes = plt.subplots(1, len(sizes)+1, figsize=(15, 3))
        values = [target[0, :, :, 12]]+[cores[str(s)][0, :, :, 12] for s in sizes]
        lo = min(0., min(float(v.min()) for v in values)); hi = max(float(v.max()) for v in values)
        for ax, value, title in zip(axes, values, ['目标540nm']+[f'{s}上下文' for s in sizes]):
            im = ax.imshow(value, origin='lower', cmap='inferno', vmin=lo, vmax=hi); ax.set_title(title)
        fig.colorbar(im, ax=list(axes), shrink=.7); figures.append(('boundary_centers.png', fig))
        if dest:
            write_json(dest/'config_effective.json', dict(c, runtime=runtime))
            write_json(dest/'dataset_manifest.json', dict(original_metadata=meta,selection=selection,normalization_refit=False))
            write_json(dest/'metrics/scores.json', records); write_json(dest/'metrics/protocol.json', dict(protocol,boundary_diagnostic=True))
            arrays = dict(support_targets=tiles,shared_support_rgb=master_rgb,target_core=target,legacy_128_rgb=legacy_rgb,
                          legacy_128_core_predictions=legacy_predictions,wavelengths_m=waves,kernels=op.kernels,response=op.response)
            arrays.update({f'input_{s}': v for s,v in inputs.items()}); arrays.update({f'prediction_core_{s}': v for s,v in cores.items()})
            np.savez_compressed(dest/'arrays/boundary.npz', **arrays)
            names = sorted(set([*old_codes,'main_g5_boundary.py','config_g5_boundary.json','optics/g5_boundary.py','tests/test_g5_boundary.py','G5_BOUNDARY_README.md']))
            write_json(dest/'source_manifest.json', {n:file_sha(ROOT/n) for n in names})
            write_json(dest/'source_evidence.json', dict(retest=dict(path=str(source),sha256=before),
                       dataset=dict(path=str(folder),sha256=dataset_before),expanded_training=evidence['expanded_training']))
            for name, fig in figures: fig.savefig(dest/'figures'/name, dpi=c['plots']['dpi'])
            with np.load(dest/'arrays/boundary.npz', allow_pickle=False) as z:
                if any(not np.array_equal(z[f'prediction_core_{s}'], cores[str(s)]) for s in sizes):
                    raise RuntimeError('预测保存重载不一致')
        if tree_sha(source) != before or tree_sha(folder) != dataset_before or any(tree_sha(Path(i['path'])) != i['sha256'] for i in evidence.values()):
            raise RuntimeError('冻结来源被修改')
        if dest: write_json(dest/'metrics/validation.json', dict(report,runtime=runtime,backend=backend))
        if runtime['show_plots']: plt.show()
        print(json.dumps(dict(summary=summaries,change_vs_largest=comparisons,adjacent_change=adjacent,
                             legacy_128_prediction_change=legacy_comparison,last_pair_below_threshold=report['last_pair_below_threshold']),ensure_ascii=False,allow_nan=False,indent=2))
        print('边界诊断结果：'+str(dest) if dest else '边界诊断无保存：未写结果文件')
        return report
    except Exception as exc:
        if dest: write_json(dest/'failed.json', dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        for _,fig in figures: plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='config_g5_boundary.json')
    p.add_argument('--no-save',action='store_true'); p.add_argument('--show-plots',action='store_true'); a=p.parse_args()
    c=json.loads(resolve_project_path(a.config,ROOT).read_text(encoding='utf-8'))
    return 0 if run(c,a.no_save,a.show_plots)['boundary_diagnostic_completed'] else 2


if __name__ == '__main__': raise SystemExit(main())
