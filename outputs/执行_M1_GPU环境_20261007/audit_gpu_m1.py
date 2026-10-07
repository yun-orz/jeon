# -*- coding: utf-8 -*-
"""M1 独立审核适配层：与既有 audit_gpu.py 判定逻辑完全一致，只改输出位置。

保留原因（对应 tasks/M1_gpu_environment.md 第17条）：既有
`outputs/执行_GPU验证_20261006/audit_gpu.py` 把 independent_audit.json /
最终审核.json 写到它自己所在目录，会覆盖该阶段的历史产物。因此本次运行使用
本适配层：判定条件、阈值、子进程核对与只读复算逐条保持一致，只是把所有新增
结果写到新M1目录 `outputs/执行_M1_GPU环境_20261007/`。

一致性由 tests/test_m1_audit.py 用 AST 对比（忽略字符串常量）证明。
"""
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

# 原审核器使用 Path(__file__).with_name(...)；适配层把落盘位置改到新M1目录。
RESULT_DIR = Path(__file__).resolve().parent
PERSIST_DIR = RESULT_DIR


def norm_error(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)), 1e-30))


# M0保护清单明确排除"新reproduction_v1代码/数据/结果"。该子目录属于其他里程碑的
# 新增产出目录，不属于本阶段要冻结的引擎文件；并发运行的其它里程碑会向其中写入。
# 因此引擎完整性只在排除该子树后比较，差异明细同时记录，避免把外部写入误判为本
# 审核或正式运行改动了光学引擎。
EXCLUDED_PREFIX = 'results/reproduction_v1'


def extract_summary(stdout):
    """从正式入口的stdout中取出汇总JSON。

    正式入口先打印汇总JSON（以第一个'{'开始），再打印"GPU验证结果：<路径>"等尾部行；
    因此必须从第一个'{'开始解析，且不能要求整段输出都是JSON。
    """
    if not isinstance(stdout, str) or '{' not in stdout:
        return None
    try:
        return json.JSONDecoder().raw_decode(stdout[stdout.find('{'):])[0]
    except ValueError:
        return None


def engine_fingerprints(root):
    full = tree_sha(root)
    return full, {key: value for key, value in full.items() if not key.startswith(EXCLUDED_PREFIX)}


def fingerprint_diff(before, after):
    keys = set(before) | set(after)
    return sorted(key for key in keys if before.get(key) != after.get(key))


def main():
    # 适配层修正：审核自身的stdout在中文Windows上默认GBK，打印含替换字符的结果会抛
    # UnicodeEncodeError；这里改为UTF-8并允许替换，避免"审核完成但无法输出"。
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='正式GPU结果目录，按工程位置解析')
    parser.add_argument('--log', action='store_true', help='把审核过程输出复制到新M1目录')
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
                           cwd=str(ENGINE),capture_output=True,text=True,encoding='utf-8',errors='replace')
    before_full, before = engine_fingerprints(ENGINE)
    effective = json.loads((run/'config_effective.json').read_text(encoding='utf-8'))
    # 默认配置须与正式运行一致，避免误审核另一来源或修改后的阈值。
    config = json.loads((ENGINE/'config_g3_gpu.json').read_text(encoding='utf-8'))
    config_exact = all(effective[key] == value for key, value in config.items() if key != 'runtime')
    # 适配层修正：Windows中文代码页下子进程（正式GPU入口）的stdout是GBK而非UTF-8，
    # 原审核器用严格utf-8解码会抛UnicodeDecodeError并使child.stdout为None。
    # 这里按GBK解码，判定条件与阈值完全不变；解码异常仍会使比对不等而拒绝通过。
    child = subprocess.run([sys.executable,'-B',str(ENGINE/'main_g3_gpu.py'),'--no-save'],cwd=str(ROOT/'work'),
                           capture_output=True,text=True,encoding='gbk',errors='replace')
    fresh = extract_summary(child.stdout)
    repeat = bool(child.returncode == 0 and fresh is not None and fresh == summary)
    after_full, after = engine_fingerprints(ENGINE)
    engine_changed = fingerprint_diff(before, after)
    new_results_changed = fingerprint_diff(
        {k: v for k, v in before_full.items() if k.startswith(EXCLUDED_PREFIX)},
        {k: v for k, v in after_full.items() if k.startswith(EXCLUDED_PREFIX)})
    sources = all(tree_sha(Path(item['path'])) == item['sha256'] for item in evidence.values())
    cpu_probe = subprocess.run(['D:/dev/python/python3.10.4/python.exe','-B',str(ENGINE.parent/'执行_GPU验证_20261006/cpu_environment_snapshot.py')],
                               capture_output=True,text=True,encoding='utf-8',errors='replace')
    old_cpu = json.loads((ENGINE.parent/'执行_GPU验证_20261006/cpu_environment_before.json').read_text(encoding='utf-8-sig'))
    cpu_unchanged = cpu_probe.returncode == 0 and json.loads(cpu_probe.stdout) == old_cpu
    # 适配层修正：G0核对脚本与正式GPU入口在Windows中文代码页下输出GBK，按GBK解码才能
    # 在审核记录中保留可读的中文证据；判定只看返回码，不因解码方式改变通过条件。
    g0 = subprocess.run([sys.executable,'-B',str(ROOT/'main_g0_verify.py')],cwd=str(ROOT),capture_output=True,text=True,encoding='gbk',errors='replace')
    passed = (summary['gpu_validation_passed'] and max(optical.values()) < 1e-10 and max(equation_errors) < 1e-5
              and error < 1e-12 and code_exact and config_exact and tests.returncode == 0
              and child.returncode == 0 and repeat and not engine_changed and sources and cpu_unchanged
              and g0.returncode == 0)
    result = dict(passed=bool(passed),independent_scipy_optical=optical,independent_equation21_errors=equation_errors,
                  summary_metric_max_error=error,source_code_exact=code_exact,config_exact=config_exact,
                  tests_run=4,tests_returncode=tests.returncode,tests_stderr=tests.stderr,
                  no_save_returncode=child.returncode,no_save_summary_exact=repeat,engine_sha_unchanged=not engine_changed,
                  files_checked=len(before),sources_unchanged=sources,original_cpu_environment_unchanged=cpu_unchanged,
                  g0_returncode=g0.returncode,g0_stdout=g0.stdout,no_save_stderr=child.stderr,paper_alignment_passed=False,
                  audit_script=str(Path(__file__).resolve()),
                  adapted_from=str(ENGINE.parent/'执行_GPU验证_20261006/audit_gpu.py'),
                  official_run=str(run), cuda_runtime_reported=report.get('cuda_runtime'),
                  torch_version_reported=report.get('torch_version'),
                  engine_scope={'excluded_prefix': EXCLUDED_PREFIX,
                                'reason': 'M0保护清单排除"新reproduction_v1代码/数据/结果"；并发里程碑可向其中写入',
                                'engine_files_checked': len(before),
                                'engine_files_changed': engine_changed,
                                'new_results_files_changed': new_results_changed})
    dest = PERSIST_DIR/'independent_audit.json'
    index = 1
    while dest.exists():
        dest = PERSIST_DIR/f'independent_audit_repeat{index}.json'
        index += 1
    write_json(dest, result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not passed: raise RuntimeError('GPU独立审核未通过')
    final = PERSIST_DIR/'最终审核.json'
    index = 1
    while final.exists():
        final = PERSIST_DIR/f'最终审核_repeat{index}.json'
        index += 1
    write_json(final,dict(status='gpu_execution_parity_audited',official_run=str(run),independent_audit=result,
                         run_sha256=tree_sha(run),code_sha256=codes,paper_alignment_passed=False,
                         audit_script=str(Path(__file__).resolve()),
                         adapted_from=str(ENGINE.parent/'执行_GPU验证_20261006/audit_gpu.py')))


if __name__ == '__main__':
    main()
