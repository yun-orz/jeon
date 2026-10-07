# -*- coding: utf-8 -*-
"""M1 check：驱动/GPU、解释器实际导入位置、环境继承与峰值空间核算。

设计原则：

- 只读检查：不安装、不下载、不删除、不改动原CPU环境。
- 空间按"剩余下载 + 整包拼接副本 + 解压安装 + 审核输出"计算峰值，数据留在D盘；
  不足即返回停止原因，不以其他盘或降低规模冒称通过。
- Torch 实际导入位置（而非环境目录推断）是判定GPU是否真正生效的唯一依据。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "outputs/jeon2019_optics/results/reproduction_v1"

TORCH_BYTES = 2449372784
TORCH_SHA256 = "9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f"
SYMPY_BYTES = 6189177
PREFIX = ROOT / "work/dependencies/gpu_cu121_20261006/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
ENVIRONMENT = ROOT / "work/environments/jeon_gpu_cu121"
BASE_PYTHON = Path("D:/dev/python/python3.10.4/python.exe")
RESUME_WORKDIR = ROOT / "work/dependencies/gpu_cu121_resume_m1_20261007"
AUDIT_RESERVE_BYTES = 512 * 1024 * 1024

# 安装后解压体积按官方轮子的保守倍数估算：CUDA版Torch解压后会明显大于轮子本身。
EXTRACT_RATIO = 1.6
EXTRACT_MIN_BYTES = 3 * 1024 ** 3


def space_budget(free_bytes, remaining_download, assembled_copy, install_estimate,
                 audit_reserve=AUDIT_RESERVE_BYTES, safety_ratio=0.05):
    """按任务要求计算峰值空间；不足则给出缺口与停止原因。"""
    components = {"remaining_download": float(remaining_download),
                  "assembled_copy": float(assembled_copy),
                  "install_estimate": float(install_estimate),
                  "audit_reserve": float(audit_reserve)}
    peak = sum(components.values())
    safety = peak * float(safety_ratio)
    required = peak + safety
    headroom = float(free_bytes) - required
    passed = headroom >= 0
    return {"passed": passed, "components": components, "peak_bytes": peak,
            "safety_margin_bytes": safety, "required_bytes": required,
            "free_bytes": float(free_bytes), "headroom_bytes": headroom,
            "shortfall_bytes": 0.0 if passed else -headroom,
            "stop_reason": None if passed else
            "D盘剩余空间不足：需要 %.3f GiB（含 %.0f%% 安全余量），可用 %.3f GiB，缺口 %.3f GiB。"
            "按任务要求停止，不切换到其他盘、不缩小数据规模。"
            % (required / 1024 ** 3, safety_ratio * 100, float(free_bytes) / 1024 ** 3,
               -headroom / 1024 ** 3)}


def disk_free(path):
    return shutil.disk_usage(str(path)).free


def run_command(command, timeout=120):
    """执行只读诊断命令；失败不抛异常，返回结构化结果。"""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout)
        return {"command": command, "returncode": completed.returncode,
                "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}


def gpu_hardware():
    """驱动与GPU：nvidia-smi 为主，失败时如实记录而不是推断。"""
    query = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"]
    result = run_command(query)
    devices = []
    if result["returncode"] == 0:
        for line in result["stdout"].splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) == 4:
                devices.append({"name": fields[0], "driver_version": fields[1],
                                "memory_total_mib": fields[2], "compute_capability": fields[3]})
    banner = run_command(["nvidia-smi"], timeout=60)
    cuda_line = None
    for line in banner["stdout"].splitlines():
        if "CUDA Version" in line:
            cuda_line = line.strip()
            break
    return {"query": result, "devices": devices, "driver_cuda_banner": cuda_line,
            "available": result["returncode"] == 0 and bool(devices)}


def interpreter_report(python, label):
    """解释器实际导入位置与环境继承；用子进程避免把本进程状态当成目标环境状态。"""
    script = (
        "import json,sys,configparser,importlib.metadata as md,os,pathlib\n"
        "here=pathlib.Path(sys.executable).resolve().parent\n"
        "cfg={}\n"
        "p=here.parent/'pyvenv.cfg'\n"
        "if p.is_file():\n"
        "    c=configparser.ConfigParser()\n"
        "    c.read_string('[root]\\n'+p.read_text(encoding='utf-8'))\n"
        "    cfg=dict(c['root'])\n"
        "versions={}\n"
        "for name in ['torch','sympy','numpy','scipy','matplotlib']:\n"
        "    try: versions[name]=md.version(name)\n"
        "    except Exception: versions[name]=None\n"
        "out=dict(executable=sys.executable,version=sys.version.split()[0],prefix=sys.prefix,\n"
        "         base_prefix=sys.base_prefix,is_virtualenv=sys.prefix!=sys.base_prefix,\n"
        "         pyvenv_cfg=cfg,versions=versions,sys_path=sys.path[:8],\n"
        "         torch_file=None,torch_version=None,cuda_runtime=None,cuda_available=False,cuda_error=None)\n"
        "try:\n"
        "    import torch\n"
        "    out['torch_file']=torch.__file__\n"
        "    out['torch_version']=torch.__version__\n"
        "    out['cuda_runtime']=torch.version.cuda\n"
        "    out['cuda_available']=bool(torch.cuda.is_available())\n"
        "    out['torch_cuda_arch_list']=torch.cuda.get_arch_list() if out['cuda_available'] else []\n"
        "    if out['cuda_available']:\n"
        "        out['device_name']=torch.cuda.get_device_name(0)\n"
        "        out['device_capability']=list(torch.cuda.get_device_capability(0))\n"
        "except Exception as exc:\n"
        "    out['cuda_error']=repr(exc)\n"
        "print(json.dumps(out,ensure_ascii=False))\n")
    result = run_command([str(python), "-B", "-c", script], timeout=300)
    report = {"label": label, "requested_python": str(python), "exists": Path(python).is_file(),
              "probe": result}
    if result["returncode"] == 0 and result["stdout"]:
        try:
            report.update(json.loads(result["stdout"].splitlines()[-1]))
        except ValueError as exc:
            report["parse_error"] = repr(exc)
    return report


def inherited_packages(base_prefix, environment):
    """列出新环境通过 include-system-site-packages 继承自原解释器的包目录。"""
    candidates = []
    for pattern in ["lib/site-packages", "Lib/site-packages"]:
        path = Path(base_prefix) / pattern
        if path.is_dir():
            candidates.append(path)
    inherited = {"base_site_packages": [str(item) for item in candidates],
                 "inherit_flag": None, "note": "原科学计算包由pyvenv.cfg继承，不在新环境重复安装"}
    config = Path(environment) / "pyvenv.cfg"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("include-system-site-packages"):
                inherited["inherit_flag"] = line.split("=", 1)[1].strip().lower() == "true"
    return inherited


def current_prefix_bytes():
    if not PREFIX.is_file():
        return {"exists": False, "path": str(PREFIX), "bytes": 0,
                "remaining_bytes": TORCH_BYTES,
                "note": "旧下载前缀不存在；重新获取需下载整包，不表示已安装GPU环境缺失"}
    size = PREFIX.stat().st_size
    return {"exists": True, "path": str(PREFIX), "bytes": size,
            "remaining_bytes": max(TORCH_BYTES - size, 0),
            "note": "已有部分下载不是恒定现状；只有整包官方SHA通过才能安装"}


def check(persist=True):
    """组装M1 check结果；只读，不安装。"""
    prefix = current_prefix_bytes()
    remaining = prefix["remaining_bytes"] + SYMPY_BYTES + 1024 * 1024
    install_estimate = max(int(TORCH_BYTES * EXTRACT_RATIO), EXTRACT_MIN_BYTES)
    budget = space_budget(disk_free(ROOT), remaining, TORCH_BYTES, install_estimate)
    cpu = interpreter_report(BASE_PYTHON, "原CPU解释器")
    gpu = interpreter_report(ENVIRONMENT / "Scripts/python.exe", "新GPU环境解释器")
    inherit = inherited_packages(cpu.get("base_prefix") or "D:/dev/python/python3.10.4", ENVIRONMENT)
    temp = Path(os.environ.get("TEMP", "C:/Windows/Temp"))
    gpu_torch_file = gpu.get("torch_file")
    resolved = bool(gpu_torch_file) and Path(str(gpu_torch_file)).resolve().is_relative_to(ENVIRONMENT.resolve())
    report = {
        "schema": "jeon2019-m1-check-v1",
        "scope": "M1 环境与执行一致性检查（只读；不安装、不下载、不删除）",
        "operator": {"executable": sys.executable, "version": sys.version.split()[0]},
        "gpu_hardware": gpu_hardware(),
        "interpreters": {"cpu": cpu, "gpu_environment": gpu},
        "environment_inheritance": inherit,
        "frozen_download": {"url": "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl",
                            "total_bytes": TORCH_BYTES, "sha256": TORCH_SHA256,
                            "sympy_bytes": SYMPY_BYTES, "block_bytes": 32 * 1024 * 1024},
        "prefix": prefix, "resume_workdir": str(RESUME_WORKDIR),
        "space": {"project_root": str(ROOT), "free_on_project_drive": budget["free_bytes"],
                  "temp_dir": str(temp), "free_on_temp_drive": disk_free(temp) if temp.exists() else None,
                  "budget": budget,
                  "note": "数据留在D盘；TEMP在C盘，pip不得使用缓存/构建隔离，只读取D盘已核验轮子"},
        "gpu_environment_ready": bool(gpu.get("cuda_available") and resolved),
        "torch_from_environment": resolved,
        "status": None,
    }
    if not budget["passed"]:
        report["status"] = "space_stopped"
    elif not gpu.get("exists"):
        report["status"] = "environment_missing"
    elif not gpu.get("cuda_available") and resolved and gpu.get("cuda_runtime"):
        # 已有CUDA版Torch但运行时看不到设备；重装轮子不能代替设备/驱动排查。
        report["status"] = "gpu_device_unavailable"
    elif not gpu.get("cuda_available"):
        report["status"] = "gpu_runtime_pending"
    else:
        report["status"] = "gpu_ready"
    report["next_action"] = {"space_stopped": "记录需求后停止；腾出空间再重跑 check",
                             "environment_missing": "先创建 work/environments/jeon_gpu_cu121 解释器",
                             "gpu_device_unavailable": "CUDA版Torch已安装；检查Windows显卡连接状态、笔记本GPU模式和NVIDIA驱动，再重跑CUDA自检；不要据此重新下载Torch",
                             "gpu_runtime_pending": "执行 m1_download.py all 续传并通过整包SHA后离线安装",
                             "gpu_ready": "运行 main_g3_gpu.py 正式验收，再跑独立审核"}[report["status"]]
    report["paper_alignment_passed"] = False
    if persist:
        RESULTS.mkdir(parents=True, exist_ok=True)
        destination = RESULTS / ("m1_check_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["written_to"] = str(destination)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true", help="只打印，不写结果文件")
    arguments = parser.parse_args()
    outcome = check(persist=not arguments.no_save)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    raise SystemExit(0 if outcome["status"] in {"gpu_ready", "gpu_runtime_pending"} else 2)
