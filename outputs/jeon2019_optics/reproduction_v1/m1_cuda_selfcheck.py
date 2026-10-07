# -*- coding: utf-8 -*-
"""新GPU解释器CUDA自检：真实矩阵运算、架构列表、显存与环境指纹。

只读取环境，不安装、不下载、不改动任何文件；结果以UTF-8 JSON保存在新M1结果目录。
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
M1 = HERE.parents[2] / "outputs/jeon2019_optics/results/reproduction_v1/m1"


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    info = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": "m1_cuda_selfcheck",
            "executable": sys.executable, "python": sys.version.split()[0],
            "platform": platform.platform(), "cwd": os.getcwd(),
            "torch_version": torch.__version__, "torch_file": torch.__file__,
            "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(), "arch_list": torch.cuda.get_arch_list(),
            "tf32_matmul_default_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn_default_allowed": bool(torch.backends.cudnn.allow_tf32)}
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["total_memory_mib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2)
        torch.manual_seed(20261006)
        torch.cuda.manual_seed_all(20261006)
        a = torch.randn(512, 512, dtype=torch.float64, device="cuda")
        b = torch.randn(512, 512, dtype=torch.float64, device="cuda")
        product = a @ b
        reference = a.cpu().double() @ b.cpu().double()
        info["fp64_matmul_relative_error"] = float((product.cpu() - reference).norm() / reference.norm())
        single = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
        info["fp32_finite"] = bool(torch.isfinite(single.sum()) and torch.isfinite(single @ single).all())
        # 关闭TF32后复算同一双精度矩阵乘，确认未被降精度替换。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        info["fp64_matmul_relative_error_tf32_off"] = float(
            ((a @ b).cpu() - reference).norm() / reference.norm())
        info["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        info["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    destination = M1 / "cuda_selfcheck.json"
    destination.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("已写入 " + str(destination))
    return 0 if info["cuda_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
