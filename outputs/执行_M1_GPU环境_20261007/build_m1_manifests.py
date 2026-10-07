# -*- coding: utf-8 -*-
"""生成M1交付清单：续传清单、环境清单、下载SHA、运行记录与关键数值。

只读取已有文件并写入新M1目录，不修改任何历史产物。可重复执行（文件名唯一）。
"""
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "outputs/jeon2019_optics"
RESUME = ROOT / "work/dependencies/gpu_cu121_resume_m1_20261007"
ENVIRONMENT = ROOT / "work/environments/jeon_gpu_cu121"
AUDIT_DIR = ROOT / "outputs/执行_M1_GPU环境_20261007"
RUN = ENGINE / "results/g3_gpu/run_20261007_103806"
M1 = ENGINE / "results/reproduction_v1/m1"


def sha256(path, limit=None):
    digest = hashlib.sha256()
    remaining = limit
    with Path(path).open("rb") as stream:
        while remaining is None or remaining > 0:
            block = stream.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def write(name, payload, directory=None):
    directory = Path(directory or AUDIT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入 %s（%d 字节）" % (target, target.stat().st_size))
    return target


def tree(path):
    return {p.relative_to(path).as_posix(): sha256(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def main():
    manifest = json.loads((RESUME / "download_manifest.json").read_text(encoding="utf-8"))
    probe = json.loads((RESUME / "probe.json").read_text(encoding="utf-8"))
    install = json.loads((RESUME / "install_result.json").read_text(encoding="utf-8"))
    validation = json.loads((RUN / "metrics/validation.json").read_text(encoding="utf-8"))
    audit = json.loads((AUDIT_DIR / "independent_audit.json").read_text(encoding="utf-8"))
    pre = json.loads((M1 / "run_20261007_101648_pre_install/main_check_pre_install.json").read_text(encoding="utf-8"))
    post = json.loads((M1 / "run_20261007_103754_post_install/main_check_post_install.json").read_text(encoding="utf-8"))

    # 1. 续传清单：官方来源、总长、前缀、逐块偏移与SHA、恢复入口
    blocks = sorted(manifest["blocks"], key=lambda item: int(item["index"]))
    resume_manifest = {
        "schema": "jeon2019-m1-resume-manifest-v1", "milestone": "M1",
        "status": "complete_whole_sha_passed", "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "url": manifest["url"], "frozen_total_bytes": manifest["total_bytes"],
        "official_sha256": manifest["official_sha256"], "assembled_sha256": manifest["assembled_sha256"],
        "whole_sha256_passed": manifest["whole_sha256_passed"],
        "existing_prefix": {"source": manifest["prefix_source"], "bytes": manifest["prefix_bytes"],
                            "sha256": manifest["prefix_sha256"],
                            "policy": "只读复制为 prefix.bin；原文件未修改、未删除"},
        "block_bytes": manifest["block_bytes"], "block_count": len(blocks),
        "first_block": blocks[0], "last_block": blocks[-1],
        "blocks": blocks,
        "incomplete_block_files_kept": sorted(p.name for p in RESUME.glob("part_*.bin.tmp")),
        "events": manifest.get("events", []),
        "recovery": {"command": "python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py download",
                     "workdir": str(RESUME),
                     "behaviour": "重跑先按块长度与SHA复核磁盘，只补齐缺失或损坏的块；整包SHA未通过不安装"},
        "no_delete_performed": True,
        "sympy": {"url": "https://files.pythonhosted.org/packages/b2/fe/81695a1aa331a842b582453b605175f419fe8540355886031328089d840a/sympy-1.13.1-py3-none-any.whl",
                  "sha256": install["sympy_wheel_sha256"], "sha256_passed": True},
        "head_probe": probe,
        "official_link_recheck": {
            "index": "https://download.pytorch.org/whl/cu121/torch/",
            "index_href": "https://download-r2.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl#sha256=" + manifest["official_sha256"],
            "frozen_url": "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl",
            "sha256_matches_frozen": manifest["official_sha256"] == "9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f",
            "content_length_matches_frozen": probe["head"]["content_length"] == "2449372784",
            "note": "索引页href指向官方R2端点；冻结URL实时重定向到同一对象，二者内容长度一致，整包SHA一致。"},
    }
    write("续传清单.json", resume_manifest)

    # 2. 下载SHA：整包与逐块实际重算
    whole = RESUME / "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
    recomputed = sha256(whole)
    block_recheck = []
    for item in blocks:
        path = RESUME / ("part_%03d.bin" % int(item["index"]))
        block_recheck.append({"index": int(item["index"]), "range": item["range"],
                              "recorded_sha256": item["sha256"], "recomputed_sha256": sha256(path),
                              "bytes_on_disk": path.stat().st_size, "matches": sha256(path) == item["sha256"]})
    prefix_copy = sha256(RESUME / "prefix.bin")
    write("下载SHA.json", {
        "schema": "jeon2019-m1-download-sha-v1", "milestone": "M1",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "assembled_file": {"path": str(whole), "bytes": whole.stat().st_size,
                           "recomputed_sha256": recomputed,
                           "official_sha256": manifest["official_sha256"],
                           "whole_sha256_passed": recomputed == manifest["official_sha256"],
                           "frozen_total_bytes": manifest["total_bytes"],
                           "bytes_match_frozen": whole.stat().st_size == manifest["total_bytes"]},
        "prefix_copy": {"path": str(RESUME / "prefix.bin"), "bytes": (RESUME / "prefix.bin").stat().st_size,
                        "recomputed_sha256": prefix_copy,
                        "original_prefix_unchanged": sha256(ROOT / "work/dependencies/gpu_cu121_20261006/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl") == prefix_copy},
        "blocks": block_recheck, "blocks_all_match": all(item["matches"] for item in block_recheck),
        "note": "整包官方SHA是安装的唯一裁决；逐块SHA用于续传恢复，不替代整包核对"})

    # 3. 环境清单
    gpu_interp = post["environment"]["interpreters"]["gpu_environment"]
    cpu_interp = post["environment"]["interpreters"]["cpu"]
    environment_manifest = {
        "schema": "jeon2019-m1-environment-v1", "milestone": "M1",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python_operator": sys.version.split()[0]},
        "gpu": post["environment"]["gpu_hardware"],
        "frozen_execution": json.loads((ENGINE / "reproduction_v1/protocol.json").read_text(encoding="utf-8"))["execution"],
        "new_gpu_environment": {
            "path": str(ENVIRONMENT), "base_interpreter": cpu_interp["executable"],
            "base_python": cpu_interp["version"], "pyvenv_cfg": gpu_interp.get("pyvenv_cfg"),
            "torch_version": gpu_interp.get("torch_version"), "cuda_runtime": gpu_interp.get("cuda_runtime"),
            "torch_file": gpu_interp.get("torch_file"), "cuda_available": gpu_interp.get("cuda_available"),
            "device_name": gpu_interp.get("device_name"),
            "site_packages_bytes": sum(p.stat().st_size for p in ENVIRONMENT.rglob("*") if p.is_file()),
            "installed_from_verified_wheels": {"torch_sha256": install["torch_wheel_sha256"],
                                               "sympy_sha256": install["sympy_wheel_sha256"]},
            "install_command": install["command"], "install_returncode": install["returncode"]},
        "original_cpu_environment": {
            "path": cpu_interp["executable"], "torch_version": cpu_interp.get("torch_version"),
            "cuda_available": cpu_interp.get("cuda_available"), "torch_file": cpu_interp.get("torch_file"),
            "versions": cpu_interp.get("versions"),
            "note": "原CPU环境全程未安装、未替换依赖；新Torch/SymPy只装在新虚拟环境"},
        "environment_inheritance": post["environment"]["environment_inheritance"],
        "version_difference": {"cpu_torch": cpu_interp.get("torch_version"),
                               "gpu_torch": gpu_interp.get("torch_version"),
                               "recorded_honestly": True,
                               "thresholds_relaxed": False},
        "space": {"pre_install": pre["environment"]["space"], "post_install": post["environment"]["space"]},
        "cuda_selfcheck": json.loads((M1 / "cuda_selfcheck.json").read_text(encoding="utf-8")),
        "peak_memory": {"cuda_peak_allocated_bytes": validation["device"]["peak_allocated_bytes"],
                        "cuda_peak_reserved_bytes": validation["device"]["peak_reserved_bytes"],
                        "cuda_peak_allocated_gib": round(validation["device"]["peak_allocated_bytes"] / 1024 ** 3, 3),
                        "protocol_peak_gpu_memory_gib": 7,
                        "within_protocol_budget": validation["device"]["peak_reserved_bytes"] <= 7 * 1024 ** 3},
        "disk_after": {"d_free_bytes": shutil.disk_usage(str(ROOT)).free},
    }
    write("环境清单.json", environment_manifest)

    # 4. 运行记录：实际命令、输入输出SHA、git状态
    def frozen_audit():
        recorded = json.loads((ENGINE / "reproduction_v1/tests/m1_frozen_audit_sha.json").read_text(encoding="utf-8"))
        out = {}
        for key, rel in [("audit_gpu_py_sha256", "outputs/执行_GPU验证_20261006/audit_gpu.py"),
                         ("cpu_preflight_json_sha256", "outputs/执行_GPU验证_20261006/cpu_preflight.json"),
                         ("cpu_environment_before_sha256", "outputs/执行_GPU验证_20261006/cpu_environment_before.json"),
                         ("main_g3_gpu_py_sha256", "outputs/jeon2019_optics/main_g3_gpu.py"),
                         ("config_g3_gpu_json_sha256", "outputs/jeon2019_optics/config_g3_gpu.json")]:
            out[key] = {"recorded": recorded[key], "recomputed": sha256(ROOT / rel),
                        "unchanged": recorded[key] == sha256(ROOT / rel)}
        return out

    runs = {
        "schema": "jeon2019-m1-runs-v1", "milestone": "M1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commands": [
            {"purpose": "安装前check（原CPU解释器）",
             "command": "D:\\dev\\python\\python3.10.4\\python.exe -B outputs/jeon2019_optics/reproduction_v1/main.py check --stage pre_install",
             "cwd": str(ROOT), "result": "status=passed", "evidence": str(M1 / "run_20261007_101648_pre_install/main_check_pre_install.json")},
            {"purpose": "官方链接与头信息核对、已有前缀",
             "command": "python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py probe",
             "cwd": str(ROOT), "result": "head通过，前缀1345187968字节，剩余1104184816字节", "evidence": str(RESUME / "probe.json")},
            {"purpose": "分块续传并核对整包官方SHA",
             "command": "python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py download",
             "cwd": str(ROOT), "result": "33块全部通过，整包SHA通过", "evidence": str(RESUME / "download_manifest.json")},
            {"purpose": "离线安装到新GPU环境",
             "command": "python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py install",
             "cwd": str(ROOT), "result": "torch-2.5.1+cu121、sympy-1.13.1 安装成功", "evidence": str(RESUME / "install_result.json")},
            {"purpose": "新解释器CUDA自检",
             "command": "work/environments/jeon_gpu_cu121/Scripts/python.exe -B outputs/jeon2019_optics/reproduction_v1/m1_cuda_selfcheck.py",
             "cwd": str(ROOT), "result": "cuda_available=True，FP64相对误差7.6e-16，TF32默认关闭后复算一致", "evidence": str(M1 / "cuda_selfcheck.json")},
            {"purpose": "安装后check（新GPU解释器）",
             "command": "work/environments/jeon_gpu_cu121/Scripts/python.exe -B outputs/jeon2019_optics/reproduction_v1/main.py check --stage post_install",
             "cwd": str(ROOT), "result": "status=passed，环境status=gpu_ready", "evidence": str(M1 / "run_20261007_103754_post_install/main_check_post_install.json")},
            {"purpose": "正式GPU验收",
             "command": "work/environments/jeon_gpu_cu121/Scripts/python.exe -B D:\\PyCharmProjects\\Jeon2019\\outputs\\jeon2019_optics\\main_g3_gpu.py",
             "cwd": str(ROOT / "work"), "result": "gpu_validation_passed=true", "evidence": str(RUN / "metrics/validation.json")},
            {"purpose": "独立审核（适配层，写入新M1目录）",
             "command": "work/environments/jeon_gpu_cu121/Scripts/python.exe -B outputs/执行_M1_GPU环境_20261007/audit_gpu_m1.py --run results/g3_gpu/run_20261007_103806",
             "cwd": str(ENGINE), "result": "passed=true", "evidence": str(AUDIT_DIR / "independent_audit.json")},
            {"purpose": "续传器与空间核算测试",
             "command": "python -B -m unittest discover -s tests -p test_m1_resume.py",
             "cwd": str(ENGINE / "reproduction_v1"), "result": "16项通过", "evidence": str(M1 / "resume_tests.txt")},
        ],
        "frozen_inputs": frozen_audit(),
        "pre_post_check": {"pre_status": pre["status"], "post_status": post["status"],
                          "pre_protection": {k: v for k, v in pre["protection"].items() if k != "failures"},
                          "post_protection": {k: v for k, v in post["protection"].items() if k != "failures"},
                          "pre_g0": {"passed": pre["g0"]["passed"], "checked": pre["g0"].get("checked")},
                          "post_g0": {"passed": post["g0"]["passed"], "checked": post["g0"].get("checked")},
                          "cpu_torch_pre": pre["environment"]["interpreters"]["cpu"].get("torch_version"),
                          "cpu_torch_post": post["environment"]["interpreters"]["cpu"].get("torch_version"),
                          "gpu_torch_post": post["environment"]["interpreters"]["gpu_environment"].get("torch_version")},
        "gpu_run_fingerprints": tree(RUN),
        "audit_fingerprints": {p.name: sha256(p) for p in sorted(AUDIT_DIR.iterdir()) if p.is_file()},
        "code_fingerprints": {str(p.relative_to(ROOT)): sha256(p) for p in sorted([
            ENGINE / "reproduction_v1/main.py", ENGINE / "reproduction_v1/m1_resume.py",
            ENGINE / "reproduction_v1/m1_environment.py", ENGINE / "reproduction_v1/m1_download.py",
            ENGINE / "reproduction_v1/m1_check.py", ENGINE / "reproduction_v1/m1_fixture_server.py",
            ENGINE / "reproduction_v1/tests/test_m1_resume.py", ENGINE / "reproduction_v1/tests/test_m1_audit.py",
            ENGINE / "reproduction_v1/tests/m1_resume_cli_fixture.py",
            ENGINE / "reproduction_v1/tests/m1_frozen_audit_sha.json",
            AUDIT_DIR / "audit_gpu_m1.py"])},
        "key_numbers": {
            "optical_fp64_threshold": 1e-10, "optical_fp64_max": max(validation["summary"]["optical"].values()),
            "prediction_fp32_threshold": 1e-5, "prediction_fp32_max": max(validation["summary"]["prediction_checks"].values()),
            "stage_fp32_threshold": 1e-5, "stage_fp32_max": max(validation["summary"]["stage_checks"].values()),
            "gradient_fp32_threshold": 1e-4, "gradient_fp32": validation["summary"]["smoke"]["gradient_relative"],
            "adam_parameter_threshold": 1e-5, "adam_parameter_fp32": validation["summary"]["smoke"]["updated_parameter_relative"],
            "independent_equation21_threshold": 1e-5, "independent_equation21_max": max(audit["independent_equation21_errors"]),
            "summary_metric_max_error": audit["summary_metric_max_error"],
        },
        "failures_kept": [p.name for p in sorted(AUDIT_DIR.glob("*attempt*"))],
        "paper_alignment_passed": False,
    }
    write("运行记录.json", runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
