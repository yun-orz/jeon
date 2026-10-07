# -*- coding: utf-8 -*-
"""M1 续传器入口：官方Torch轮子续传下载与整包SHA核验，随后可选离线安装。

PyCharm 可直接无参数运行（等同 ``probe``）。不做任何删除；已有前缀只读复制。

子命令：

- ``probe``：HEAD核对官方链接、总长与头信息，并报告当前前缀与剩余字节。
- ``download``：执行分块续传到工作目录，核对整包官方SHA，写 download_manifest.json。
- ``install``：仅在整包官方SHA通过后离线安装到新GPU环境（Torch与SymPy）。
- ``all``：probe 后 download，并在整包SHA通过时离线安装到新GPU环境。

退出码：0 通过；2 失败（保留全部文件，可重跑同一命令续传）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_resume import Resumer, ResumeError, sha256_file, write_json  # noqa: E402

ROOT = HERE.parents[2]
TORCH_URL = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl"
TORCH_BYTES = 2449372784
TORCH_SHA256 = "9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f"
PREFIX = ROOT / "work/dependencies/gpu_cu121_20261006/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
WORKDIR = ROOT / "work/dependencies/gpu_cu121_resume_m1_20261007"
ENVIRONMENT = ROOT / "work/environments/jeon_gpu_cu121"
SYMPY_URL = "https://files.pythonhosted.org/packages/b2/fe/81695a1aa331a842b582453b605175f419fe8540355886031328089d840a/sympy-1.13.1-py3-none-any.whl"
SYMPY_SHA256 = "db36cdc64bf61b9b24578b6f7bab1ecdd2452cf008f34faa33776680c26d66f8"
BLOCK_BYTES = 32 * 1024 * 1024


def resumer(block_bytes=BLOCK_BYTES, attempts=3):
    return Resumer(url=TORCH_URL, total_bytes=TORCH_BYTES, sha256=TORCH_SHA256,
                   destination=WORKDIR / "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl",
                   block_bytes=block_bytes, attempts=attempts, prefix=PREFIX,
                   expected_headers={"last-modified": "Tue, 29 Oct 2024 23:16:21 GMT"})


def probe():
    engine = resumer()
    engine.directory.mkdir(parents=True, exist_ok=True)
    engine.load_state()
    result = engine.probe()
    prefix = {"path": str(PREFIX), "exists": PREFIX.is_file()}
    if prefix["exists"]:
        prefix["bytes"] = PREFIX.stat().st_size
        prefix["remaining_bytes"] = TORCH_BYTES - prefix["bytes"]
    assembled = engine.destination
    whole = {"path": str(assembled), "exists": assembled.is_file()}
    if whole["exists"]:
        whole["bytes"] = assembled.stat().st_size
        whole["matches_frozen_bytes"] = whole["bytes"] == TORCH_BYTES
    report = {"status": "passed" if result["passed"] else "failed", "url": TORCH_URL,
              "frozen_total_bytes": TORCH_BYTES, "frozen_sha256": TORCH_SHA256,
              "head": result["observed"], "head_problems": result["problems"],
              "workdir": str(WORKDIR), "prefix": prefix, "assembled": whole,
              "block_bytes": BLOCK_BYTES, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(WORKDIR / "probe.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


def download():
    engine = resumer()
    try:
        manifest = engine.run()
    except ResumeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "workdir": str(WORKDIR),
                          "note": "保留全部分块与日志，可重跑同一命令继续"}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "passed", "destination": manifest["destination"],
                      "bytes": manifest["total_bytes"], "whole_sha256": manifest["assembled_sha256"],
                      "blocks": len(manifest["blocks"]), "prefix_bytes": manifest["prefix_bytes"],
                      "manifest": str(WORKDIR / "download_manifest.json")}, ensure_ascii=False, indent=2))
    return 0


def sympy(engine_dir):
    """官方SymPy轮子独立下载与SHA核验；已存在且通过则复用。"""
    target = WORKDIR / "sympy-1.13.1-py3-none-any.whl"
    if target.is_file() and sha256_file(target) == SYMPY_SHA256:
        return target, "复用已核验文件"
    import urllib.request
    with urllib.request.urlopen(SYMPY_URL, timeout=300) as response, target.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    actual = sha256_file(target)
    if actual != SYMPY_SHA256:
        raise ResumeError("SymPy官方SHA不符：%s" % actual)
    return target, "新下载并通过官方SHA"


def install():
    import subprocess
    torch_wheel = WORKDIR / "torch-2.5.1+cu121-cp310-cp310-win_amd64.whl"
    if not torch_wheel.is_file() or torch_wheel.stat().st_size != TORCH_BYTES:
        print(json.dumps({"status": "failed", "error": "整包缺失或长度不符，拒绝安装",
                          "path": str(torch_wheel)}, ensure_ascii=False))
        return 2
    digest = sha256_file(torch_wheel)
    if digest != TORCH_SHA256:
        print(json.dumps({"status": "failed", "error": "整包SHA不符，拒绝安装", "sha256": digest},
                         ensure_ascii=False))
        return 2
    sympy_wheel, note = sympy(ENVIRONMENT)
    python = ENVIRONMENT / "Scripts/python.exe"
    if not python.is_file():
        print(json.dumps({"status": "failed", "error": "找不到新GPU环境解释器", "path": str(python)},
                         ensure_ascii=False))
        return 2
    command = [str(python), "-B", "-m", "pip", "install", "--ignore-installed", "--no-deps",
               "--no-index", "--no-cache-dir", "--no-compile", str(torch_wheel), str(sympy_wheel)]
    print("[安装] " + " ".join(command), flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = WORKDIR / "install_log.txt"
    log.write_text("命令：" + " ".join(command) + "\n退出码：%d\n\nSTDOUT\n%s\n\nSTDERR\n%s"
                   % (completed.returncode, completed.stdout, completed.stderr), encoding="utf-8")
    print(completed.stdout[-4000:])
    print(completed.stderr[-4000:], file=sys.stderr)
    report = {"status": "passed" if completed.returncode == 0 else "failed",
              "command": command, "returncode": completed.returncode, "sympy": note,
              "sympy_wheel_sha256": sha256_file(sympy_wheel), "log": str(log),
              "environment": str(ENVIRONMENT), "torch_wheel_sha256": digest}
    write_json(WORKDIR / "install_result.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if completed.returncode == 0 else 2


def selftest_fixture(block_bytes=1024, attempts=2):
    """用本地样例服务实际跑一遍续传器；不访问外网、不重复下载2.3GiB。"""
    from m1_fixture_server import start_fixture
    payload = bytes(range(256)) * 4096          # 1 MiB 样例
    fixture, server, url = start_fixture(payload)
    workdir = WORKDIR / "selftest_run"
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        engine = Resumer(url=url, total_bytes=len(payload), sha256=fixture.sha256(),
                         destination=workdir / "sample.bin", block_bytes=block_bytes,
                         attempts=attempts, logger=lambda *a: None)
        manifest = engine.run()
        print(json.dumps({"status": "passed", "blocks": len(manifest["blocks"]),
                          "sha256": manifest["assembled_sha256"], "bytes": manifest["total_bytes"],
                          "range_requests": fixture.range_requests}, ensure_ascii=False, indent=2))
        return 0
    finally:
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="probe",
                        choices=["probe", "download", "install", "all", "selftest"])
    parser.add_argument("--block-bytes", type=int, default=BLOCK_BYTES)
    args = parser.parse_args()
    try:
        if args.command == "probe":
            return probe()
        if args.command == "download":
            return download()
        if args.command == "install":
            return install()
        if args.command == "selftest":
            return selftest_fixture()
        code = probe()
        if code != 0:
            return code
        code = download()
        if code != 0:
            return code
        return install()
    except ResumeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
