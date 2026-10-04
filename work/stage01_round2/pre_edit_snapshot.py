# -*- coding: utf-8 -*-
"""第二轮修正：修改前的可追溯快照（只新增文件，不改动生产代码）。

输出：
* ``pre_edit_sha256.json``：修改前所有生产文件的 SHA256、大小、时间戳
* ``environment_pre.json``：解释器、依赖、CPU、工作目录、实际命令

用法::

    python -B work/stage01_round2/pre_edit_snapshot.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
OUT_DIR = Path(r"D:\PyCharmProjects\Jeon2019\work\stage01_round2")
SKIP_DIRS = {"__pycache__", ".venv", "results"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            yield Path(dirpath) / name


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for p in walk(PROJECT):
        rel = str(p.relative_to(PROJECT)).replace("\\", "/")
        st = p.stat()
        files[rel] = {
            "sha256": sha256_file(p),
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }

    env = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "cwd_at_invocation": os.getcwd(),
        "project_root": str(PROJECT),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical": os.cpu_count(),
        "argv": sys.argv,
    }
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            m = __import__(mod)
            env[mod + "_version"] = getattr(m, "__version__", None)
            env[mod + "_file"] = getattr(m, "__file__", None)
        except Exception as exc:                       # pragma: no cover
            env[mod + "_version"] = "import failed: %r" % (exc,)
    try:
        out = subprocess.run(
            ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"],
            capture_output=True, text=True, timeout=20)
        env["total_physical_memory_bytes"] = out.stdout.strip()
    except Exception as exc:                           # pragma: no cover
        env["total_physical_memory_bytes"] = "unavailable: %r" % (exc,)
    try:
        out = subprocess.run(["wmic", "cpu", "get", "Name", "/value"],
                             capture_output=True, text=True, timeout=20)
        env["cpu_name"] = out.stdout.strip()
    except Exception as exc:                           # pragma: no cover
        env["cpu_name"] = "unavailable: %r" % (exc,)

    (OUT_DIR / "pre_edit_sha256.json").write_text(
        json.dumps({"project_root": str(PROJECT), "captured_at": env["captured_at"],
                    "file_count": len(files), "files": files},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "environment_pre.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已记录 %d 个文件的 SHA256 → %s" % (len(files), OUT_DIR / "pre_edit_sha256.json"))
    print("环境 → %s" % (OUT_DIR / "environment_pre.json"))
    print("python:", env["python_version"])
    print("numpy/scipy/matplotlib:", env.get("numpy_version"), env.get("scipy_version"),
          env.get("matplotlib_version"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
