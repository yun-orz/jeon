# -*- coding: utf-8 -*-
"""阶段 02B-1 修正批的测试支撑工具（R8）。

本模块解决两条被审核点名的问题：

1. **不自动删除任何测试证据。** 旧测试用 ``tempfile.mkdtemp`` +
   ``shutil.rmtree`` 清理；审核没有授权删除，只能拦截。这里改为把证据写到
   **项目内的唯一目录** ``results/_test_evidence/<批次>/<用例>/`` 并**永久保留**，
   在目录里放一个 ``EVIDENCE_RETAINED.txt`` 说明来源与用途。任何真正的删除都必须
   先取得用户明确同意。

2. **子进程编码双向确定，不依赖终端预设。** 读端显式 ``encoding='utf-8'`` 只解决
   解码；审核指出中文 Windows 下子进程默认用 GBK 输出，因此必须**同时**显式指定
   子进程的 I/O 编码。这里通过 ``PYTHONIOENCODING=utf-8`` + ``PYTHONUTF8=1``
   设置子进程环境，读端用 ``encoding='utf-8'``，并且**不使用** ``errors='replace'``：
   编码不一致必须暴露为错误，而不是被静默替换成乱码。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_ROOT = PROJECT_ROOT / "results" / "_test_evidence" / "stage02b1"


def retained_evidence_dir(case_name: str, tag: str = "") -> Path:
    """分配一个**永久保留**的证据目录（绝不自动删除）。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in case_name)
    name = "%s_%s%s" % (safe, stamp, ("_" + tag) if tag else "")
    path = EVIDENCE_ROOT / name
    idx = 2
    while path.exists():
        path = EVIDENCE_ROOT / ("%s_%02d" % (name, idx))
        idx += 1
    path.mkdir(parents=True, exist_ok=False)
    (path / "EVIDENCE_RETAINED.txt").write_text(
        "本目录是自动化测试留下的**证据**，按阶段 02B-1 修正批 R8 的要求"
        "**不自动删除**。\n"
        "用途：保留测试运行产生的配置、日志与中间产物，便于复审复现。\n"
        "如需清理，请由用户明确指示后再删除。\n"
        "生成时间：%s\n用例：%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), case_name),
        encoding="utf-8")
    return path


def child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """构造**确定 UTF-8** 的子进程环境（同时约束编码与解码）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run_child(cmd: Sequence[str], cwd: Path,
              extra_env: Optional[Dict[str, str]] = None,
              timeout: float = 600.0) -> subprocess.CompletedProcess:
    """运行子进程并**确定性地**双向使用 UTF-8。

    关键点：``encoding='utf-8'``（读端）与 ``child_env`` 里的 ``PYTHONIOENCODING``
    （写端）必须成对出现；这里**不传** ``errors``，让真实编码问题直接报错。
    """
    return subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", env=child_env(extra_env), timeout=timeout)


def write_evidence(dirpath: Path, name: str, content: str) -> Path:
    p = Path(dirpath) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def describe_encoding_setup() -> Dict[str, Any]:
    """记录本次测试实际使用的编码设置，便于报告写明“谁设置的”。"""
    return {
        "python_executable": sys.executable,
        "stdout_encoding": getattr(sys.stdout, "encoding", None),
        "filesystem_encoding": sys.getfilesystemencoding(),
        "PYTHONIOENCODING_in_parent": os.environ.get("PYTHONIOENCODING"),
        "child_env_sets": {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                           "PYTHONLEGACYWINDOWSSTDIO": "0"},
        "read_side_encoding": "utf-8",
        "errors_policy": "未设置 errors（编码不一致必须报错，不用 replace 掩盖）",
        "note": ("子进程编码由本测试支撑模块显式设置，不依赖聊天终端或用户预置环境；"
                 "若必须依赖外部环境，报告会写明是谁设置的。"),
    }
