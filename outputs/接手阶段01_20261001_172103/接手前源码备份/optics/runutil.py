# -*- coding: utf-8 -*-
"""运行环境、日志与结果目录管理（阶段 01 工具层，不含光学计算）。

约定
----
* 所有路径都基于 ``Path(__file__).resolve().parent.parent``（即项目根），
  与启动时的工作目录无关。
* 每次运行新建 ``results/<stage>/<run编号>/`` 目录；若编号已存在则递增，
  **绝不删除或覆盖已有运行目录**。
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"


def now_stamp() -> str:
    """本地时间戳，用于运行编号。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def allocate_run_dir(stage: str, force_new: bool = False,
                     root: Optional[Path] = None,
                     results_root: Optional[Path] = None) -> Tuple[Path, str]:
    """分配一个新的运行目录 ``<root>/<stage>/run_<时间戳>[_NN]/``。

    ``root``（推荐）与旧名 ``results_root`` 等价；缺省用项目内的 ``results/``，
    因此默认输出位置始终按项目 ``__file__`` 解析，与启动时的工作目录无关。

    Returns
    -------
    (run_dir, run_id)
        目录已创建；``run_id`` 是目录名（例如 ``run_20260930_153012``）。
        若同名目录已存在，则追加 ``_02``、``_03``…，**不删除任何已有内容**。
    """
    base_root = root if root is not None else (
        Path(results_root) if results_root is not None else RESULTS_ROOT)
    stage_dir = Path(base_root) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    base = "run_" + now_stamp()
    candidate = stage_dir / base
    if candidate.exists() or force_new:
        idx = 2
        while True:
            candidate = stage_dir / ("%s_%02d" % (base, idx))
            if not candidate.exists():
                break
            idx += 1
    candidate.mkdir(parents=True, exist_ok=False)
    for sub in ("arrays", "figures", "metrics"):
        (candidate / sub).mkdir()
    return candidate, candidate.name


def setup_logger(run_dir: Optional[Path], level: str = "INFO", name: str = "jeon2019"
                 ) -> logging.Logger:
    """日志器：有结果目录时同时写 ``run.log``，否则**只写标准输出**。

    ``run_dir=None`` 对应 ``--no-save``：此时不创建也不覆盖任何文件
    （旧版固定写入 ``results/_nosave``，会覆盖上一次的日志，已修正）。
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    if run_dir is not None:
        fh = logging.FileHandler(Path(run_dir) / "run.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def environment_info(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """记录解释器、依赖版本、平台与时间。缺失的依赖如实记为 None。"""
    info: Dict[str, Any] = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "cwd_at_invocation": os.getcwd(),
        "project_root_resolved": str(PROJECT_ROOT),
        "argv": list(sys.argv),
    }
    for mod in ("numpy", "scipy", "matplotlib", "pytest"):
        try:
            m = __import__(mod)
            info["%s_version" % mod] = getattr(m, "__version__", None)
        except Exception as exc:  # pragma: no cover - 仅在缺依赖时触发
            info["%s_version" % mod] = None
            info["%s_import_error" % mod] = "%s: %s" % (type(exc).__name__, exc)
    if extra:
        info.update(extra)
    return info


def write_json(path: Path, payload: Any) -> None:
    """写 UTF-8 JSON（中文不转义），供人工复核。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)


class Timer:
    """上下文计时器，用于记录每个验证块的真实耗时。"""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.t0
