# -*- coding: utf-8 -*-
"""计算应用排除规则后，**实际会进入仓库**的文件与体积。

规则必须与稍后写入的 `.gitignore` 保持一致；本脚本是推送前的自查闸门。
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

ROOT = Path(r"D:\PyCharmProjects\Jeon2019")

# 目录级排除（相对仓库根的 POSIX 前缀）
EXCLUDE_DIR_PREFIXES = [
    "work/papers/",
    "outputs/jeon2019_optics/.idea/",
    "outputs/jeon2019_optics/results/_test_evidence/",
]
# 通配排除（对相对路径做 fnmatch）
EXCLUDE_GLOBS = [
    "*.npz",
    "*.pyc",
    "*/__pycache__/*",
    ".idea/*",
    "*.pdf",
]
# 体积闸门：任何超过此值的文件都会被点名（GitHub 单文件硬限 100MB）
SIZE_WARN = 50 * 1024 * 1024

kept: list[tuple[int, str]] = []
skipped = {"npz": 0, "pdf": 0, "pycache": 0, "idea": 0, "evidence": 0, "other": 0}

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d != ".git"]
    rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
    rel_dir_slash = (rel_dir + "/") if rel_dir != "." else ""
    if any(rel_dir_slash.startswith(p) for p in EXCLUDE_DIR_PREFIXES):
        for f in filenames:
            skipped["evidence" if "_test_evidence" in rel_dir_slash else "other"] += 1
        continue
    for f in filenames:
        rel = (rel_dir_slash + f) if rel_dir_slash else f
        ext = Path(f).suffix.lower()
        if ext == ".npz":
            skipped["npz"] += 1
            continue
        if ext == ".pdf":
            skipped["pdf"] += 1
            continue
        if ext == ".pyc":
            skipped["pycache"] += 1
            continue
        if any(fnmatch.fnmatch(rel, g) for g in EXCLUDE_GLOBS):
            skipped["other"] += 1
            continue
        kept.append(((Path(dirpath) / f).stat().st_size, rel))

total = sum(s for s, _ in kept)
print("== 会进入仓库 ==")
print("  文件数: %d" % len(kept))
print("  总体积: %.1f MB" % (total / 2 ** 20))
print()
print("== 按顶层目录 ==")
bydir: dict[str, tuple[int, int]] = {}
for s, rel in kept:
    top = rel.split("/")[0] if "/" in rel else "(根目录)"
    if rel.startswith("outputs/"):
        parts = rel.split("/")
        top = "/".join(parts[:2]) + ("/…" if len(parts) > 2 else "")
    n, b = bydir.get(top, (0, 0))
    bydir[top] = (n + 1, b + s)
for k, (n, b) in sorted(bydir.items(), key=lambda kv: -kv[1][1])[:20]:
    print("  %-42s %5d 文件 %8.2f MB" % (k, n, b / 2 ** 20))
print()
print("== 被排除 ==")
for k, v in skipped.items():
    print("  %-10s %d 文件" % (k, v))
print()
over = [(s, rel) for s, rel in kept if s > SIZE_WARN]
print("== 体积闸门（> %.0f MB）==" % (SIZE_WARN / 2 ** 20))
if over:
    for s, rel in sorted(over, reverse=True):
        print("  %8.2f MB  %s" % (s / 2 ** 20, rel))
else:
    print("  无超限文件（GitHub 单文件硬限 100 MB）")
