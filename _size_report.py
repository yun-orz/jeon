# -*- coding: utf-8 -*-
"""统计“精选结果”去掉 .npz 之后的真实体积，并列出大文件。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\PyCharmProjects\Jeon2019")
sel = json.loads((ROOT / "_select_runs.json").read_text(encoding="utf-8"))["selected"]
KEEP_EXT = {".png", ".json", ".csv", ".md", ".log", ".txt", ".svg"}

total = count = 0
big: list[tuple[int, Path]] = []
per_stage: dict[str, tuple[int, int]] = {}
for st, info in sel.items():
    if not info.get("run"):
        continue
    s_stage = n_stage = 0
    for f in Path(info["run"]).rglob("*"):
        if not f.is_file() or f.suffix.lower() not in KEEP_EXT:
            continue
        s = f.stat().st_size
        total += s
        count += 1
        s_stage += s
        n_stage += 1
        if s > 2 * 1024 * 1024:
            big.append((s, f))
    per_stage[st] = (n_stage, s_stage)

print("精选结果（图/指标/报告/日志）: %d 个文件, %.1f MB" % (count, total / 2 ** 20))
print()
print("按阶段:")
for st, (n, s) in sorted(per_stage.items(), key=lambda kv: -kv[1][1]):
    print("  %-16s %4d 文件  %8.2f MB" % (st, n, s / 2 ** 20))
print()
print("超过 2MB 的文件:")
for s, f in sorted(big, reverse=True)[:15]:
    rel = str(f).replace(str(ROOT) + "\\", "")
    print("  %7.2f MB  %s" % (s / 2 ** 20, rel))

# 全部源码/文档（非 results 大目录）体积，供总量估算
print()
print("== 排除清单（不进仓库）==")
for pat, why in [
    ("outputs/jeon2019_optics/results/**/*.npz", "结果大数组，可由源码+配置重算"),
    ("outputs/jeon2019_optics/results/_test_evidence/**", "测试自动产物"),
    ("outputs/jeon2019_optics/results/test_*/**", "测试入口产物"),
    ("work/papers/**", "论文 PDF，有版权，不宜公开"),
    ("outputs/jeon2019_optics/.idea/**", "PyCharm 个人配置"),
    ("**/__pycache__/**, **/*.pyc", "字节码缓存"),
]:
    print("  %-58s %s" % (pat, why))
