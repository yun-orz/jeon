# -*- coding: utf-8 -*-
"""推送前体积方案对比：全量（仅排除大数组）vs 精选（只留每阶段代表 run）。

只做统计，不修改任何文件。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\PyCharmProjects\Jeon2019")
OPT = ROOT / "outputs" / "jeon2019_optics"
RESULTS = OPT / "results"
sel = json.loads((ROOT / "_select_runs.json").read_text(encoding="utf-8"))["selected"]
CURATED = {Path(v["run"]).resolve() for v in sel.values() if v.get("run")}

KEEP_EXT = {".py", ".json", ".md", ".txt", ".csv", ".png", ".svg", ".log",
            ".ps1", ".iml", ".xml", ".cfg", ".toml", ".yml", ".yaml",
            ".jsonl", ".bat", ".ipynb"}

EXCLUDE_DIR_TOPS = {
    OPT / ".idea",
    ROOT / "work" / "papers",
    RESULTS / "_test_evidence",
    RESULTS / "test_stage02_entry",
    RESULTS / "test_stage02_r2",
    RESULTS / "test_failures",
}


def under(p: Path, parents: set[Path]) -> bool:
    return any(par == p or par in p.parents for par in parents)


rows = []
for mode in ("全量（排除 .npz/PDF/缓存）", "精选（results 只留代表 run）"):
    n = tot = 0
    big = []
    for dirpath, dirnames, filenames in Path(ROOT).walk() if hasattr(Path(ROOT), "walk") else []:
        pass
    import os
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        cur = Path(dirpath)
        if under(cur, EXCLUDE_DIR_TOPS):
            dirnames[:] = []
            continue
        for f in filenames:
            fp = cur / f
            if fp.suffix.lower() not in KEEP_EXT:
                continue
            if mode.startswith("精选") and under(fp, {RESULTS}):
                if not any(c == fp.parent or c in fp.parents for c in CURATED):
                    continue
            s = fp.stat().st_size
            n += 1
            tot += s
            if s > 20 * 1024 * 1024:
                big.append((s, str(fp).replace(str(ROOT) + "\\", "")))
    rows.append((mode, n, tot, big))

print("%-28s %6s %12s %8s" % ("方案", "文件数", "体积", "GB"))
for mode, n, tot, _ in rows:
    print("%-28s %6d %9.1f MB %8.2f" % (mode, n, tot / 2 ** 20, tot / 2 ** 30))
print()
for mode, n, tot, big in rows:
    print("[%s] 超过 20MB 的文件：" % mode)
    if big:
        for s, rel in sorted(big, reverse=True)[:8]:
            print("   %7.1f MB  %s" % (s / 2 ** 20, rel))
    else:
        print("   无")
