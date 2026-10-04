# -*- coding: utf-8 -*-
"""生成清理清单 `_cleanup_manifest.json`（只写清单，不删除任何东西）。

口径（用户 2026-10-02 确认）：
* A 组：本会话自建的调试探针、空 run、缓存、临时脚本 —— 全部删除；
* B 组：中间过程 run —— 保留 224235 / 225933 / 230511 / 163101，其余删除；
* C 组：测试证据 —— 每个用例前缀保留最新 2 个目录，其余删除；
* D 组：历史测试产物 —— 每个目录保留最新 5 个文件，其余删除。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
WORK = Path(r"D:\PyCharmProjects\Jeon2019\work")
OUT = Path(r"D:\PyCharmProjects\Jeon2019\_cleanup_manifest.json")

delete: list[str] = []
keep: list[str] = []
reasons: dict[str, str] = {}


def add(p: Path, why: str) -> None:
    if p.exists():
        delete.append(str(p))
        reasons[str(p)] = why


# ---------------- A 组：自建调试与临时物 ----------------
for name in ("_probe_dbg", "_probe_dbg2", "_probe_dbg3", "_probe_dbg4",
             "_r4_height_probe", "_r4_control_probe", "_r4_preview_probe",
             "_r5_analyze_probe", "_final_height", "_final_control",
             "_final_preview", "_final_analyze",
             "run_20261001_223308", "run_20261001_223409"):
    add(ROOT / "results" / "stage02b1" / name, "A 组：本会话自建的调试/终检探针或空 run")

add(ROOT / "_src_before.json", "A 组：本会话核对 from-run 只读性用的临时快照")
for cache in (ROOT / "__pycache__", ROOT / "optics" / "__pycache__",
              ROOT / "tests" / "__pycache__"):
    add(cache, "A 组：Python 字节码缓存，可随时重建")
add(WORK / "stage02b1" / "probe_single_field.py", "A 组：本会话一次性诊断脚本")

# ---------------- B 组：中间过程 run ----------------
KEEP_RUNS = {"run_20261001_224235",   # 审核指定送审
             "run_20261001_225933",   # 审核额外正式 run
             "run_20261001_230511",   # 审核独立 run
             "run_20261002_163101",   # 本批最终送审 run
             "run_20261002_163254"}   # 非本会话产生的 run，按“保留历史”原则不动
s1 = ROOT / "results" / "stage02b1"
for d in sorted(s1.iterdir()):
    if not d.is_dir() or not d.name.startswith("run_"):
        continue
    if d.name in KEEP_RUNS:
        keep.append(str(d))
        continue
    if (d / "completion.json").exists():
        add(d, "B 组：中间过程 run，最终送审 run 已含等效且修正后的结果")
    else:
        add(d, "B 组：无 completion 的未完成 run")

# ---------------- C 组：测试证据（每用例保留最新 2 个） ----------------
te = ROOT / "results" / "_test_evidence" / "stage02b1"
if te.is_dir():
    prefix = lambda n: re.sub(r"_\d{8}_\d{6}(_\d+)?$", "", n)
    groups: dict[str, list[Path]] = {}
    for d in te.iterdir():
        if d.is_dir():
            groups.setdefault(prefix(d.name), []).append(d)
    for pfx, lst in sorted(groups.items()):
        lst = sorted(lst, key=lambda d: d.name)
        keep += [str(d) for d in lst[-2:]]
        for d in lst[:-2]:
            add(d, "C 组：用例 %s 的旧证据，同用例已保留最新 2 个" % pfx)

# ---------------- D 组：历史测试产物（每目录保留最新 5 个文件） ----------------
for rel in ("results/test_stage02_entry", "results/test_stage02_r2",
            "results/test_failures", "results/_nosave"):
    d = ROOT / rel
    if not d.is_dir():
        continue
    files = [f for f in d.rglob("*") if f.is_file()]
    files.sort(key=lambda f: (f.stat().st_mtime, str(f)))
    newest = {f.resolve() for f in files[-5:]}
    keep += [str(f) for f in sorted(newest, key=str)]
    for f in files[:-5]:
        add(f, "D 组：历史测试产物，同目录已保留最新 5 个文件")

manifest = {
    "created": "2026-10-02",
    "method": "删除到回收站（可还原），非永久删除",
    "confirmed_by_user": "删除到回收站；C 组每用例保留最新 2 个",
    "keep_policy": {
        "B 组 run": sorted(KEEP_RUNS),
        "C 组": "每用例前缀保留最新 2 个目录",
        "D 组": "每目录保留最新 5 个文件",
    },
    "counts": {"delete": len(delete), "keep": len(keep)},
    "delete": delete,
    "keep": keep,
    "reasons": reasons,
}
OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

files = bytes_ = 0
for p in delete:
    pp = Path(p)
    if pp.is_dir():
        for f in pp.rglob("*"):
            if f.is_file():
                files += 1
                bytes_ += f.stat().st_size
    elif pp.is_file():
        files += 1
        bytes_ += pp.stat().st_size

print("清单已写入:", OUT)
print("删除项 %d 个（涉及 %d 个文件，约 %.1f MB）" % (len(delete), files, bytes_ / 2 ** 20))
print("保留项 %d 个" % len(keep))
from collections import Counter
c = Counter(v.split("：")[0] for v in reasons.values())
for k in sorted(c):
    print("  %s: %d 项" % (k, c[k]))
