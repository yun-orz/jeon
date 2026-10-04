# -*- coding: utf-8 -*-
"""为 GitHub 上传挑选“每个阶段有代表性的结果 run”，并生成精选清单。

原则
----
* 每个阶段取**最新**的、并且**同时含有图与指标**的 run 目录；
* 干净 run 优先（名字以 `run_` 开头，或带 `completion.json`），
  临时副本/篡改测试/探针目录（名字含 probe/tamper/伪造/副本/_final_ 等）排除；
* 只保留“人能直接看/审”的产物：PNG 图、JSON/CSV 指标、报告 MD、日志；
  **不保留 .npz 大数组**（占 96% 体积，且可由源码+配置重算）。

输出：`_select_runs.json`，含每个阶段选中的 run、跳过原因与体积。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"D:\PyCharmProjects\Jeon2019")
RESULTS = ROOT / "outputs" / "jeon2019_optics" / "results"
OUT = ROOT / "_select_runs.json"

# 排除临时/测试性目录（不是给审核看的正式 run）
EXCLUDE_PAT = re.compile(
    r"(probe|tamper|伪造|篡改|副本|_final_|deleted|scratch|tmp|debug)", re.I)

# 整个阶段级的测试产物：不属于“代表性结果”，只放大小且无审核价值
EXCLUDE_STAGES = {"_test_evidence", "test_stage02_entry", "test_stage02_r2",
                  "test_failures"}

# 固定指定的代表 run（审核报告明确引用过的必须入选，不用自动挑）
PINNED = {
    # 本批（02B-1 修正批）的正式送审 run
    "stage02b1": "run_20261002_163101",
    # 审核报告引用的 02A 正式 run
    "stage02a": "run_20261001_215848_02",
}


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def has_figures(p: Path) -> bool:
    return any(p.rglob("*.png"))


def has_metrics(p: Path) -> bool:
    return any(p.rglob("*.json")) or any(p.rglob("*.csv"))


def score(p: Path) -> tuple:
    """越大越好：优先正式 run、优先有 completion、再按名字（≈时间）排序。"""
    name = p.name
    formal = name.startswith("run_")
    complete = (p / "completion.json").exists()
    figs = has_figures(p)
    mets = has_metrics(p)
    return (formal, complete, figs, mets, name)


selected: dict[str, dict] = {}
skipped: dict[str, list[str]] = {}

for stage in sorted(RESULTS.iterdir(), key=lambda d: d.name):
    if not stage.is_dir():
        continue
    if stage.name in EXCLUDE_STAGES:
        skipped.setdefault(stage.name, []).append("整阶段排除：测试产物，非代表性结果")
        continue
    # 固定指定优先
    pinned_name = PINNED.get(stage.name)
    if pinned_name:
        pinned = stage / pinned_name
        if pinned.is_dir():
            selected[stage.name] = {
                "run": str(pinned), "run_name": pinned.name,
                "size_MB": round(dir_size(pinned) / 2 ** 20, 1),
                "npz_count": sum(1 for f in pinned.rglob("*.npz")),
                "png": sum(1 for f in pinned.rglob("*.png")),
                "json_csv": (sum(1 for f in pinned.rglob("*.json"))
                             + sum(1 for f in pinned.rglob("*.csv"))),
                "pinned": True, "candidates": [],
            }
            continue
        skipped.setdefault(stage.name, []).append(
            "指定的 %s 不存在，改为自动挑选" % pinned_name)
    cands = []
    for d in stage.iterdir():
        if not d.is_dir():
            continue
        if EXCLUDE_PAT.search(d.name):
            skipped.setdefault(stage.name, []).append("%s（临时/测试目录）" % d.name)
            continue
        if not has_figures(d):
            skipped.setdefault(stage.name, []).append("%s（无图）" % d.name)
            continue
        cands.append(d)
    if not cands:
        selected[stage.name] = {"run": None, "reason": "该阶段没有含图的 run"}
        continue
    best = max(cands, key=score)
    size = dir_size(best)
    npz = sum(1 for f in best.rglob("*.npz"))
    selected[stage.name] = {
        "run": str(best),
        "run_name": best.name,
        "size_MB": round(size / 2 ** 20, 1),
        "npz_count": npz,
        "png": sum(1 for f in best.rglob("*.png")),
        "json_csv": sum(1 for f in best.rglob("*.json")) + sum(1 for f in best.rglob("*.csv")),
        "candidates": [d.name for d in sorted(cands, key=score, reverse=True)],
    }

payload = {
    "policy": ("每阶段保留最新且含图与指标的正式 run；只留图/指标/报告/日志，"
               "不含 .npz 大数组"),
    "selected": selected,
    "skipped": skipped,
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

total_keep = 0
print("%-16s %-34s %8s %5s %5s" % ("阶段", "选中 run", "MB", "图", "指标"))
for st, info in selected.items():
    if not info.get("run"):
        print("%-16s %-34s %8s" % (st, "（无）", info.get("reason", "")))
        continue
    total_keep += info["size_MB"]
    print("%-16s %-34s %8.1f %5d %5d"
          % (st, info["run_name"][:34], info["size_MB"], info["png"], info["json_csv"]))
print()
print("选中 %d 个 run，原始体积合计 %.1f MB（去掉 .npz 后会小很多）"
      % (sum(1 for v in selected.values() if v.get("run")), total_keep))
print("清单已写入:", OUT)
