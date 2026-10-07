# -*- coding: utf-8 -*-
"""G0一次性建档脚本：新增清单和审核记录，已有文件时拒绝覆盖。"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "outputs/jeon2019_optics"
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))
from main_g0_verify import file_sha256
from optics.stage02c_source import tree_sha
from optics.stage02f3_1_source import load_holdout_source


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    # 先检查目标，重复点击只会拒绝，不更新或清空已有记录。
    if any((DEST / name).exists() for name in ["audit.json", "manifest.json"]):
        raise FileExistsError("G0已建档；重复运行不会覆盖，请运行main_g0_verify.py核对")
    start = time.perf_counter()
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    items, evidence = load_holdout_source(PROJECT / "results/stage02f3_1/run_20261004_101012", PROJECT)
    latest = PROJECT / "results/stage02f3_2/run_20261004_231220"
    report = json.loads((latest / "metrics/validation.json").read_text(encoding="utf-8"))
    if not (report["status"] == "completed" and report["validation_passed"] is True
            and report["budget_exhausted"] is False and report["completed_jobs"] == report["planned_jobs"] == 5
            and not (latest / "failed.json").exists()):
        raise ValueError("最新基础重建没有完整通过")
    if json.loads((latest / "source_evidence.json").read_text(encoding="utf-8")) != evidence:
        raise ValueError("最新实验来源与实际递归核对不相容")
    for name, digest in json.loads((latest / "source_manifest.json").read_text(encoding="utf-8")).items():
        if file_sha256(PROJECT / name) != digest:
            raise ValueError("最新实验源码改变：" + name)
    source_dirs = [Path(s["path"]) for s in evidence["sources"]]
    diagnostic_dirs = [PROJECT / "results/stage01/run_20261001_195908",
                       ROOT / "outputs/复审并执行_02B_20261002/02B1正式复跑",
                       PROJECT / "results/stage02b2/run_20261002_164825"]
    folders = source_dirs + [latest] + diagnostic_dirs
    if any(not path.is_dir() for path in folders):
        raise FileNotFoundError("冻结诊断或正式来源目录缺失")
    # 只保存当前工作区中的源码和证据；不调整任何原有目录。
    code = {ROOT / "README.md", ROOT / ".gitignore", ROOT / "main_g0_verify.py",
            DEST / "README.md", Path(__file__).resolve()}
    code.update(p for p in PROJECT.iterdir() if p.is_file() and p.suffix in {".py", ".json", ".md", ".txt"})
    for folder in ["optics", "tests", "docs"]:
        code.update(p for p in (PROJECT / folder).rglob("*")
                    if p.is_file() and p.suffix in {".py", ".json", ".md"} and "__pycache__" not in p.parts)
    local = {p for folder in folders for p in folder.rglob("*") if p.is_file()}
    # 记录先前测试的日期；此次不把历史250项计为新运行。
    historical = ROOT / "outputs/执行_阶段02F3_2_20261004/最终审核.json"
    local.add(historical)
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0"))
    records = []
    for path in sorted(code | local):
        name = path.relative_to(ROOT).as_posix()
        records.append(dict(path=name, scope="code_document" if path in code else "local_evidence",
                            bytes=path.stat().st_size, sha256=file_sha256(path),
                            tracked_at_start=name in tracked))
    for source in evidence["sources"]:
        if tree_sha(Path(source["path"])) != source["sha256"]:
            raise RuntimeError("建档期间来源改变")
    import numpy, scipy, matplotlib
    old = json.loads(historical.read_text(encoding="utf-8"))
    audit = dict(timestamp_local=datetime.now().astimezone().isoformat(), base_commit=base_commit,
                 scope="Jeon光学编码复现＋基础重建验证；非25波段RGB/论文网络/双孔径",
                 recursive_source_validation_passed=True, source_count=len(source_dirs), control_items=len(items),
                 latest_completion_and_source_manifest_passed=True,
                 code_document_count=len(code), local_evidence_count=len(local),
                 frozen_bytes=sum(r["bytes"] for r in records),
                 npz_count=sum(Path(r["path"]).suffix == ".npz" for r in records),
                 environment=dict(python=platform.python_version(), numpy=numpy.__version__,
                                  scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                                  executable=sys.executable, platform=platform.platform()),
                 historical_tests=dict(date="2026-10-04", count=old["tests"], passed=old["passed"], rerun_in_g0=False),
                 limitations=["旋转方向论文对应未解决", "全部25波段收敛未验证", "真实GUI未人工验证",
                              "Git不包含NPZ及多数源run；独立数组归档和纯克隆恢复未验证"],
                 elapsed_seconds=time.perf_counter()-start)
    write_new(DEST / "audit.json", audit)
    records.append(dict(path=(DEST / "audit.json").relative_to(ROOT).as_posix(), scope="code_document",
                        bytes=(DEST / "audit.json").stat().st_size, sha256=file_sha256(DEST / "audit.json"),
                        tracked_at_start=False))
    manifest = dict(schema="jeon2019-g0-v1", target_tag="jeon-optics-g0-20261005", base_commit=base_commit,
                    project="outputs/jeon2019_optics", source_directories=[p.relative_to(ROOT).as_posix() for p in folders],
                    files=records)
    write_new(DEST / "manifest.json", manifest)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
