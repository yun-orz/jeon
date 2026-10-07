"""正式基线入口；M0提供只读check/report，其余命令必须经过对应里程碑实施。"""
import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from audit import HERE, ROOT, g0_check, protocol_check, read_json, safe_path, sha256, verify_manifest


def load_m0():
    index = read_json(HERE / "m0_index.json")
    if index.get("schema") != "jeon2019-m0-index-v1":
        raise ValueError("M0索引格式不符")
    for item in index["artifacts"]:
        path = safe_path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError("M0交接对象SHA不符：" + item["path"])
    audit = read_json(safe_path(index["audit"]))
    if audit.get("status") != "passed" or not audit.get("acceptance", {}).get("all_passed"):
        raise ValueError("M0尚未通过验收")
    return index, audit


def environment():
    versions = {}
    for name in ["numpy", "scipy", "matplotlib", "torch", "sympy"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    disk = shutil.disk_usage(ROOT)
    return {"python": sys.executable, "python_version": sys.version.split()[0], "versions": versions,
            "free_bytes": disk.free, "free_gib": round(disk.free / 1024 ** 3, 3),
            "gpu_execution_verified": False,
            "gpu_note": "M0不执行CUDA验收；GPU状态只能由M1实际审核登记",
            "space_note": "仅报告当前空间；M1下载安装和M3数据需求尚未核算，不能据此判定大型阶段可执行"}


def check():
    index, audit = load_m0()
    protection = verify_manifest(read_json(safe_path(index["protection"])))
    protocol = protocol_check()
    g0 = g0_check()
    passed = protection["passed"] and protocol["passed"] and g0["passed"]
    return {"status": "passed" if passed else "failed", "scope": "M0历史保护及协议交接检查",
            "protection": protection, "protocol": protocol, "g0": g0, "environment": environment(),
            "milestones": {"M0": audit["status"], **{"M" + str(i): "dependency_pending" for i in range(1, 8)}},
            "paper_alignment_passed": False}


def report():
    index, audit = load_m0()
    return {"M0": "passed", "M1-M7": "尚未实施，不能据此启动正式训练或最终测试",
            "report": str(safe_path(index["report"])), "historical_stages": str(safe_path(index["stages"])),
            "parameters": str(HERE / "parameter_evidence.json"), "tasks": str(HERE / "tasks"),
            "next_inputs": audit["next_inputs"], "environment": environment(), "paper_alignment_passed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="check",
                        choices=["check", "optics", "data", "smoke", "train", "evaluate", "report"])
    args, remaining = parser.parse_known_args()
    try:
        if args.command not in {"check", "report"}:
            raise ValueError("该命令尚未实现；请按tasks执行对应里程碑：" + args.command)
        if remaining:
            raise ValueError("未知参数：" + " ".join(remaining))
        result = check() if args.command == "check" else report()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status") == "failed" else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "paper_alignment_passed": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
