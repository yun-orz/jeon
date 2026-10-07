"""正式基线入口；M0/M1检查、M2物理冻结及M3正式数据流水线。"""
import argparse
import importlib.metadata
import json
import shutil
import sys
import time
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
    try:
        import m1_environment
        report = m1_environment.check(persist=False)
        return {"stage": "M1", "status": report["status"], "gpu_hardware": report["gpu_hardware"],
                "interpreters": report["interpreters"], "environment_inheritance": report["environment_inheritance"],
                "prefix": report["prefix"], "space": report["space"],
                "gpu_environment_ready": report["gpu_environment_ready"],
                "torch_from_environment": report["torch_from_environment"],
                "gpu_execution_verified": report["gpu_environment_ready"],
                "next_action": report["next_action"],
                "note": "M1只读检查：不安装、不下载、不删除；GPU是否真正生效以Torch实际导入位置为准"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # M3不依赖GPU；旧环境诊断异常须明确登记，不能阻断独立数据审核或冒称M1通过。
        disk = shutil.disk_usage(ROOT)
        return {"stage":"M1", "status":"failed", "error":repr(exc),
                "free_bytes":disk.free, "gpu_execution_verified":False,
                "gpu_environment_ready":False,
                "note":"M1环境检查异常，需单独修复；未将此状态视为M1通过。M3数据阶段不依赖GPU。"}
    except ImportError:
        versions = {}
        for name in ["numpy", "scipy", "matplotlib", "torch", "sympy"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        disk = shutil.disk_usage(ROOT)
        return {"stage": "M0", "python": sys.executable, "python_version": sys.version.split()[0], "versions": versions,
                "free_bytes": disk.free, "free_gib": round(disk.free / 1024 ** 3, 3),
                "gpu_execution_verified": False,
                "gpu_note": "M1检查模块不可用；GPU状态只能由M1实际审核登记",
                "space_note": "仅报告当前空间；M1下载安装和M3数据需求尚未核算，不能据此判定大型阶段可执行"}


def check(stage=None):
    index, audit = load_m0()
    protection = verify_manifest(read_json(safe_path(index["protection"])))
    protocol = protocol_check()
    g0 = g0_check()
    passed = protection["passed"] and protocol["passed"] and g0["passed"]
    live = environment()
    if stage is not None and live.get("stage") == "M1":
        live["stage"] = stage
    return {"status": "passed" if passed else "failed", "scope": "M0历史保护、协议交接及M1只读环境检查",
            "protection": protection, "protocol": protocol, "g0": g0, "environment": live,
            "milestones": {"M0": audit["status"], **{"M" + str(i): "dependency_pending" for i in range(1, 8)}},
            "paper_alignment_passed": False}


def report():
    index, audit = load_m0()
    return {"M0": "passed", "M1,M4-M7": "须以各里程碑实际审核为准，不能据此启动正式训练或最终测试",
            "report": str(safe_path(index["report"])), "historical_stages": str(safe_path(index["stages"])),
            "parameters": str(HERE / "parameter_evidence.json"), "tasks": str(HERE / "tasks"),
            "next_inputs": audit["next_inputs"], "environment": environment(), "paper_alignment_passed": False}


def persist_check(result, stage):
    """按实际检查范围写入新结果目录，不覆盖历史；M2检查留在M2目录。"""
    milestone = "m3" if "M3" in result else "m2" if "M2" in result else "m1"
    base = ROOT / "outputs/jeon2019_optics/results/reproduction_v1" / milestone
    destination = base / ("run_%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), stage))
    index = 1
    while destination.exists():
        destination = base / ("run_%s_%s_r%d" % (time.strftime("%Y%m%d_%H%M%S"), stage, index))
        index += 1
    destination.mkdir(parents=True)
    result = dict(result, stage=stage, run_dir=str(destination),
                  operator={"executable": sys.executable, "argv": sys.argv})
    target = destination / ("main_check_%s.json" % stage)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main():
    # Windows重定向输出默认GBK，UTF-8确保中文及物理公式可完整写入日志。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="check",
                        choices=["check", "optics", "data", "smoke", "train", "evaluate", "report"])
    parser.add_argument("--stage", default=None,
                        help="M1：标记本次check的阶段（安装前/安装后），结果写入新M1 run目录")
    parser.add_argument("--config", help="M2/M3有效配置JSON")
    parser.add_argument("--resume", help="M2/M3已有运行目录；续接写入新目录")
    parser.add_argument("--m2-run", help="check额外审核指定M2运行目录")
    parser.add_argument("--m3-run", help="check额外审核指定M3运行目录")
    parser.add_argument("--phase", choices=["all","download","metadata","split","preprocess","index"], help="M3独立阶段；默认all")
    args, remaining = parser.parse_known_args()
    try:
        if args.command not in {"check", "report", "optics", "data"}:
            raise ValueError("该命令尚未实现；请按tasks执行对应里程碑：" + args.command)
        if remaining:
            raise ValueError("未知参数：" + " ".join(remaining))
        if args.command == "optics":
            if not args.config or args.m2_run or args.m3_run or args.stage or args.phase:
                raise ValueError("optics需要--config，不能搭配--m2-run/--stage")
            from m2_optics import run
            result = run(args.config, args.resume)
        elif args.command == "data":
            if not args.config or args.m2_run or args.m3_run or args.stage:
                raise ValueError("data需要--config，阶段用--phase指定")
            from m3_pipeline import run
            result = run(args.config, args.phase or "all", args.resume)
        else:
            if args.config or args.resume or args.phase:
                raise ValueError("--config/--resume/--phase仅供optics/data使用")
            result = check(args.stage) if args.command == "check" else report()
            if args.m2_run and args.command != "check":
                raise ValueError("--m2-run仅供check使用")
            m2_root = ROOT / "outputs/jeon2019_optics/results/reproduction_v1/m2"
            candidates = sorted(p.parent for p in m2_root.glob("run_*/audit.json"))
            selected = args.m2_run or (str(candidates[-1]) if candidates else None)
            if selected:
                from m2_optics import check_run
                m3_config_path = HERE / "config_m3.json"
                if m3_config_path.exists() and Path(selected).resolve() == safe_path(read_json(m3_config_path)["m2_run"]):
                    from m3_common import verify_parent
                    config = read_json(m3_config_path)
                    verify_parent(Path(selected), safe_path(config["parent_bridge"]))
                    result["M2"] = {"status":"passed", "run_dir":str(Path(selected).resolve()), "gain_pending":True,
                                    "entry_source_bridge":config["parent_bridge"], "paper_alignment_passed":False}
                else:
                    result["M2"] = check_run(selected)
                if args.command == "check":
                    result["milestones"]["M2"] = result["M2"]["status"]
                    result["scope"] += "；M2最新或指定物理算子包指纹与合同"
                if result["M2"]["status"] != "passed":result["status"]="failed"
            if args.m3_run and args.command != "check":raise ValueError("--m3-run仅供check使用")
            m3_root = ROOT / "outputs/jeon2019_optics/results/reproduction_v1/m3"
            m3_candidates = sorted(p.parent for p in m3_root.glob("run_*/audit.json"))
            m3_selected = args.m3_run or (str(m3_candidates[-1]) if m3_candidates else None)
            if m3_selected:
                from m3_pipeline import check_run as check_m3
                result["M3"] = check_m3(m3_selected)
                if args.command == "check":
                    result["milestones"]["M3"] = result["M3"]["status"]
                    result["scope"] += "；M3数据/父物理包/场景及增益状态"
                if result["M3"]["status"] != "passed" and result.get("status") != "failed":result["status"]=result["M3"]["status"]
        if args.command == "check":
            try:
                print("check结果已保存：" + str(persist_check(result, args.stage or "unlabelled")))
            except OSError as exc:
                print("check结果保存失败（不影响检查结论）：" + repr(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in {None, "passed"} else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "paper_alignment_passed": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
