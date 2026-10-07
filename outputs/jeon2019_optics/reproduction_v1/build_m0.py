"""一次性建立M0证据；既有冻结索引存在时拒绝覆盖，任何失败产物均保留。"""
import json
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone

from audit import HERE, ROOT, RESULTS, capture_protection, g0_check, protocol_check, read_json, safe_path, sha256, verify_manifest

STAGES = [
    ("G1", "g1/run_20261005_231424", "执行_G1_20261005/最终审核.json", "原CCW算子；原论文方向未对齐"),
    ("G1_CW", "g1_alignment/run_20261005_234023", "执行_G1对齐_20261005/independent_audit.json", "CW25波段；细输入仅420/540/660nm"),
    ("G2", "g2/run_20261005_234941", "执行_G2_20261005/最终审核.json", "原CCW核及人为Gaussian响应；不能混入新正式算子"),
    ("G3", "g3/run_20261006_000710", "执行_G3_20261006/最终审核.json", "式21结构/梯度已核对；作者精确层表未知"),
    ("G4", "g4/run_20261006_074407", "执行_G4_20261006/最终审核.json", "小规模真数据；中央32监督；不是论文规模"),
    ("G4_expand", "g4_expand/run_20261006_090723", "执行_G4扩充_20261006/最终审核.json", "16训练上下文/128更新；热启动不是完整恢复"),
    ("G5", "g5/run_20261006_080431", "执行_G5_20261006/最终审核.json", "已审核指标；作者SAM单位及实现未知"),
    ("G5_test", "g5_test/run_20261006_081215", "执行_G5独立测试_20261006/最终审核.json", "img5单场景4patch，已查看，只作诊断"),
    ("G5_retest", "g5_retest/run_20261006_093137", "执行_G5固定复测_20261006/最终审核.json", "固定img5复测；非新封存测试"),
    ("G5_boundary", "g5_boundary/run_20261006_130532", "执行_G5边界诊断_20261006/最终审核.json", "三个区域上下文效应小；不推广整图"),
    ("G1_window", "g1_window/run_20261006_131121", "执行_G1窗口补核_20261006/最终审核.json", "九波长97/161窗口；输入细采样未重做"),
    ("G1_lightpipes", "g1_lightpipes/run_20261006_131933", "执行_G1LightPipes_20261006/最终审核.json", "执行审核通过但算法一致性未通过"),
    ("G1_lp_sampling", "g1_lp_sampling/run_20261006_133000", "执行_G1LP采样拆分_20261006/最终审核.json", "采样模型已核对，上一填充对660nm未收敛"),
    ("G1_padding", "g1_padding/run_20261006_133832", "执行_G1填充5120_20261006/最终审核.json", "三个控制波长有限填充对通过；非无限域证明")
]


def write_json(path, value):
    """排他创建，不覆盖任何既有证据。"""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def stage_index():
    stages, failures = [], []
    for name, run, audit_path, limitation in STAGES:
        run_name = "outputs/jeon2019_optics/results/" + run
        audit_name = "outputs/" + audit_path
        evidence = read_json(safe_path(audit_name))
        if name == "G1":
            passed = evidence.get("numerical_audit_passed") is True and evidence.get("independent_complex_points_passed") is True
        elif name == "G1_CW":
            passed = evidence.get("passed") is True and evidence.get("source_unchanged") is True
        else:
            passed = evidence.get("independent_audit", {}).get("passed") is True
        declared_run = evidence.get("official_run", evidence.get("formal_run", evidence.get("run")))
        # 历史审核使用绝对Windows路径或反斜线相对路径，转换后只在项目内读取。
        declared = str(declared_run).replace("\\", "/")
        prefix = ROOT.as_posix() + "/"
        if declared.lower().startswith(prefix.lower()):
            declared = declared[len(prefix):]
        path = safe_path(run_name)
        exists = path.is_dir()
        failed_marker = any(path.glob("*failed*.json")) if exists else True
        source_matches = declared == run_name
        record = {"stage": name, "source_run": run_name, "audit": audit_name,
                  "audit_sha256": sha256(safe_path(audit_name)), "historical_audit_passed": passed,
                  "declared_source_matches": source_matches, "run_present": exists,
                  "failed_marker_present": failed_marker,
                  "status": evidence.get("status", "historical_numerical_audit_passed" if passed else "failed"),
                  "paper_alignment_passed": False, "limitation": limitation}
        stages.append(record)
        if not (passed and exists and source_matches and not failed_marker):
            failures.append(record)
    gpu_name = "outputs/执行_GPU验证_20261006/cpu_preflight.json"
    gpu = read_json(safe_path(gpu_name))
    gpu_pending_correct = (gpu.get("cpu_preflight_passed") is True
                           and gpu.get("gpu_validation_passed") is False
                           and gpu.get("gpu_execution_verified") is False)
    stages.append({"stage": "G3_GPU", "audit": gpu_name, "audit_sha256": sha256(safe_path(gpu_name)),
                   "status": "cpu_preflight_only_gpu_installation_pending", "historical_cpu_preflight_passed": True,
                   "gpu_validation_passed": False, "paper_alignment_passed": False,
                   "limitation": "没有GPU最终审核，M1必须实际完成"})
    if not gpu_pending_correct:
        failures.append({"stage": "G3_GPU", "error": "GPU预审来源状态与登记不符"})
    return {"schema": "jeon2019-historical-stages-v1", "passed": not failures,
            "stages": stages, "failures": failures,
            "forbidden_mixing": ["旧CCW/Gaussian/光子域与新CW/实测相对响应/能量域不得混用",
                                 "热启动权重不是Adam恢复状态", "历史img5测试/复测不是新的10场景封存测试",
                                 "独立执行审核通过不等于所有算法一致性或论文对齐通过"]}


def main():
    index_path = HERE / "m0_index.json"
    if index_path.exists():
        print("M0已冻结；拒绝覆盖。请用main.py check核对；新协议另建版本。")
        return 2
    started = time.perf_counter()
    now = datetime.now(timezone(timedelta(hours=8)))
    run = RESULTS / "m0" / ("run_" + now.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    commands = [f'"{sys.executable}" -B "{HERE / "build_m0.py"}"',
                f'"{sys.executable}" -B "{HERE / "tests/test_m0.py"}"']
    try:
        tests = subprocess.run([sys.executable, "-B", str(HERE / "tests/test_m0.py")],
                               cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
        test_report = {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr}
        write_json(run / "tests.json", test_report)
        if tests.returncode:
            raise ValueError("M0检查器测试失败")
        protocol = protocol_check()
        stages = stage_index()
        g0_before = g0_check()
        if not (protocol["passed"] and stages["passed"] and g0_before["passed"]):
            write_json(run / "preflight_failed.json", {"protocol": protocol, "stages": stages, "g0": g0_before})
            raise ValueError("协议/历史来源/G0预检未通过")
        print("开始对历史文件逐一建立SHA保护清单；不修改来源。", flush=True)
        protection = capture_protection()
        write_json(run / "protection.json", protection)
        write_json(run / "historical_stages.json", stages)
        protected = verify_manifest(protection)
        g0_after = g0_check()
        if not (protected["passed"] and g0_after["passed"]):
            write_json(run / "protection_failed.json", {"protection": protected, "g0": g0_after})
            raise ValueError("历史来源保护核对未通过")
        # 只追加字节，保留状态文档原始换行和全部历史内容。
        summary = ("\n\n## 正式仿真基线 M0：协议与Goal任务包冻结（" + now.strftime("%Y-%m-%d") + "）\n\n"
                   "已建立 outputs/jeon2019_optics/reproduction_v1；冻结参数证据、来源、三类交接对象和M0–M7八份任务。"
                   "历史正式来源及失败限制登记完成，GPU仍只有CPU预审；新共同测量增益待训练集拟合。\n\n"
                   f"历史保护覆盖{protected['checked']}个文件，状态文档原始字节前缀只读；G0全部{g0_after['checked']}项通过。"
                   "本阶段未下载大包、未更换旧算子、未训练或读取新测试；paper_alignment_passed=false。\n\n"
                   "实际验收和恢复/交接入口见 " + (run / "report.md").relative_to(ROOT).as_posix()
                   + "，索引见 outputs/jeon2019_optics/reproduction_v1/m0_index.json。M1–M7尚未完成；无删除操作。\n")
        with (ROOT / "CURRENT_STATUS.md").open("ab") as stream:
            stream.write(summary.encode("utf-8"))
        prefix_item = next(item for item in protection["files"] if item["path"] == "CURRENT_STATUS.md")
        prefix_ok = sha256(ROOT / "CURRENT_STATUS.md", prefix_item["bytes"]) == prefix_item["sha256"]
        if not prefix_ok:
            raise ValueError("状态文档追加改变了历史前缀")
        fingerprint_paths = [HERE / name for name in ["protocol.json", "sources.json", "parameter_evidence.json", "contracts.json"]]
        fingerprint_paths += sorted((HERE / "tasks").glob("*.md"))
        fingerprint_paths += sorted((HERE / "docs").glob("*.md"))
        fingerprints = {p.relative_to(ROOT).as_posix(): sha256(p) for p in fingerprint_paths}
        audit = {"schema": "jeon2019-m0-audit-v1", "milestone": "M0", "status": "passed",
                 "created_at": now.isoformat(), "commands": commands, "effective_config": "protocol.json",
                 "input_fingerprints": fingerprints,
                 "output_fingerprints": {"protection.json": sha256(run / "protection.json"),
                                         "historical_stages.json": sha256(run / "historical_stages.json")},
                 "key_numbers": {"protected_files": protected["checked"], "protected_bytes": protected["checked_bytes"],
                                 "parameters": protocol["parameters"], "tasks": protocol["tasks"]},
                 "failures": [{"phase": "development_first_unit_run", "resolved": True,
                               "error": "M4/M5任务缺明确输入段，已补齐；门槛保持",
                               "evidence": "docs/validation_history.md"}],
                 "code_fingerprints": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(HERE.rglob("*.py"))},
                 "acceptance": {"all_passed": True, "protocol": protocol,
                                               "historical_sources": stages["passed"], "protection": protected,
                                               "g0_before": g0_before, "g0_after": g0_after,
                                               "status_historical_prefix_unchanged": prefix_ok, "unit_tests": test_report},
                 "resume_command": f'"{sys.executable}" -B "{HERE / "main.py"}" check',
                 "next_inputs": ["tasks/M1_gpu_environment.md", "tasks/M2_freeze_optics.md"],
                 "pending": ["真实GPU验收", "CW全25波段细采样/实测替代响应", "多数据集空间预检和封存", "恢复训练器", "预演", "正式配对训练", "最终测试"],
                 "elapsed_seconds": time.perf_counter() - started, "paper_alignment_passed": False}
        write_json(run / "audit.json", audit)
        report = ("# M0实际交付报告\n\nM0通过；M1–M7尚未实施。程序核对通过不等于论文对齐。\n\n"
                  f"创建时间：{now.isoformat()}。保护{protected['checked']}个历史文件、{protected['checked_bytes']}字节；"
                  f"G0全量{g0_after['checked']}项通过；{protocol['parameters']}项参数证据、8份Goal任务。\n\n"
                  "历史核对逐一比较SHA与字节长度；CURRENT_STATUS.md只追加，原始前缀SHA不变。"
                  "历史GPU仍待安装；现有Torch为CPU版本。没有下载大包、训练、测试开封或删除文件。\n\n"
                  "实际命令：\n\n```text\n" + "\n".join(commands) + "\n```\n\n"
                  "有效协议：入口目录protocol.json。输入/输出指纹、关键数值、测试输出和验收详情见audit.json；"
                  "历史来源索引见historical_stages.json；保护清单见protection.json。首次任务完整性检查失败已修复，"
                  "过程记录见入口docs/validation_history.md；最终未解决失败为空。后续空间需求需各阶段预检。\n\n"
                  "继续入口：main.py无参数或check进行只读核对，report查看交接。M0已冻结时build_m0.py拒绝覆盖。"
                  "下一阶段为tasks/M1_gpu_environment.md与tasks/M2_freeze_optics.md；本工作包完成后停止。\n\n"
                  "与论文未对齐项：作者响应、精确训练/测试名单、网络层表、2019传播脚本、SAM单位。"
                  "替代响应、CW坐标、连续高度、Fresnel550nm、单位、边界、数据划分和40/60轮为明确实施假设。"
                  "共享增益仍待训练拟合，不占位宣称已完成。paper_alignment_passed=false。\n")
        with (run / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report)
        artifact_paths = [run / name for name in ["protection.json", "historical_stages.json", "audit.json", "report.md", "tests.json"]]
        artifact_paths += fingerprint_paths
        index = {"schema": "jeon2019-m0-index-v1", "created_at": now.isoformat(),
                 "protection": (run / "protection.json").relative_to(ROOT).as_posix(),
                 "stages": (run / "historical_stages.json").relative_to(ROOT).as_posix(),
                 "audit": (run / "audit.json").relative_to(ROOT).as_posix(),
                 "report": (run / "report.md").relative_to(ROOT).as_posix(),
                 "artifacts": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in artifact_paths]}
        write_json(index_path, index)
        print(json.dumps({"status": "passed", "milestone": "M0", "report": str(run / "report.md"),
                          "protected_files": protected["checked"], "paper_alignment_passed": False}, ensure_ascii=False))
        return 0
    except Exception as exc:
        write_json(run / "failed.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                                          "elapsed_seconds": time.perf_counter() - started, "paper_alignment_passed": False})
        print("M0失败，证据保留于" + str(run) + "：" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
