# -*- coding: utf-8 -*-
"""记录M1下载产物的最终去向：区分"我执行的删除"与"并发会话的外部删除"。

背景：M1验收完成后（用户已明确授权），我删除了33个分块、prefix.bin副本与旧部分下载
原件，共35个文件3.534GiB，并**保留**了已核验整包wheel、sympy wheel与全部清单/审核文件。

随后（我这一侧的记录之外）另一个并发会话——本机同时在执行M2/M3——把整个
work/dependencies/gpu_cu121_resume_m1_20261007 与 work/dependencies/gpu_cu121_20261006
目录都删除了，其中包括我特意保留的整包wheel与sympy wheel、以及下载清单、probe、
续传状态与安装日志。这不是我执行的删除，也不属于本阶段授权范围。

本文件如实登记该差异，并给出当前真实状态与恢复方式；不修改任何原始核验数值。
"""
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESUME = ROOT / "work/dependencies/gpu_cu121_resume_m1_20261007"
OLD_DIR = ROOT / "work/dependencies/gpu_cu121_20261006"
AUDIT_DIR = ROOT / "outputs/执行_M1_GPU环境_20261007"
ENVIRONMENT = ROOT / "work/environments/jeon_gpu_cu121"
TORCH_URL = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl"


def main():
    record = json.loads((AUDIT_DIR / "删除记录.json").read_text(encoding="utf-8"))
    missing_from_kept = [path for path in record["kept"] if not Path(path).exists()]
    exists = {"resume_workdir": RESUME.exists(), "old_partial_dir": OLD_DIR.exists(),
              "gpu_environment_python": (ENVIRONMENT / "Scripts/python.exe").is_file(),
              "torch_in_environment": (ENVIRONMENT / "lib/site-packages/torch/__init__.py").is_file()}
    payload = {
        "schema": "jeon2019-m1-cleanup-final-state-v1", "milestone": "M1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "my_authorized_deletion": {
            "count": record["deleted_count"], "freed_bytes": record["freed_bytes"],
            "freed_gib": record["freed_gib"], "deleted_at": record["deleted_at"],
            "scope": "33个分块part_*.bin + prefix.bin副本 + 旧部分下载原件",
            "source": str(AUDIT_DIR / "删除记录.json"),
            "evidence": "删除前逐个核对路径、文件名模式与续传清单记录的块长度，并记录每个文件的SHA256与字节数"},
        "external_deletion_by_concurrent_session": {
            "detected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "what": ["work/dependencies/gpu_cu121_resume_m1_20261007（整个工作目录）",
                     "work/dependencies/gpu_cu121_20261006（旧部分下载目录）"],
            "previously_kept_but_now_gone": missing_from_kept,
            "attribution": "不是我执行的删除，也不在本阶段授权范围内；本机同时有另一会话在执行M2/M3（work目录存在m2_*/m3_*日志，且work/datasets/reproduction_v1下有M3数据写入）。",
            "not_verified": "我无法从本会话确认该删除的具体发起者与理由，仅登记可观察事实（目录消失、D盘可用空间由93.37GiB升至121.39GiB）。",
            "impact": {
                "m1_conclusions": "不受影响：M1判定依据是运行时的核验结果与已保存审核文件，不依赖这些二进制继续存在",
                "local_offline_reinstall": "已不可用：整包wheel被外部删除后，重装Torch需要重新联网下载约2.3GiB",
                "gpu_execution": "不受影响：GPU运行只依赖虚拟环境内已安装的包"},
        },
        "current_state": exists,
        "d_free_bytes": shutil.disk_usage(str(ROOT)).free,
        "preserved_m1_evidence": sorted(p.name for p in AUDIT_DIR.iterdir() if p.is_file()),
        "restore_commands": [
            "python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py download",
            "python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py install"],
        "note_on_manifests": "download_manifest.json / probe.json / resume_state.json / install_result.json / install_log.txt 已随工作目录被外部删除，因此无法再把删除标注写入它们；这些文件的核验数值已固化在 续传清单.json 与 下载SHA.json 中。",
        "recovery_source": {"url": TORCH_URL,
                            "official_sha256": "9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f",
                            "frozen_total_bytes": 2449372784},
        "correction": "报告第13节原先写有'整包wheel保留，仍可免联网重装'；该结论在外部删除后不再成立，已同步修订。",
    }
    target = AUDIT_DIR / "空间清理说明.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入", target)
    print("外部删除导致原先保留的文件丢失：", len(missing_from_kept), "个")
    for item in missing_from_kept:
        print("  -", item)
    print("GPU环境可用：", exists["gpu_environment_python"], "torch在环境内：", exists["torch_in_environment"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
