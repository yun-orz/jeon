"""M0 的路径、历史保护和协议检查；所有核对均只读。"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "outputs/jeon2019_optics/results/reproduction_v1"
SCHEMA = "jeon2019-protection-v1"
IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path, limit=None):
    """按块计算SHA；limit用于状态文档的历史前缀保护。"""
    digest = hashlib.sha256()
    remaining = limit
    with Path(path).open("rb") as stream:
        while remaining is None or remaining > 0:
            block = stream.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def safe_path(name, root=ROOT):
    """只接受规范项目相对POSIX路径，拒绝别名、越界和符号链接逃逸。"""
    if not isinstance(name, str) or not name:
        raise ValueError("保护路径必须是非空字符串")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
            or relative.as_posix() != name or name == "."):
        raise ValueError("非法相对路径：" + name)
    base = Path(root).resolve()
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        raise ValueError("路径越出项目：" + name)
    return target


def validate_records(manifest, root=ROOT):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("未知保护清单格式")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("保护清单不能为空")
    seen = set()
    for item in records:
        name = item["path"]
        canonical = os.path.normcase(str(safe_path(name, root)))
        if canonical in seen:
            raise ValueError("重复保护路径：" + name)
        seen.add(canonical)
        digest = item["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("非法SHA：" + name)
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ValueError("非法文件长度：" + name)
        if item.get("mode") not in {"immutable", "append_only"}:
            raise ValueError("未知保护方式：" + name)
        if item["mode"] == "append_only" and name != "CURRENT_STATUS.md":
            raise ValueError("仅状态文档允许追加：" + name)
    return records


def verify_manifest(manifest, root=ROOT):
    records = validate_records(manifest, root)
    failures, checked_bytes = [], 0
    for item in records:
        name = item["path"]
        path = safe_path(name, root)
        if not path.is_file():
            failures.append({"path": name, "error": "缺少文件"})
            continue
        count = item["bytes"]
        actual = path.stat().st_size
        length_ok = actual >= count if item["mode"] == "append_only" else actual == count
        if not length_ok:
            failures.append({"path": name, "error": "字节长度不符合保护方式"})
            continue
        limit = count if item["mode"] == "append_only" else None
        if sha256(path, limit) != item["sha256"]:
            failures.append({"path": name, "error": "历史内容SHA变化"})
        checked_bytes += count
    return {"passed": not failures, "checked": len(records), "checked_bytes": checked_bytes,
            "failures": failures, "read_only": True}


def historical_files():
    """保护历史产物，不把新流程或可变GPU安装目录纳入冻结范围。"""
    roots = [ROOT / "outputs", ROOT / "baseline", ROOT / "work/papers",
             ROOT / "work/datasets", ROOT / "work/dependencies/lightpipes_2_1_5"]
    paths = {p.resolve() for p in ROOT.iterdir() if p.is_file()}
    resume_script = ROOT / "work/resume_gpu_cu121.ps1"
    if resume_script.exists():
        paths.add(resume_script.resolve())
    for top in roots:
        if not top.exists():
            continue
        for directory, folders, files in os.walk(top, followlinks=False):
            base = Path(directory).resolve()
            folders[:] = sorted(name for name in folders if name not in IGNORED_DIRS
                                and not (base / name).resolve().is_relative_to(HERE)
                                and not (base / name).resolve().is_relative_to(RESULTS)
                                and not (base / name).resolve().is_relative_to(ROOT / "work/datasets/reproduction_v1"))
            for name in files:
                path = base / name
                if not path.is_symlink() and path.suffix not in {".pyc", ".pyo"}:
                    paths.add(path.resolve())
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def capture_protection():
    records = []
    for path in historical_files():
        name = path.relative_to(ROOT).as_posix()
        count = path.stat().st_size
        records.append({"path": name, "bytes": count, "sha256": sha256(path),
                        "mode": "append_only" if name == "CURRENT_STATUS.md" else "immutable"})
    return {"schema": SCHEMA, "files": records,
            "excluded": ["新reproduction_v1代码/数据/结果", "解释器缓存", "GPU虚拟环境及安装包/续传工作目录", ".git/.idea/.venv", "work中其他恢复演练副本"],
            "note": "除CURRENT_STATUS.md原始字节前缀外，清单内历史文件按完整SHA保护；排除不表示可以删除。"}


def g0_check():
    """直接调用冻结的只读G0核对函数，避免改写其实现。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("jeon_g0_readonly", ROOT / "main_g0_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_manifest(read_json(ROOT / "baseline/g0_20261005/manifest.json"), root=ROOT)


def protocol_check():
    protocol = read_json(HERE / "protocol.json")
    evidence = read_json(HERE / "parameter_evidence.json")
    sources = read_json(HERE / "sources.json")
    contracts = read_json(HERE / "contracts.json")
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(protocol.get("schema") == "jeon2019-reproduction-protocol-v1", "协议schema错误")
    require(protocol.get("paper_alignment_passed") is False, "不能宣称论文已对齐")
    require(protocol["optics"]["wavelengths_nm"] == list(range(420, 661, 10)), "25波长被改变")
    require(protocol["optics"]["normalize_each_band_to_one"] is False, "禁止逐波段核归一化")
    require(protocol["optics"]["automatic_lambda_over_540"] is False, "禁止自动光子换算")
    require(protocol["execution"]["amp"] is False and protocol["execution"]["tf32"] is False, "AMP/TF32必须关闭")
    require(protocol["execution"]["delete_without_explicit_permission"] is False, "删除保护失效")
    require(protocol["execution"]["max_hours_per_work_package"] == 8, "单工作包预算改变")
    require(protocol["training"]["effective_batch_size"] == 16, "有效batch必须16")
    require(protocol["training"]["updates_per_epoch"] * 16 == protocol["data"]["patches_per_epoch"], "更新/样本计数不符")
    require(protocol["training"]["updates_at_40_epochs"] == 75000, "40轮更新计数不符")
    require(protocol["evaluation"]["halo"] == 128, "最终halo改变")
    require(protocol["evaluation"]["paper_reference"]["is_acceptance_threshold"] is False, "文献成绩不能成为开发通过线")
    for key, value in protocol["paths"].items():
        safe_path(value)
        require(value == {"code": "outputs/jeon2019_optics/reproduction_v1", "data": "work/datasets/reproduction_v1",
                          "results": "outputs/jeon2019_optics/results/reproduction_v1"}[key], "写入根目录不符：" + key)
    states = {"paper_confirmed", "implementation_verified", "implementation_assumption", "unknown"}
    ids = set()
    for item in evidence["parameters"]:
        require(item["id"] not in ids, "重复参数ID：" + item["id"])
        ids.add(item["id"])
        require(item["state"] in states and bool(item.get("source")), "参数状态/来源缺失：" + item["id"])
        require(item["state"] == "unknown" or item.get("value") is not None, "非未知参数缺默认值：" + item["id"])
    require({"author_response", "author_scene_lists", "exact_network_table", "metrics", "noise_placement", "budget"}.issubset(ids), "关键差异未登记")
    tasks = sorted((HERE / "tasks").glob("M[0-7]_*.md"))
    require(len(tasks) == 8 and {p.name[:2] for p in tasks} == {"M" + str(i) for i in range(8)}, "必须恰好8份里程碑任务")
    for task in tasks:
        body = task.read_text(encoding="utf-8")
        require(all(word in body for word in ["输入", "输出", "停止", "成功"]), "任务缺交接/停止条件：" + task.name)
    for name in ["operator_package", "data_manifest", "checkpoint", "milestone_report"]:
        require(bool(contracts.get(name, {}).get("required")), "交接字段缺失：" + name)
    for source in sources["sources"]:
        if source.get("path"):
            path = safe_path(source["path"])
            require(path.is_file(), "缺少来源文件：" + source["path"])
            if source.get("sha256") and path.is_file():
                require(sha256(path) == source["sha256"], "论文来源SHA不符：" + source["id"])
    return {"passed": not failures, "parameters": len(ids), "tasks": len(tasks), "failures": failures}
