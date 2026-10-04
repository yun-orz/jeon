# -*- coding: utf-8 -*-
"""S6 验证：配置里的 show/save 开关是否**真的**接入运行逻辑。

为什么需要它：第二轮审核指出 ``runtime.show_plots`` 与 ``runtime.save_results`` 都没有
参与任何决定，README 却写着“改配置就能弹窗”。这里用 **mock 后端选择** 与
**临时配置**在自动化测试里验证接线，避免测试阻塞在真实图窗上。

做法：把 ``main`` 作为模块导入，替换 ``main.select_backend`` 为记录调用的桩函数，
再直接调用 ``main.resolve_run_settings`` 与 ``main.main()``，比较实际行为。

运行::

    python -B docs/verify_config_switches.py [--report <目录>]
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "config.json"


def project_files() -> set:
    """项目输出目录的**轻量**快照：运行目录名 + 各运行目录下的子目录名。

    故意**不**遍历 ``arrays/*.npz``：单个运行目录可能有上亿字节的数组文件，
    逐个 stat 会让本脚本从几秒变成几分钟（实测踩过）。只要"有没有新增/消失的运行
    目录及子目录"即可核对“不落盘”这一断言。
    """
    root = PROJECT_ROOT / "results"
    if not root.exists():
        return set()
    # 检查文件及元信息，也能发现错误覆盖；不读取大型数组内容。
    return {(str(p.relative_to(PROJECT_ROOT)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}
    out = set()
    for stage_dir in root.iterdir():
        if not stage_dir.is_dir():
            out.add(str(stage_dir.relative_to(PROJECT_ROOT)))
            continue
        for run in stage_dir.iterdir():
            rel = str(run.relative_to(PROJECT_ROOT))
            out.add(rel)
            if run.is_dir():
                for sub in run.iterdir():
                    if sub.is_dir():
                        out.add(str(sub.relative_to(PROJECT_ROOT)))
    return out


def fresh_main(argv: List[str], backend_calls: Optional[List[bool]] = None):
    """以给定 argv 重新导入 main，并可把后端选择替换成记录桩。"""
    if "main" in sys.modules:
        del sys.modules["main"]
    old_argv = sys.argv[:]
    sys.argv = ["main.py"] + list(argv)
    try:
        if backend_calls is not None:
            # 在导入前替换后端选择，避免测试真实弹窗或卡在GUI事件循环。
            import ast
            import types
            import matplotlib
            source = PROJECT_ROOT / "main.py"
            tree = ast.parse(source.read_text(encoding="utf-8-sig"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "select_backend":
                    node.body = ast.parse("return __backend_stub__(show_plots)").body
            def stub(show_plots):
                backend_calls.append(bool(show_plots))
                matplotlib.use("Agg", force=True)
                return "MOCK:Agg(show=%s)" % bool(show_plots)
            mod = types.ModuleType("main")
            mod.__file__ = str(source)
            mod.__dict__["__backend_stub__"] = stub
            sys.modules["main"] = mod
            exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), mod.__dict__)
            mod.plt.show = lambda *a, **k: None
            return mod
        mod = importlib.import_module("main")
    finally:
        sys.argv = old_argv
    if backend_calls is not None:
        def stub(show_plots):
            backend_calls.append(bool(show_plots))
            return "MOCK(show=%s)" % bool(show_plots)
        mod.select_backend = stub
        mod.BACKEND = "MOCK"
    return mod


def run_case(name: str, cfg_overrides: Dict[str, Any], argv_extra: List[str],
             tmp: Path, report: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把配置副本写到临时目录，运行 main()，记录实际行为。"""
    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["runtime"].update(cfg_overrides)
    cfg_path = tmp / ("cfg_%s.json" % name)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    calls: List[bool] = []
    before = project_files()
    out_dir = tmp / ("run_" + name)
    # 用 --only direct + --light：只验证**开关接线**，不必跑完整四组（物理正确性由
    # 默认完整运行与单元测试负责）。这样本脚本几秒钟即可完成。
    argv = (["--light", "--only", "direct", "--config", str(cfg_path),
             "--run-dir", str(out_dir)] + argv_extra)
    # 先按 argv 建模块以取得 SETTINGS，再用同一个 argv 建一次带桩的模块执行
    mod = fresh_main(argv, backend_calls=calls)
    settings = dict(mod.SETTINGS)
    buf = io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = mod.main()
    finally:
        pass
    after = project_files()
    rec = {
        "case": name,
        "config_runtime": cfg_overrides,
        "argv_extra": argv_extra,
        "effective_show_plots": settings["show_plots"],
        "effective_save_results": settings["save_results"],
        "show_source": settings["show_source"],
        "save_source": settings["save_source"],
        "backend_called_with_show": calls,
        "run_dir_exists": out_dir.exists(),
        "project_output_added": sorted(after - before),
        "project_output_removed": sorted(before - after),
        "exit_code": code,
    }
    report.append(rec)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description="S6：配置 show/save 开关接线验证（mock 后端）")
    p.add_argument("--report", type=str, default=None, help="报告目录")
    args = p.parse_args()
    report_dir = Path(args.report) if args.report else (
        PROJECT_ROOT / "results" / "stage01" / "_config_switch_check" /
        ("run_" + __import__("uuid").uuid4().hex))
    report_dir.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="cases_", dir=str(report_dir)))
    report: List[Dict[str, Any]] = []
    try:
        run_case("default_config_off", {"show_plots": False, "save_results": True},
                 [], tmp, report)
        run_case("config_show_true", {"show_plots": True, "save_results": True},
                 [], tmp, report)
        run_case("config_save_false", {"show_plots": False, "save_results": False},
                 [], tmp, report)
        run_case("cli_show_overrides_config_off", {"show_plots": False, "save_results": True},
                 ["--show-plots"], tmp, report)
        run_case("cli_no_save_overrides_config_on", {"show_plots": False, "save_results": True},
                 ["--no-save"], tmp, report)
    finally:
        # 用户要求删除前确认：这里保留配置、副本与运行证据。
        pass

    # 断言式检查
    checks = []
    by = {r["case"]: r for r in report}

    def check(desc: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": desc, "ok": bool(ok), "detail": detail})

    r = by["default_config_off"]
    check("默认配置：不弹窗、落盘", (not r["effective_show_plots"]) and r["effective_save_results"]
          and r["run_dir_exists"], json.dumps(r, ensure_ascii=False)[:200])
    check("默认配置：后端未被要求 show", r["backend_called_with_show"] in ([], [False]),
          str(r["backend_called_with_show"]))

    r = by["config_show_true"]
    check("配置 show_plots=true 真的生效（后端收到 show=True）",
          r["effective_show_plots"] and r["backend_called_with_show"] == [True],
          str(r["backend_called_with_show"]))

    r = by["config_save_false"]
    check("配置 save_results=false 真的不落盘",
          (not r["effective_save_results"]) and (not r["run_dir_exists"])
          and not r["project_output_added"],
          "added=%r" % (r["project_output_added"],))

    r = by["cli_show_overrides_config_off"]
    check("CLI --show-plots 覆盖配置 false",
          r["effective_show_plots"] and r["backend_called_with_show"] == [True],
          str(r["backend_called_with_show"]))

    r = by["cli_no_save_overrides_config_on"]
    check("CLI --no-save 覆盖配置 true 且无项目输出",
          (not r["effective_save_results"]) and (not r["run_dir_exists"])
          and not r["project_output_added"],
          "added=%r" % (r["project_output_added"],))

    ok = all(c["ok"] for c in checks)
    summary = {"checks": checks, "all_ok": ok, "records": report}
    (report_dir / "config_switch_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in checks:
        print("%-55s %s %s" % (c["check"], "OK" if c["ok"] else "FAIL", c["detail"]))
    print("ALL_OK=%s  报告：%s" % (ok, report_dir / "config_switch_check.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
