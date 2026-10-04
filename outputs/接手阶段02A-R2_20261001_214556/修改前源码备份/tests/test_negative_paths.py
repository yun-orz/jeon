# -*- coding: utf-8 -*-
"""第二轮修正的**负面测试与状态语义**验证（S4.3）。

本模块**不修改生产源码**，而是按任务书允许的方式工作：

* 在测试进程内**注入返回数据**（monkeypatch 某个 study 函数的返回值）来制造失败；
* 或把一份仅供失败测试使用的配置写到**临时目录**，其中只改阈值，
  **不把该配置作为默认配置提交**；
* 每个负面测试都记录：命令、预期退出码、实际退出码、逐项状态、日志路径。

运行::

    python -B -m unittest discover -s tests -v
    python -B -m pytest tests -v

或者单独运行本模块以生成完整的中文证据报告::

    python -B tests/test_negative_paths.py --report <输出目录>
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "results" / "stage01" / "_negative_tests" / ("run_" + __import__("uuid").uuid4().hex)


# ------------------------------------------------------------------ 工具
def fresh_main():
    """重新导入 main，确保模块级的参数解析与设置解析只发生一次。"""
    for name in ("main",):
        if name in sys.modules:
            del sys.modules[name]
    return importlib.import_module("main")


def run_main(argv: List[str], monkeypatch=None) -> Dict[str, Any]:
    """在进程内以给定 argv 运行 ``main.main()``，返回退出码与产出目录。

    ``monkeypatch`` 是一个 ``callable(module)``，在 ``main()`` 被调用前执行，
    用于注入返回数据（模拟失败或非有限值）。**不修改磁盘上的生产源码。**
    """
    old_argv = sys.argv[:]
    sys.argv = ["main.py"] + list(argv)
    mod = fresh_main()
    if monkeypatch is not None:
        monkeypatch(mod)
    buf = io.StringIO()
    code = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = mod.main()
    finally:
        sys.argv = old_argv
    return {"exit_code": code, "log": buf.getvalue(), "module": mod}


def base_argv(out_dir: Path, extra: Optional[List[str]] = None) -> List[str]:
    """负面测试统一使用 ``--light``（缩小 FFT 规模，不改变任何阈值与物理定义）。"""
    argv = ["--light", "--run-dir", str(out_dir)]
    if extra:
        argv += list(extra)
    return argv


def read_overall(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "acceptance_check.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_failing_config(tmp: Path, **overrides) -> Path:
    """把默认配置复制到临时目录并只改**测试用**阈值（不提交为默认配置）。"""
    cfg = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
    cfg["acceptance"].update(overrides)
    p = tmp / "config_negative_test.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ------------------------------------------------------------------ 负面测试
class TestNegativePaths(unittest.TestCase):
    """S4.3：失败、缺测、非有限值都必须被发现，且已执行的失败返回非零。"""

    @classmethod
    def setUpClass(cls):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="cases_", dir=str(REPORT_DIR)))

    @classmethod
    def tearDownClass(cls):
        # 按用户要求保留所有测试证据，不自动删除临时目录。
        pass

    # 1. --only direct 且阈值 0 → B 组 fail、退出码非零
    def test_case1_direct_identity_threshold_zero_fails(self):
        cfg = write_failing_config(self.tmp, discrete_identity_rel_l2_max=0.0)
        out = self.tmp / "case1"
        r = run_main(["--light", "--config", str(cfg), "--run-dir", str(out), "--only", "direct"])
        ov = read_overall(out)
        self.assertNotEqual(r["exit_code"], 0, "阈值 0 时离散恒等必须失败并返回非零")
        self.assertEqual(ov["checks"]["discrete_identity_rel_L2"]["status"], "fail")
        self.assertEqual(ov["status"], "fail")
        self._record("case1_only_direct_threshold0", r, ov, out)

    # 2. Airy 第一个通过、第二个失败 → 非零
    def test_case2_airy_second_case_fails(self):
        def patch(mod):
            real = mod.study_airy

            def fake(cfg, run_dir, logger, show):
                res = real(cfg, run_dir, logger, show)
                res["cases"][1]["first_ring_rel_err"] = 0.5
                res["cases"][1]["line_L1_vs_airy"] = 0.7
                return res
            mod.study_airy = fake
        out = self.tmp / "case2"
        r = run_main(base_argv(out, ["--only", "airy"]), monkeypatch=patch)
        ov = read_overall(out)
        self.assertNotEqual(r["exit_code"], 0)
        self.assertEqual(ov["checks"]["airy_first_dark_ring_rel_err"]["status"], "fail")
        self.assertIn("D0.2mm_f25mm", ov["checks"]["airy_first_dark_ring_rel_err"]["detail"])
        self._record("case2_airy_second_fails", r, ov, out)

    # 3. Airy 前两个通过、论文尺度第三个失败 → 非零
    def test_case3_airy_third_paper_scale_fails(self):
        def patch(mod):
            real = mod.study_airy

            def fake(cfg, run_dir, logger, show):
                res = real(cfg, run_dir, logger, show)
                res["cases"][2]["first_ring_rel_err"] = 0.5
                res["cases"][2]["line_L1_vs_airy"] = 0.8
                return res
            mod.study_airy = fake
        out = self.tmp / "case3"
        r = run_main(base_argv(out, ["--only", "airy"]), monkeypatch=patch)
        ov = read_overall(out)
        self.assertNotEqual(r["exit_code"], 0)
        self.assertEqual(ov["checks"]["airy_line_L1_vs_analytic"]["status"], "fail")
        self.assertIn("D1mm_f50mm",
                      ov["checks"]["airy_line_L1_vs_analytic"]["detail"])
        self._record("case3_airy_paper_scale_fails", r, ov, out)

    # 4. 注入 NaN / Inf / 缺失字段 → fail 且非零
    def test_case4_nan_inf_missing_fields_fail(self):
        for tag, value in (("nan", float("nan")), ("inf", float("inf")),
                           ("missing", None)):
            def patch(mod, value=value, tag=tag):
                real = mod.study_energy

                def fake(cfg, run_dir, logger, show):
                    res = real(cfg, run_dir, logger, show)
                    if value is None:
                        res.pop("parseval_worst_rel_err", None)
                    else:
                        res["parseval_worst_rel_err"] = value
                    return res
                mod.study_energy = fake
            out = self.tmp / ("case4_" + tag)
            r = run_main(base_argv(out, ["--only", "energy"]), monkeypatch=patch)
            ov = read_overall(out)
            self.assertNotEqual(r["exit_code"], 0, "%s 必须使运行失败" % tag)
            self.assertEqual(ov["checks"]["native_fft_parseval_rel_err"]["status"], "fail")
            self._record("case4_%s" % tag, r, ov, out)

    # 5. 默认组抛异常 → fail、非零、保留可诊断堆栈
    def test_case5_study_exception_fails_with_traceback(self):
        def patch(mod):
            def boom(*a, **kw):
                raise RuntimeError("注入的异常（负面测试）")
            mod.study_sampling = boom
        out = self.tmp / "case5"
        r = run_main(base_argv(out, ["--only", "sampling"]), monkeypatch=patch)
        ov = read_overall(out)
        self.assertNotEqual(r["exit_code"], 0)
        self.assertTrue(ov["failures"], "必须记录异常")
        self.assertIn("注入的异常", json.dumps(ov, ensure_ascii=False))
        log = (out / "run.log").read_text(encoding="utf-8", errors="ignore")
        self.assertIn("Traceback", log, "run.log 必须保留可诊断堆栈")
        self._record("case5_study_exception", r, ov, out)

    # 6. 通过的 --only direct → partial，其他组 not_run，B 组不是 not_run
    def test_case6_subset_pass_is_partial_and_direct_measured(self):
        out = self.tmp / "case6"
        r = run_main(base_argv(out, ["--only", "direct"]))
        ov = read_overall(out)
        self.assertEqual(r["exit_code"], 0, "成功的子集运行允许返回 0")
        self.assertEqual(ov["status"], "partial")
        self.assertEqual(ov["checks"]["discrete_identity_rel_L2"]["status"], "pass")
        for k in ("native_fft_parseval_rel_err", "input_sampling_intensity_L1"):
            self.assertEqual(ov["checks"][k]["status"], "not_run")
        self.assertNotIn("direct", ov["not_run"])
        self.assertIn("airy", ov["not_run"])
        self._record("case6_subset_partial", r, ov, out)

    # 7. 全部必需组通过 → pass（防止新汇总误判正常结果）
    def test_case7_all_groups_pass(self):
        out = self.tmp / "case7"
        r = run_main(base_argv(out))
        ov = read_overall(out)
        self.assertEqual(r["exit_code"], 0)
        self.assertEqual(ov["status"], "pass")
        self.assertFalse(ov["not_run"])
        for k, v in ov["checks"].items():
            self.assertEqual(v["status"], "pass", "%s 应为 pass：%s" % (k, v))
        self._record("case7_all_pass", r, ov, out)

    # ------------------------------------------------------------------ 记录
    def _record(self, name: str, r: Dict[str, Any], ov: Dict[str, Any],
                out: Path) -> None:
        rec = {
            "name": name,
            "argv": ["python", "-B", "main.py", "--run-dir", str(out)],
            "expected_nonzero": name not in ("case6_subset_partial", "case7_all_pass"),
            "exit_code": r["exit_code"],
            "overall_status": ov.get("status"),
            "checks": {k: v["status"] for k, v in (ov.get("checks") or {}).items()},
            "failures": ov.get("failures"),
            "log_path": str(out / "run.log"),
            "acceptance_path": str(out / "acceptance_check.json"),
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / ("%s.json" % name)).write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="阶段 01 第二轮：负面测试与状态语义验证")
    p.add_argument("--report", type=str, default=None,
                   help="把每条记录写入该目录（默认 results/stage01/_negative_tests）")
    args, rest = p.parse_known_args()
    global REPORT_DIR
    if args.report:
        REPORT_DIR = Path(args.report)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNegativePaths)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    summary = {
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "report_dir": str(REPORT_DIR),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
