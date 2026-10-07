# -*- coding: utf-8 -*-
"""核对M1独立审核适配层与既有审核器的判定逻辑一致。

对应 tasks/M1_gpu_environment.md 第17条：既有审核器的固定输出路径会覆盖历史
产物，因此改用适配层；本测试用 AST 对比证明"只改输出路径，没改判定与阈值"。
"""
import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
ADAPTED = ROOT / "outputs/执行_M1_GPU环境_20261007/audit_gpu_m1.py"
ORIGINAL = ROOT / "outputs/执行_GPU验证_20261006/audit_gpu.py"
EXPECTED_THRESHOLDS = {1e-10, 1e-5, 1e-12, 1e-30, 3}


def strip(node):
    """去掉文档字符串、字符串常量与名称差异，只保留结构与数值。"""
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = item.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                item.body = body[1:] or [ast.Pass()]
    return node


def normalised(path, function):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return ast.dump(strip(node))
    raise AssertionError("找不到函数 %s：%s" % (function, path))


def numbers(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            found.add(node.value)
    return found


class TestAuditAdapter(unittest.TestCase):
    def test_both_scripts_exist(self):
        self.assertTrue(ADAPTED.is_file())
        self.assertTrue(ORIGINAL.is_file())

    def test_norm_error_identical(self):
        self.assertEqual(normalised(ORIGINAL, "norm_error"), normalised(ADAPTED, "norm_error"))

    def test_thresholds_and_counts_unchanged(self):
        """审核阈值、容差与追溯阶段数必须逐字保留。"""
        original = numbers(ORIGINAL)
        adapted = numbers(ADAPTED)
        self.assertTrue(EXPECTED_THRESHOLDS.issubset(original))
        self.assertTrue(EXPECTED_THRESHOLDS.issubset(adapted))
        self.assertEqual(original - adapted, set(), "适配层删除了原有数值：%s" % (original - adapted))

    def test_adapted_adds_no_relaxed_comparison(self):
        """适配层不得新增任何放宽比较的运算符或阈值替换。"""
        source = ADAPTED.read_text(encoding="utf-8")
        # 引擎完整性作用域辅助函数之外的正文不得出现放宽阈值。
        body = source.split("def main():", 1)[1]
        for forbidden in ["1e-9", "1e-8", "1e-7", "1e-6", "1e-4", "> 1e-10", "or True"]:
            self.assertNotIn(forbidden, body, "出现可疑的放宽阈值：" + forbidden)
        self.assertIn("passed = (summary['gpu_validation_passed']", body)
        self.assertIn("max(optical.values()) < 1e-10", body)
        self.assertIn("max(equation_errors) < 1e-5", body)
        self.assertIn("error < 1e-12", body)
        self.assertIn("not engine_changed", body)

    def test_engine_integrity_scope_matches_m0_exclusions(self):
        """引擎完整性比较必须排除M0已排除的新结果目录，并同时记录该子树的差异。"""
        source = ADAPTED.read_text(encoding="utf-8")
        self.assertIn("EXCLUDED_PREFIX = 'results/reproduction_v1'", source)
        self.assertIn("engine_files_changed", source)
        self.assertIn("new_results_files_changed", source)
        protection = HERE.parents[2] / "outputs/jeon2019_optics/results/reproduction_v1/m0/run_20261006_230835_592a12d3/protection.json"
        import json
        excluded = " ".join(json.loads(protection.read_text(encoding="utf-8"))["excluded"])
        self.assertIn("新reproduction_v1代码/数据/结果", excluded,
                      "M0保护清单未声明排除新结果目录时，本作用域不成立")

    def test_adapted_writes_only_into_new_m1_directory(self):
        """适配层不得再指向历史审核目录下的固定输出名。"""
        source = ADAPTED.read_text(encoding="utf-8")
        self.assertNotIn("Path(__file__).with_name('independent_audit.json')", source)
        self.assertNotIn("Path(__file__).with_name('最终审核.json')", source)
        self.assertIn("PERSIST_DIR/'independent_audit.json'", source)
        self.assertIn("PERSIST_DIR/'最终审核.json'", source)
        self.assertEqual(ADAPTED.parent.name, "执行_M1_GPU环境_20261007")

    def test_original_audit_untouched(self):
        """原审核器本体必须与M1开始前记录的SHA一致。"""
        import hashlib
        import json
        baseline = json.loads((HERE / "tests/m1_frozen_audit_sha.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
        self.assertEqual(digest, baseline["audit_gpu_py_sha256"],
                         "原审核器被改动；M1只允许新增适配层")
        self.assertEqual(baseline["cpu_preflight_json_sha256"],
                         hashlib.sha256((ROOT / "outputs/执行_GPU验证_20261006/cpu_preflight.json")
                                        .read_bytes()).hexdigest(),
                         "M1前置的CPU预审记录被改动")
        self.assertEqual(baseline["main_g3_gpu_py_sha256"],
                         hashlib.sha256((ROOT / "outputs/jeon2019_optics/main_g3_gpu.py").read_bytes()).hexdigest(),
                         "正式GPU入口被改动")
        self.assertEqual(baseline["config_g3_gpu_json_sha256"],
                         hashlib.sha256((ROOT / "outputs/jeon2019_optics/config_g3_gpu.json").read_bytes()).hexdigest(),
                         "正式GPU阈值配置被改动")


class TestSummaryExtraction(unittest.TestCase):
    """正式入口先打印汇总JSON，再打印"GPU验证结果：<路径>"等尾部行。"""

    def adapter(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("audit_gpu_m1_under_test", ADAPTED)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_nested_json_before_trailing_lines(self):
        module = self.adapter()
        summary = {"gpu_validation_passed": True, "optical": {"numpy_forward_relative": 5e-16},
                   "stage_checks": {"initial": 2e-07}, "smoke": {"cpu": {"loss": 0.0157}}, "n": 1}
        import json
        stdout = ("GPU双精度光学算子、伴随及梯度完成；核对冻结模型的8个验证块\n"
                  + json.dumps(summary, ensure_ascii=False, indent=2)
                  + "\nGPU验证结果：D:\\x\\results\\g3_gpu\\run_20261007_103806\n")
        self.assertEqual(module.extract_summary(stdout), summary)

    def test_missing_or_broken_output_returns_none(self):
        module = self.adapter()
        self.assertIsNone(module.extract_summary(None))
        self.assertIsNone(module.extract_summary("没有JSON的普通输出"))
        self.assertIsNone(module.extract_summary("{不是合法JSON"))

    def test_engine_scope_helper_excludes_new_results_only(self):
        module = self.adapter()
        self.assertEqual(module.EXCLUDED_PREFIX, "results/reproduction_v1")
        before = {"results/reproduction_v1/m2/a.json": "1", "main_g1.py": "x", "optics/g3_hqs.py": "y"}
        after = {"results/reproduction_v1/m2/a.json": "2", "results/reproduction_v1/m2/b.json": "3",
                 "main_g1.py": "x", "optics/g3_hqs.py": "y"}
        self.assertEqual(module.fingerprint_diff(
            {k: v for k, v in before.items() if not k.startswith(module.EXCLUDED_PREFIX)},
            {k: v for k, v in after.items() if not k.startswith(module.EXCLUDED_PREFIX)}), [])
        self.assertNotEqual(module.fingerprint_diff(before, after), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
