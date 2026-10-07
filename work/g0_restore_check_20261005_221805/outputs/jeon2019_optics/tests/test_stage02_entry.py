# -*- coding: utf-8 -*-
"""阶段 02A 入口与命令行参数测试。"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _child_env():
    """显式确定子进程的 I/O 编码（R8）。

    读端 `encoding='utf-8'` 只解决解码；中文 Windows 下子进程默认按 GBK 输出，
    因此必须**同时**设置写端编码。这里不传 `errors`，让真实编码问题直接暴露，
    而不是被 `replace` 静默掩盖成乱码。
    """
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    return env


class TestStage02Entry(unittest.TestCase):
    """测试 main_stage02.py 的执行参数与路径行为。"""

    def setUp(self):
        self.python_exe = sys.executable
        self.main_stage02 = PROJECT_ROOT / "main_stage02.py"

    def test_entry_no_save_mode(self):
        """测试 --no-save 模式：内存完成自检，退出码为 0，不创建垃圾文件。"""
        cmd = [
            self.python_exe,
            "-B",
            str(self.main_stage02),
            "--only", "height",
            "--no-save",
        ]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding='utf-8', env=_child_env())
        self.assertEqual(res.returncode, 0, f"执行失败:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        self.assertIn("阶段 02A 开始执行", res.stdout)
        self.assertIn("阶段 02A 执行完成", res.stdout)

    def test_invalid_config_fails_cleanly(self):
        """测试非法配置时非零退出并给出明确错误（永久保留测试配置与日志证据，严禁自动删除）。"""
        bad_configs = [
            # 1. 波长范围倒置 (min > max)
            ("wavelength_inverted", {
                "optical": {
                    "diameter_m": 1e-3, "focal_length_m": 50e-3, "distance_m": 50e-3,
                    "wings_N": 3, "design_wavelength_min_m": 700e-9, "design_wavelength_max_m": 400e-9,
                    "material_model": "fused_silica_malitson1965", "control_wavelength_m": 550e-9,
                    "preview_wavelengths_m": [420e-9, 540e-9, 660e-9]
                },
                "grid": {
                    "input": {"half_width_m": 550e-6, "n": 1101, "spacing_m": 1e-6},
                    "output": {"half_width_m": 150e-6, "n": 301, "spacing_m": 1e-6},
                },
                "propagation": {"method": "separable_kernel", "include_global_phase": True},
                "runtime": {"save_results": False, "show_plots": False},
                "acceptance": {"energy_efficiency_max": 1.02}
            }),
            # 2. 输入半宽与实际采样不符
            ("input_half_width_mismatch", {
                "optical": {
                    "diameter_m": 1e-3, "focal_length_m": 50e-3, "distance_m": 50e-3,
                    "wings_N": 3, "design_wavelength_min_m": 420e-9, "design_wavelength_max_m": 660e-9,
                    "material_model": "fused_silica_malitson1965", "control_wavelength_m": 550e-9,
                    "preview_wavelengths_m": [420e-9, 540e-9, 660e-9]
                },
                "grid": {
                    "input": {"half_width_m": 300e-6, "n": 1101, "spacing_m": 1e-6}, # (1101//2)*1e-6=550e-6 != 300e-6
                    "output": {"half_width_m": 150e-6, "n": 301, "spacing_m": 1e-6},
                },
                "propagation": {"method": "separable_kernel", "include_global_phase": True},
                "runtime": {"save_results": False, "show_plots": False},
                "acceptance": {"energy_efficiency_max": 1.02}
            }),
            # 3. 非整数翼数 (如 3.5)
            ("float_wings_N", {
                "optical": {
                    "diameter_m": 1e-3, "focal_length_m": 50e-3, "distance_m": 50e-3,
                    "wings_N": 3.5, "design_wavelength_min_m": 420e-9, "design_wavelength_max_m": 660e-9,
                    "material_model": "fused_silica_malitson1965", "control_wavelength_m": 550e-9,
                    "preview_wavelengths_m": [420e-9, 540e-9, 660e-9]
                },
                "grid": {
                    "input": {"half_width_m": 550e-6, "n": 1101, "spacing_m": 1e-6},
                    "output": {"half_width_m": 150e-6, "n": 301, "spacing_m": 1e-6},
                },
                "propagation": {"method": "separable_kernel", "include_global_phase": True},
                "runtime": {"save_results": False, "show_plots": False},
                "acceptance": {"energy_efficiency_max": 1.02}
            }),
            # 4. 未知材料模型
            ("unknown_material", {
                "optical": {
                    "diameter_m": 1e-3, "focal_length_m": 50e-3, "distance_m": 50e-3,
                    "wings_N": 3, "design_wavelength_min_m": 420e-9, "design_wavelength_max_m": 660e-9,
                    "material_model": "unknown_glass", "control_wavelength_m": 550e-9,
                    "preview_wavelengths_m": [420e-9, 540e-9, 660e-9]
                },
                "grid": {
                    "input": {"half_width_m": 550e-6, "n": 1101, "spacing_m": 1e-6},
                    "output": {"half_width_m": 150e-6, "n": 301, "spacing_m": 1e-6},
                },
                "propagation": {"method": "separable_kernel", "include_global_phase": True},
                "runtime": {"save_results": False, "show_plots": False},
                "acceptance": {"energy_efficiency_max": 1.02}
            }),
            # 5. 未知传播模型
            ("unknown_propagation", {
                "optical": {
                    "diameter_m": 1e-3, "focal_length_m": 50e-3, "distance_m": 50e-3,
                    "wings_N": 3, "design_wavelength_min_m": 420e-9, "design_wavelength_max_m": 660e-9,
                    "material_model": "fused_silica_malitson1965", "control_wavelength_m": 550e-9,
                    "preview_wavelengths_m": [420e-9, 540e-9, 660e-9]
                },
                "grid": {
                    "input": {"half_width_m": 550e-6, "n": 1101, "spacing_m": 1e-6},
                    "output": {"half_width_m": 150e-6, "n": 301, "spacing_m": 1e-6},
                },
                "propagation": {"method": "unknown_propagation_method", "include_global_phase": True},
                "runtime": {"save_results": False, "show_plots": False},
                "acceptance": {"energy_efficiency_max": 1.02}
            }),
            # 6. NaN 传播距离
            ("nan_distance", {
                "optical": {
                    "diameter_m": 1e-3, "focal_length_m": 50e-3, "distance_m": float("nan"),
                    "wings_N": 3, "design_wavelength_min_m": 420e-9, "design_wavelength_max_m": 660e-9,
                    "material_model": "fused_silica_malitson1965", "control_wavelength_m": 550e-9,
                    "preview_wavelengths_m": [420e-9, 540e-9, 660e-9]
                },
                "grid": {
                    "input": {"half_width_m": 550e-6, "n": 1101, "spacing_m": 1e-6},
                    "output": {"half_width_m": 150e-6, "n": 301, "spacing_m": 1e-6},
                },
                "propagation": {"method": "separable_kernel", "include_global_phase": True},
                "runtime": {"save_results": False, "show_plots": False},
                "acceptance": {"energy_efficiency_max": 1.02}
            }),
        ]

        import time, uuid
        evidence_root = PROJECT_ROOT / "results" / "test_stage02_entry"
        evidence_root.mkdir(parents=True, exist_ok=True)

        for case_name, bad_cfg in bad_configs:
            case_dir = evidence_root / f"case_{case_name}_{uuid.uuid4().hex[:8]}"
            case_dir.mkdir(parents=True, exist_ok=False)

            cfg_file = case_dir / "bad_config.json"
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(bad_cfg, f, ensure_ascii=False, indent=2)

            cmd = [
                self.python_exe,
                "-B",
                str(self.main_stage02),
                "--config", str(cfg_file),
                "--no-save",
            ]
            res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding='utf-8', env=_child_env())

            # 写入证据日志，绝不删除任何测试产生的文件
            (case_dir / "stdout.log").write_text(res.stdout, encoding="utf-8")
            (case_dir / "stderr.log").write_text(res.stderr, encoding="utf-8")
            (case_dir / "result.json").write_text(
                json.dumps({"exit_code": res.returncode, "cmd": cmd}, indent=2),
                encoding="utf-8"
            )

            self.assertNotEqual(
                res.returncode, 0,
                f"非法配置用例 [{case_name}] 必须返回非零退出码！\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

    def test_invocation_from_outside_project(self):
        """测试从项目外部工作目录调用 main_stage02.py 能够正常通过 __file__ 解析路径。"""
        outside_cwd = PROJECT_ROOT.parent  # 上级目录
        cmd = [
            self.python_exe,
            "-B",
            str(self.main_stage02),
            "--only", "height",
            "--no-save",
        ]
        res = subprocess.run(cmd, cwd=outside_cwd, capture_output=True, text=True, encoding='utf-8', env=_child_env())
        self.assertEqual(res.returncode, 0, f"外部调用失败:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    def test_only_height_is_partial_not_all_pass(self):
        """测试 --only height 运行结果状态必须是 partial，未运行项在 self_check 中标为 not_run。"""
        import uuid
        evidence_root = PROJECT_ROOT / "results" / "test_stage02_entry" / f"partial_height_{uuid.uuid4().hex[:8]}"
        evidence_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_exe,
            "-B",
            str(self.main_stage02),
            "--only", "height",
            "--run-dir", str(evidence_root),
        ]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding='utf-8', env=_child_env())
        self.assertEqual(res.returncode, 0)

        # 检查 completion_state.json
        state_file = evidence_root / "completion_state.json"
        self.assertTrue(state_file.exists(), "未生成 completion_state.json")
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state.get("status"), "partial", f"只跑 height 时状态必须是 partial，得到 {state.get('status')}")
        self.assertFalse(state.get("all_checks_pass"), "未完整执行时不应声明 all_checks_pass 为 True")

        # 检查 self_check.json
        chk_file = evidence_root / "self_check.json"
        self.assertTrue(chk_file.exists(), "未生成 self_check.json")
        with open(chk_file, "r", encoding="utf-8") as f:
            chk = json.load(f)
        self.assertEqual(chk.get("control_550nm", {}).get("status"), "not_run")
        self.assertEqual(chk.get("psf_preview", {}).get("status"), "not_run")


if __name__ == "__main__":
    unittest.main()

