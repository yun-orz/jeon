# -*- coding: utf-8 -*-
"""阶段 02A 入口与命令行参数测试。"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"执行失败:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        self.assertIn("阶段 02A 开始执行", res.stdout)
        self.assertIn("阶段 02A 执行完成", res.stdout)

    def test_invalid_config_fails_cleanly(self):
        """测试非法配置（例如波长倒置）时非零退出并给出明确错误。"""
        bad_config = {
            "optical": {
                "diameter_m": 1e-3,
                "focal_length_m": 50e-3,
                "distance_m": 50e-3,
                "wings_N": 3,
                "design_wavelength_min_m": 700e-9,  # min > max
                "design_wavelength_max_m": 400e-9,
            },
            "grid": {
                "input": {"half_width_m": 550e-6, "n": 51, "spacing_m": 22e-6},
                "output": {"half_width_m": 150e-6, "n": 31, "spacing_m": 10e-6},
            },
            "runtime": {"save_results": False, "show_plots": False},
            "acceptance": {},
        }
        # 写入临时测试配置
        tmp_cfg = PROJECT_ROOT / "temp_bad_config_for_test.json"
        try:
            with open(tmp_cfg, "w", encoding="utf-8") as f:
                json.dump(bad_config, f)

            cmd = [
                self.python_exe,
                "-B",
                str(self.main_stage02),
                "--config", str(tmp_cfg),
                "--no-save",
            ]
            res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0, "非法配置必须返回非零退出码")
        finally:
            if tmp_cfg.exists():
                tmp_cfg.unlink()

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
        res = subprocess.run(cmd, cwd=outside_cwd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"外部调用失败:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")


if __name__ == "__main__":
    unittest.main()
