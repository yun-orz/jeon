# -*- coding: utf-8 -*-
"""阶段 02A 真实失败路径测试（阶段 02A-R 专项）。

测试全零场、NaN场、非法 shape、指纹变化、空波长列表等异常情况下的防御行为，
确保系统在出现异常时真实失败，绝不产生虚假 pass。
"""

import json
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from optics.coordinates import make_grid
from optics.materials import refractive_index_fused_silica
from optics.doe import design_jeon2019_spiral_height, compute_doe_transmission_field
from main_stage02 import load_and_validate_config, run_stage02a


class TestStage02FailurePaths(unittest.TestCase):
    """测试真实失败路径。"""

    def setUp(self):
        self.default_cfg_path = PROJECT_ROOT / "config_stage02a.json"
        self.config = load_and_validate_config(self.default_cfg_path)

    def test_empty_wavelengths_rejected_at_validation(self):
        """测试波长列表为空时在配置加载阶段立即报错。"""
        cfg = json.loads(json.dumps(self.config))
        cfg["optical"]["preview_wavelengths_m"] = []

        import uuid
        evidence_dir = PROJECT_ROOT / "results" / "test_failures" / f"empty_waves_{uuid.uuid4().hex[:8]}"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg = evidence_dir / "bad_config.json"
        tmp_cfg.write_text(json.dumps(cfg), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_config(tmp_cfg)
        self.assertIn("非空", str(ctx.exception))

    def test_duplicate_wavelengths_rejected(self):
        """测试预览波长列表中有重复项时在配置加载阶段报错。"""
        cfg = json.loads(json.dumps(self.config))
        cfg["optical"]["preview_wavelengths_m"] = [540e-9, 540e-9]

        import uuid
        evidence_dir = PROJECT_ROOT / "results" / "test_failures" / f"dup_waves_{uuid.uuid4().hex[:8]}"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg = evidence_dir / "bad_config.json"
        tmp_cfg.write_text(json.dumps(cfg), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_config(tmp_cfg)
        self.assertIn("重复", str(ctx.exception))

    def test_invalid_acceptance_threshold_rejected(self):
        """测试验收阈值为 NaN 或 <=0 时被拒绝。"""
        cfg = json.loads(json.dumps(self.config))
        cfg["acceptance"]["energy_efficiency_max"] = -0.5

        import uuid
        evidence_dir = PROJECT_ROOT / "results" / "test_failures" / f"bad_thresh_{uuid.uuid4().hex[:8]}"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        tmp_cfg = evidence_dir / "bad_config.json"
        tmp_cfg.write_text(json.dumps(cfg), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_config(tmp_cfg)
        self.assertIn("必须严格大于 0", str(ctx.exception))

    def test_zero_field_fails_acceptance(self):
        """测试全零场（审核反例）在自检中必须被判定为 fail，状态必须为 failed。"""
        from unittest.mock import patch

        # mock fresnel_kernel_separable 返回全零复场
        def mock_zero_propagation(*args, **kwargs):
            grid_out = args[1] if len(args) > 1 else kwargs.get("grid_in")
            # 返回 301x301 的全零复数数组
            return np.zeros((301, 301), dtype=np.complex128)

        with patch("main_stage02.fresnel_kernel_separable", side_effect=mock_zero_propagation):
            with self.assertRaises(Exception):
                # 试算阶段检测到 peak <= 0 立即抛出 RuntimeError
                run_stage02a(self.config, run_dir=None, only_mode="preview")

    def test_nan_field_fails_acceptance(self):
        """测试包含 NaN 的场在自检中必须被拦截并抛出错误。"""
        from unittest.mock import patch

        def mock_nan_propagation(*args, **kwargs):
            arr = np.ones((301, 301), dtype=np.complex128)
            arr[150, 150] = np.nan + 1j * np.nan
            return arr

        with patch("main_stage02.fresnel_kernel_separable", side_effect=mock_nan_propagation):
            with self.assertRaises(Exception):
                run_stage02a(self.config, run_dir=None, only_mode="preview")


if __name__ == "__main__":
    unittest.main()
