# -*- coding: utf-8 -*-
"""材料色散模型单元测试（阶段 02A）。"""

import unittest
import numpy as np

from optics.materials import (
    refractive_index_fused_silica,
    delta_refractive_index,
    get_material_model_metadata,
    FUSED_SILICA_WAVELENGTH_MIN_M,
    FUSED_SILICA_WAVELENGTH_MAX_M,
)


class TestMaterials(unittest.TestCase):
    """测试熔融石英 Sellmeier 色散模型。"""

    def test_independent_point_check_550nm(self):
        """550nm 处折射率独立验算：标准熔融石英 n ≈ 1.459917，误差 <= 2e-5。"""
        lam = 550e-9  # 550 nm = 0.55 μm
        n = refractive_index_fused_silica(lam)
        self.assertIsInstance(n, float)
        self.assertAlmostEqual(n, 1.459917, places=4)
        self.assertLess(abs(n - 1.459917), 2e-5)

    def test_visible_spectrum_monotonicity(self):
        """可见光区（420nm 到 660nm）折射率正常且随波长增大单调递减（正常色散）。"""
        wavelengths_nm = np.linspace(420, 660, 25)
        wavelengths_m = wavelengths_nm * 1e-9
        n_vals = refractive_index_fused_silica(wavelengths_m)

        # 全为有限正数
        self.assertTrue(np.all(np.isfinite(n_vals)))
        self.assertTrue(np.all(n_vals > 1.4))
        self.assertTrue(np.all(n_vals < 1.6))

        # 单调递减 dn/dλ < 0
        diffs = np.diff(n_vals)
        self.assertTrue(np.all(diffs < 0.0), f"折射率非单调递减: {diffs}")

        # 具体三点对照：420nm > 540nm > 660nm
        n_420 = refractive_index_fused_silica(420e-9)
        n_540 = refractive_index_fused_silica(540e-9)
        n_660 = refractive_index_fused_silica(660e-9)
        self.assertGreater(n_420, n_540)
        self.assertGreater(n_540, n_660)

    def test_delta_refractive_index(self):
        """折射率差 Δn = n - n_air，对 n_air=1 应满足 Δn = n - 1。"""
        lam = 540e-9
        n = refractive_index_fused_silica(lam)
        dn = delta_refractive_index(lam, n_air=1.0)
        self.assertAlmostEqual(dn, n - 1.0, places=12)

    def test_invalid_wavelengths_raise(self):
        """非正数、NaN、Inf、超出有效范围均应报错。"""
        # 负数与 0
        with self.assertRaises(ValueError):
            refractive_index_fused_silica(0.0)
        with self.assertRaises(ValueError):
            refractive_index_fused_silica(-550e-9)

        # NaN 与 Inf
        with self.assertRaises(ValueError):
            refractive_index_fused_silica(np.nan)
        with self.assertRaises(ValueError):
            refractive_index_fused_silica(np.inf)

        # 误将 nm 作为 m 传入（例如 550.0），必须被拦截
        with self.assertRaises(ValueError):
            refractive_index_fused_silica(550.0)

        # 超出有效边界
        with self.assertRaises(ValueError):
            refractive_index_fused_silica(FUSED_SILICA_WAVELENGTH_MIN_M * 0.9)
        with self.assertRaises(ValueError):
            refractive_index_fused_silica(FUSED_SILICA_WAVELENGTH_MAX_M * 1.1)

    def test_metadata_completeness(self):
        """元数据包含文献、DOI、温度及范围。"""
        meta = get_material_model_metadata()
        self.assertIn("doi", meta)
        self.assertIn("10.1364/JOSA.55.001205", meta["doi"])
        self.assertEqual(meta["assumed_temperature_celsius"], 20.0)


if __name__ == "__main__":
    unittest.main()
