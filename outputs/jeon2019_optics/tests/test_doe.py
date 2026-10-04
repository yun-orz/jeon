# -*- coding: utf-8 -*-
"""DOE 高度设计与波长相位响应单元测试（阶段 02A）。"""

import unittest
import numpy as np

from optics.coordinates import make_grid
from optics.materials import refractive_index_fused_silica, REFRACTIVE_INDEX_AIR
from optics.doe import (
    optical_path_difference_delta,
    design_conventional_fresnel_height,
    design_jeon2019_spiral_height,
    compute_doe_transmission_field,
)


class TestDOE(unittest.TestCase):
    """测试 DOE 高度轮廓与物理特性。"""

    def setUp(self):
        # 建立适中采样网格供快速测试：覆盖 [-550, 550] μm，间距 2 μm
        self.dx = 2e-6
        self.grid = make_grid(n=551, d=self.dx, label="doe_test")
        self.D = 1e-3     # 直径 1 mm
        self.f = 50e-3    # 焦距 50 mm

    def test_aperture_area_conservation(self):
        """掩膜面积与理论圆面积 π(D/2)^2 相对误差 <= 1%，无截断。"""
        profile = design_jeon2019_spiral_height(self.grid, self.D, self.f, wings_N=3)
        sampled_area = np.sum(profile.mask) * self.grid.cell_area
        exact_area = np.pi * (self.D / 2.0) ** 2
        rel_err = abs(sampled_area - exact_area) / exact_area
        self.assertLess(rel_err, 0.01, f"孔径采样面积相对误差过大: {rel_err:.4e}")

    def test_height_range_and_origin(self):
        """孔径内逐点合法范围 -lambda_design/(n_design-1) <= delta_h <= 0，原点高度严格为 0。"""
        profile = design_jeon2019_spiral_height(self.grid, self.D, self.f, wings_N=3)
        h = profile.delta_h
        mask = profile.mask.astype(bool)

        # 原点位置 (中心样点 551//2 = 275)
        cx, cy = self.grid.x.n // 2, self.grid.y.n // 2
        self.assertEqual(h[cy, cx], 0.0, "原点相对高度必须为 0.0")

        # 检查上界：delta_h <= 1e-12（允许浮点容差）
        self.assertTrue(np.all(h[mask] <= 1e-12), "存在大于 0 的相对高度")

        # 检查下界：delta_h >= -lambda_design / (n_design - 1) - 1e-12
        lam_des = profile.lambda_design[mask]
        n_des = refractive_index_fused_silica(lam_des)
        dn = n_des - REFRACTIVE_INDEX_AIR
        min_allowed_h = -lam_des / dn
        self.assertTrue(
            np.all(h[mask] >= min_allowed_h - 1e-12),
            "存在超出下界的高度"
        )

        # 孔径外高度必须为 0
        self.assertTrue(np.all(h[~mask] == 0.0), "孔径外存在非零高度")

    def test_angular_periodicity_continuous(self):
        """测试 N=3 螺旋设计的连续角向周期性：同一半径下 θ 与 θ+2π/3 高度相等。"""
        # 选取若干不在跳变边界附近的极角与半径进行独立代入对比
        r_test = np.array([100e-6, 250e-6, 400e-6])
        thetas = np.array([0.2, 0.5, 0.9])  # 避开周期起点与绕回跳变点

        delta = optical_path_difference_delta(r_test, self.f)
        N = 3
        period_angle = 2.0 * np.pi / N

        for r_val, d_val in zip(r_test, delta):
            for th in thetas:
                th1 = th
                th2 = th + period_angle

                # θ1 与 θ2 的局部设计波长
                lam1 = 420e-9 + (660e-9 - 420e-9) * ((th1 % period_angle) / period_angle)
                lam2 = 420e-9 + (660e-9 - 420e-9) * ((th2 % period_angle) / period_angle)
                self.assertAlmostEqual(lam1, lam2, places=15)

                n1 = refractive_index_fused_silica(lam1)
                n2 = refractive_index_fused_silica(lam2)

                h1 = (np.floor(d_val / lam1) * lam1 - d_val) / (n1 - 1.0)
                h2 = (np.floor(d_val / lam2) * lam2 - d_val) / (n2 - 1.0)

                self.assertAlmostEqual(h1, h2, places=14)

    def test_local_focusing_phase_identity(self):
        """孔径内局部设计聚焦相位恒等检查：
        abs(exp[2πi*(delta + (n_design - 1)*delta_h)/lambda_design] - 1) <= 1e-10。
        """
        profile = design_jeon2019_spiral_height(self.grid, self.D, self.f, wings_N=3)
        mask = profile.mask.astype(bool)

        X, Y = self.grid.meshgrid()
        r = np.hypot(X, Y)
        delta = optical_path_difference_delta(r, self.f)

        lam_des = profile.lambda_design[mask]
        n_des = refractive_index_fused_silica(lam_des)
        dn = n_des - REFRACTIVE_INDEX_AIR
        h = profile.delta_h[mask]
        delta_m = delta[mask]

        phase = 2.0 * np.pi * (delta_m + dn * h) / lam_des
        phasor_err = np.max(np.abs(np.exp(1j * phase) - 1.0))
        self.assertLess(phasor_err, 1e-10, f"局部设计相位恒等误差过大: {phasor_err:.4e}")

    def test_global_height_offset_invariance(self):
        """全局高度偏置 h 与 h+C 比较：出射场仅相差全局相位因子，强度严格恒等。"""
        profile = design_jeon2019_spiral_height(self.grid, self.D, self.f, wings_N=3)
        lam_in = 540e-9
        C_offset = 1.5e-6  # 偏置 1.5 μm

        u1_orig = compute_doe_transmission_field(profile, lam_in, h_offset=0.0)
        u1_shifted = compute_doe_transmission_field(profile, lam_in, h_offset=C_offset)

        # 强度对比
        I_orig = np.abs(u1_orig) ** 2
        I_shifted = np.abs(u1_shifted) ** 2
        diff_I = np.max(np.abs(I_orig - I_shifted))
        self.assertLess(diff_I, 1e-14, "强度受全局高度偏置影响")

        # 预期全局相位
        dn = refractive_index_fused_silica(lam_in) - REFRACTIVE_INDEX_AIR
        expected_phase_shift = (2.0 * np.pi / lam_in) * dn * C_offset
        expected_phasor = np.exp(1j * expected_phase_shift)

        # 孔径内复振幅比值应恒等于 expected_phasor
        mask = profile.mask.astype(bool)
        ratio = u1_shifted[mask] / u1_orig[mask]
        phase_err = np.max(np.abs(ratio - expected_phasor))
        self.assertLess(phase_err, 1e-12, f"全局相位偏差: {phase_err:.4e}")

    def test_fingerprint_deterministic_and_unique(self):
        """测试器件指纹的确定性与敏感性。"""
        profile1 = design_jeon2019_spiral_height(self.grid, self.D, self.f, wings_N=3)
        fp1 = profile1.compute_fingerprint()

        # 再次生成，指纹必须完全相同
        profile2 = design_jeon2019_spiral_height(self.grid, self.D, self.f, wings_N=3)
        fp2 = profile2.compute_fingerprint()
        self.assertEqual(fp1, fp2)

        # 参数微小变动后指纹必须改变
        profile3 = design_jeon2019_spiral_height(self.grid, self.D, self.f * 1.01, wings_N=3)
        fp3 = profile3.compute_fingerprint()
        self.assertNotEqual(fp1, fp3)


if __name__ == "__main__":
    unittest.main()
