# -*- coding: utf-8 -*-
"""阶段 01 传播基线测试（修正版，标准库 unittest，无需额外依赖）。

运行::

    python -m unittest discover -s tests -v
    python -m pytest tests -v            # 若装了 pytest

测试设计原则（针对审核意见重写）
--------------------------------
1. **离散恒等**：同一离散输入、同一原生输出网格上，单次 FFT 必须与完整位移核求和
   逐点一致（目标 ≤1e-6，本机实测 ~1e-14）。这是能拦截"漏面积权重 / 频率未中心化 /
   重复输入二次相位"这类实现错误的断言。
2. **连续正确**：数值传播器与**解析解**比较（自由空间高斯、理想透镜焦平面 Airy）。
   只测解析函数自身不算证据。
3. **功率**：原生 FFT 全输出功率相对误差 ≤1e-10。
4. **接口语义**：``include_global_phase`` 只控制 ``exp(ikz)``；
   ``omit_output_quadratic_phase`` 返回的是另一种场表示，必须显式补回相位才能比较。
5. **非方形网格 / 奇数输入 / x-y 间距不同**都要覆盖。

所有长度单位为米（m）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optics import metrics as M                                        # noqa: E402
from optics import units as U                                          # noqa: E402
from optics.coordinates import (Axis1D, Grid2D, circular_aperture,      # noqa: E402
                                grid_from_half_width, ideal_thin_lens_phase,
                                make_grid_xy, plane_wave)
from optics.propagation import (FresnelConfig, fresnel_angular_spectrum,  # noqa: E402
                                fresnel_fft, fresnel_kernel_matrix,
                                fresnel_kernel_separable,
                                fresnel_transfer_fresnel,
                                path_difference_exact_m, path_difference_leading_m,
                                sampling_diagnostics, third_order_phase_error_rad)

LAM = 550e-9


# ------------------------------------------------------------------ helpers
def asymmetric_field(grid: Grid2D, lam: float) -> np.ndarray:
    """受限的非对称复振幅（椭圆高斯 + 斜向线性相位 + 弱立方相位）。"""
    X, Y = grid.meshgrid()
    hw = grid.x.half_width
    env = np.exp(-((X - 0.30 * hw) / (0.55 * hw)) ** 2
                 - ((Y + 0.22 * hw) / (0.30 * hw)) ** 2)
    k = 2.0 * np.pi / lam
    return (env * np.exp(1j * k * (1.3e-3 * X - 0.9e-3 * Y))
            * np.exp(1j * 2.0e7 * (X ** 3 + 0.5 * Y ** 3))).astype(np.complex128)


def gaussian_field(grid: Grid2D, lam: float, w0: float, tilt=(0.0, 0.0)) -> np.ndarray:
    X, Y = grid.meshgrid()
    k = 2.0 * np.pi / lam
    return (np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)
            * np.exp(1j * k * (X * tilt[0] + Y * tilt[1]))).astype(np.complex128)


def free_space_gaussian(grid: Grid2D, lam: float, z: float, w0: float,
                        tilt=(0.0, 0.0)) -> np.ndarray:
    """自由空间近轴高斯光束的解析复场（束腰在 z=0 平面、位于原点）。

    输入取 ``u_in = exp[−(x²+y²)/w0²]·exp[ik(θx·x + θy·y)]``，同一近轴 Fresnel
    模型下的**完整解析复场**为::

        q       = 1 + i·zλ/(π w0²)
        u(x,y,z) = exp(ikz)/q
                   · exp{ −[(x−zθx)² + (y−zθy)²] / (w0² q) }
                   · exp{ ik[θx·x + θy·y − z(θx²+θy²)/2] }

    要点（第二轮审核 S8）：

    1. 包络中心**随 z 平移到 (zθx, zθy)**。旧版本没有这个平移，独立审核实测
       在 θx=1 mrad、z=50 mm 时复场相对误差 0.48395；补上平移后为 5.88e−16。
       直觉：倾斜平面波携带的横向波矢使光斑在传播中沿几何方向漂移。
    2. 二次相位分母里的 ``q`` 与 ``z_R = πw0²/λ`` 写法等价：``1+iz/z_R = q``。
    3. 第三行的最后一项 ``−kzθ²/2`` 是沿倾斜轴传播距离 ``z·cosθ`` 与 ``z`` 之差
       （二阶小量）；``θ`` 是**近轴线性相位系数**，统一用它，
       不混入 ``tanθ``、``sinθ`` 的精确几何表达式。
    4. ``tilt=(0,0)`` 时退化为正入射解析高斯解（已有测试覆盖）。
    """
    X, Y = grid.meshgrid()
    k = 2.0 * np.pi / lam
    zr = np.pi * w0 ** 2 / lam
    q = 1.0 + 1j * z / zr
    tx, ty = float(tilt[0]), float(tilt[1])
    xs = X - z * tx                      # 包络横移 zθ
    ys = Y - z * ty
    return (np.exp(1j * k * z) / q
            * np.exp(-(xs ** 2 + ys ** 2) / (w0 ** 2 * q))
            * np.exp(1j * (k * (X * tx + Y * ty) - k * z * (tx ** 2 + ty ** 2) / 2.0))
            ).astype(np.complex128)


def lens_aperture(D, f, lam, grid):
    return plane_wave(grid) * circular_aperture(grid, D) * ideal_thin_lens_phase(grid, lam, f)


def rel_l2(a, b):
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2)) / np.sqrt(np.sum(np.abs(b) ** 2)))


def focal_line_via_fft(D, f, lam, W, dx, pad=4):
    """理想透镜焦平面中心截线：**完整位移核的可分离直接求值**。

    z=f 时透镜相位 ``exp(−iπr′²/(λf))`` 与传播展开式的输入二次相位
    ``exp(+iπr′²/(λf))`` 相消，焦平面是孔径的 Fourier 强度（圆孔给出 Airy）。
    这里用 :func:`optics.propagation.fresnel_kernel_separable`（与完整位移核数学等价）
    在任意细的输出坐标上求值，因此输出采样不受 FFT 原生网格 ``λz/(MΔx′)`` 的限制。
    对论文尺度 D=1 mm 尤其必要：FFT 路线要达到 20 样点/暗环需 M≈1.6e4（约 4 GB）。

    返回 ``(x, intensity_line, u1, None)``，``x`` 与 ``intensity_line`` 按半径升序，
    强度按轴上峰值归一化。
    """
    n = 2 * int(round(W / dx))
    if n % 2 == 0:
        n += 1                                   # 奇数输入，顺带覆盖原点对齐
    grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
    u1 = lens_aperture(D, f, lam, grid)
    r1 = M.airy_dark_ring_radii(lam, f, D, 1)[0]
    half = 5.0 * r1
    xo = np.linspace(-half, half, 1201)
    u2 = fresnel_kernel_separable(u1, grid, lam, f, xo, np.array([0.0]))[0]
    xs = np.abs(xo)
    line = np.abs(u2) ** 2
    order = np.argsort(xs)
    xs, line = xs[order], line[order]
    return xs, line / line.max(), u1, None


def first_local_minimum(x_asc, y, smooth=5):
    """从升序半径剖面上找**第一个有意义的**局部极小。

    两个必须的稳健化处理（都在本阶段实测踩过）：

    1. 偶数输出网格没有 r=0 的样点，中心两侧的对称点会造出数值伪极小；
    2. Airy 中心附近强度变化极慢（相邻样点差只有 1e-5 量级），浮点噪声就会
       造出一串假极小——实测在 1.4/1.7/2.2 μm 处各有一个。

    因此在平滑剖面上判"先降后升且相对下降超过 1%"，回到原始剖面取邻域最小值细化。
    """
    if len(y) < 3 * smooth:
        return float("nan")
    ys = np.convolve(y, np.ones(int(smooth)) / float(smooth), mode="same")
    n = len(y)
    for i in range(smooth, n - smooth):
        if ys[i] <= ys[i - 1] and ys[i] < ys[i + 1]:
            left = float(np.max(y[max(0, i - smooth):i]))
            right = float(np.max(y[i + 1:i + 1 + smooth]))
            if ys[i] < 0.99 * min(left, right):
                j0 = max(0, i - smooth // 2)
                j1 = min(n, i + smooth // 2 + 1)
                return float(x_asc[j0 + int(np.argmin(y[j0:j1]))])
    return float("nan")


# ================================================================== 坐标约定
class TestCoordinates(unittest.TestCase):
    def test_index_increases_with_coordinate(self):
        """索引增加 → x、y 都增加（与旧文档相反的说法已在修正版统一）。"""
        ax = Axis1D(8, 2.0)
        c = ax.coords
        self.assertTrue(np.all(np.diff(c) > 0))
        self.assertEqual(c[4], 0.0)
        self.assertAlmostEqual(c[0], -8.0)
        self.assertAlmostEqual(c[-1], 6.0)

    def test_grid_index_convention(self):
        g = grid_from_half_width(1.0, 8)
        X, Y = g.meshgrid()
        self.assertEqual(X.shape, (8, 8))
        self.assertAlmostEqual(X[3, 5], g.x.coords[5])
        self.assertAlmostEqual(Y[3, 5], g.y.coords[3])
        self.assertLess(Y[0, 0], Y[-1, 0])          # 行增加 → y 增加
        self.assertLess(X[0, 0], X[0, -1])          # 列增加 → x 增加

    def test_extent_center_vs_pixel(self):
        g = grid_from_half_width(1.0, 8)
        x0c, x1c, _, _ = g.extent_center_um
        x0p, x1p, _, _ = g.extent_pixel_um
        self.assertAlmostEqual((x1p - x0p) - (x1c - x0c), g.dx * 1e6)
        self.assertAlmostEqual(x0p, x0c - 0.5 * g.dx * 1e6)

    def test_units_roundtrip(self):
        self.assertAlmostEqual(U.to_um(U.um(3.5)), 3.5)
        np.testing.assert_allclose(U.to_um(np.array([1e-6, 2e-6])), [1.0, 2.0])


# ================================================================== 离散恒等
class TestDiscreteIdentity(unittest.TestCase):
    """同一离散输入、同一原生输出网格：FFT 必须等于完整位移核求和。"""

    def _check(self, n, dx, M, tag):
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = asymmetric_field(grid, LAM)
        res = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M))
        u2k = fresnel_kernel_matrix(u1, grid, LAM, 50e-3,
                                    res.grid.x.coords, res.grid.y.coords)
        r = rel_l2(res.field, u2k)
        self.assertLess(r, 1e-6, "%s：FFT 与完整核求和相对 L2 = %.3e" % (tag, r))
        return r

    def test_original_case_n48_dx40um_M96(self):
        """原 B 组参数（审核要求 2：离散恒等必须成立，即使连续采样不足）。"""
        self.assertLess(self._check(48, 40e-6, 96, "N48/dx40um/M96"), 1e-10)

    def test_case_n24_dx10um_M48(self):
        """原先被错误解释的算例（审核要求 3，不能接受 1e10 差异）。"""
        self.assertLess(self._check(24, 10e-6, 48, "N24/dx10um/M48"), 1e-10)

    def test_case_n64_dx5um_M256(self):
        self.assertLess(self._check(64, 5e-6, 256, "N64/dx5um/M256"), 1e-10)

    def test_fft_power_conservation(self):
        """原生 FFT 全输出功率相对误差 ≤1e-10（审核要求 4）。"""
        n, dx, M = 48, 40e-6, 96
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = asymmetric_field(grid, LAM)
        res = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M))
        p_in = float(np.sum(np.abs(u1) ** 2) * grid.cell_area)
        err = abs(res.power / p_in - 1.0)
        self.assertLess(err, 1e-10, "原生 FFT 功率相对误差 %.3e" % err)

    def test_kernel_matrix_matches_explicit_2d_sum(self):
        """矩阵形式与显式二维求和一致（独立求积自身的交叉核对）。"""
        n, dx, z = 10, 12e-6, 50e-3
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        rng = np.random.default_rng(3)
        u1 = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        xo = np.array([-30e-6, 0.0, 25e-6])
        yo = np.array([-20e-6, 0.0, 40e-6])
        mat = fresnel_kernel_matrix(u1, grid, LAM, z, xo, yo)
        xp, yp = grid.x.coords, grid.y.coords
        explicit = np.zeros_like(mat)
        pref = grid.cell_area / (1j * LAM * z) * np.exp(1j * 2 * np.pi / LAM * z)
        for q, y in enumerate(yo):
            for p, x in enumerate(xo):
                s = 0.0 + 0.0j
                for jj in range(n):
                    for ii in range(n):
                        s += u1[jj, ii] * np.exp(
                            1j * np.pi * ((x - xp[ii]) ** 2 + (y - yp[jj]) ** 2) / (LAM * z))
                explicit[q, p] = s * pref
        self.assertLess(rel_l2(mat, explicit), 1e-12)

    def test_non_square_and_different_spacing(self):
        """非方形网格、x/y 间距不同（审核要求 7）。"""
        nx, ny, dx, dy = 40, 24, 8e-6, 15e-6
        grid = make_grid_xy(nx, ny, dx, dy)
        u1 = asymmetric_field(grid, LAM)
        M = (96, 64)                       # (My, Mx)，都为偶数
        res = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M))
        self.assertEqual(res.field.shape, (M[0], M[1]))
        u2k = fresnel_kernel_matrix(u1, grid, LAM, 50e-3,
                                    res.grid.x.coords, res.grid.y.coords)
        self.assertLess(rel_l2(res.field, u2k), 1e-6)
        self.assertAlmostEqual(res.grid.dx, LAM * 50e-3 / (M[1] * dx), places=18)
        self.assertAlmostEqual(res.grid.dy, LAM * 50e-3 / (M[0] * dy), places=18)

    def test_odd_input_supported_and_origin_aligned(self):
        """奇数输入：输出坐标仍以 0 为中心且含原点（审核要求 7）。"""
        n, dx = 45, 6e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = asymmetric_field(grid, LAM)
        res = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, 96))
        self.assertTrue(np.any(np.abs(res.grid.x.coords) < 1e-18))
        u2k = fresnel_kernel_matrix(u1, grid, LAM, 50e-3,
                                    res.grid.x.coords, res.grid.y.coords)
        self.assertLess(rel_l2(res.field, u2k), 1e-6)


# ================================================================== 连续正确性
class TestContinuousAccuracy(unittest.TestCase):
    def test_numeric_vs_analytic_free_gaussian(self):
        """数值传播器 vs 连续自由空间高斯解析复场（审核要求 5）。"""
        lam, z, w0 = LAM, 50e-3, 100e-6
        n, dx = 600, 5e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = gaussian_field(grid, lam, w0)                     # 正入射（无倾斜）
        self.assertLess(float(np.abs(u1[0, 0])), 1e-6)         # 充分支撑
        ana = free_space_gaussian(grid, lam, z, w0)
        num = fresnel_kernel_matrix(u1, grid, lam, z, grid.x.coords, grid.y.coords)
        r = rel_l2(num, ana)
        self.assertLess(r, 1e-6, "直接核求和 vs 解析高斯相对 L2 = %.3e" % r)

    def test_tilted_gaussian_full_complex_field(self):
        """S8：非零倾斜高斯必须与**完整解析复场**（包络平移到 zθ）逐点一致。

        审核实测：旧参考式（包络固定在原点）在 θx=1 mrad、z=50 mm 时复场误差
        0.48395；补上平移后 5.88e−16。这里阈值取 1e−6。
        """
        lam, z, w0 = LAM, 50e-3, 100e-6
        n, dx = 600, 5e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        for tilt in ((1.0e-3, 0.0), (0.0, 1.0e-3), (1.0e-3, -0.7e-3)):
            with self.subTest(tilt=tilt):
                u1 = gaussian_field(grid, lam, w0, tilt=tilt)
                ana = free_space_gaussian(grid, lam, z, w0, tilt=tilt)
                num = fresnel_kernel_separable(u1, grid, lam, z,
                                               grid.x.coords, grid.y.coords)
                r = rel_l2(num, ana)
                self.assertLess(r, 1e-6,
                                "tilt=%r：直接核 vs 完整解析复场相对 L2 = %.3e"
                                % (tilt, r))

    def test_tilted_gaussian_envelope_is_shifted(self):
        """包络中心必须落在 (z·θx, z·θy)，而不是原点（防回归）。"""
        lam, z, w0 = LAM, 50e-3, 100e-6
        n, dx = 400, 5e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        tx, ty = 1.0e-3, -0.5e-3
        ana = free_space_gaussian(grid, lam, z, w0, tilt=(tx, ty))
        X, Y = grid.meshgrid()
        peak = np.unravel_index(int(np.argmax(np.abs(ana))), ana.shape)
        self.assertLess(abs(X[peak] - z * tx), 3 * dx)
        self.assertLess(abs(Y[peak] - z * ty), 3 * dx)
        # θ=0 必须退化为原有正确解
        zr = np.pi * w0 ** 2 / lam
        ref = (np.exp(1j * 2 * np.pi / lam * z) / (1.0 + 1j * z / zr)
               * np.exp(-(X ** 2 + Y ** 2) / (w0 ** 2 * (1.0 + 1j * z / zr))))
        self.assertLess(rel_l2(free_space_gaussian(grid, lam, z, w0), ref), 1e-14)

    def test_tilted_gaussian_analytic_matches_direct_kernel_full_field(self):
        """整幅复场对照（含绝对幅度与相位），比只核峰值更强。"""
        lam, z, w0 = LAM, 50e-3, 100e-6
        n, dx = 300, 6e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        tilt = (8.0e-4, 6.0e-4)
        u1 = gaussian_field(grid, lam, w0, tilt=tilt)
        num = fresnel_kernel_separable(u1, grid, lam, z, grid.x.coords, grid.y.coords)
        ana = free_space_gaussian(grid, lam, z, w0, tilt=tilt)
        self.assertLess(rel_l2(num, ana), 1e-6)
        # 绝对幅度也必须对（不能只对相位）
        self.assertLess(abs(float(np.max(np.abs(num))) / float(np.max(np.abs(ana))) - 1.0),
                        1e-6)

    def test_fft_vs_analytic_free_gaussian_on_native_grid(self):
        """单次 FFT 的原生输出网格上也与解析解比较（补足填充避免回绕）。"""
        lam, z, w0 = LAM, 50e-3, 100e-6
        n, dx = 600, 5e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = gaussian_field(grid, lam, w0)
        res = fresnel_fft(u1, grid, FresnelConfig(lam, z, (2048, 2048)))
        # 只在输入网格覆盖的中心区域比较（解析解在该区域内有效）
        sel_x = np.abs(res.grid.x.coords) <= 1.5e-3
        sel_y = np.abs(res.grid.y.coords) <= 1.5e-3
        sub = res.field[np.ix_(sel_y, sel_x)]
        gsub = Grid2D(Axis1D(int(sel_x.sum()), res.grid.dx, "x"),
                      Axis1D(int(sel_y.sum()), res.grid.dy, "y"))
        self.assertLess(rel_l2(sub, free_space_gaussian(gsub, lam, z, w0)), 1e-6)

    def test_lens_focal_plane_peak_matches_discrete_pupil(self):
        """z=f 时焦平面是孔径 Fourier 强度：峰值对离散孔径理论。"""
        lam, f, D = LAM, 50e-3, 0.1e-3
        n, dx = 401, 1e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = lens_aperture(D, f, lam, grid)
        res = fresnel_fft(u1, grid, FresnelConfig(lam, f, (1600, 1600)))
        q, p = res.grid.y.n // 2, res.grid.x.n // 2
        peak = float(res.intensity[q, p])
        theory = (float(np.sum(np.abs(u1))) * grid.cell_area / (lam * f)) ** 2
        self.assertLess(abs(peak / theory - 1.0), 1e-3,
                        "焦平面峰值/离散孔径理论 = %.6f" % (peak / theory))

    def test_airy_central_line_three_apertures(self):
        """三个孔径/焦距组合的焦平面中心截线对解析 Airy（含 D=1 mm 论文尺度）。"""
        cases = [(0.1e-3, 50e-3, 200e-6, 1e-6, 6),
                 (0.2e-3, 25e-3, 200e-6, 1e-6, 6),
                 (1.0e-3, 50e-3, 600e-6, 1e-6, 4)]
        for D, f, W, dx, pad in cases:
            with self.subTest(D_mm=D * 1e3, f_mm=f * 1e3):
                x, line_n, _u1, _res = focal_line_via_fft(D, f, LAM, W, dx, pad=pad)
                airy = M.airy_intensity(x, LAM, f, D)
                l1 = float(np.sum(np.abs(line_n - airy)) / np.sum(airy))
                r1 = M.airy_dark_ring_radii(LAM, f, D, 1)[0]
                r_meas = first_local_minimum(x, line_n)
                rel_ring = abs(r_meas - r1) / r1
                self.assertLess(l1, 0.01, "D=%.3g: 中心截线 L1=%.3e" % (D, l1))
                self.assertLess(rel_ring, 0.02,
                                "D=%.3g: 第一暗环实测 %.4f μm / 理论 %.4f μm，相对误差 %.3e"
                                % (D, r_meas * 1e6, r1 * 1e6, rel_ring))

    def test_paper_scale_first_ring_reference(self):
        r1 = M.airy_dark_ring_radii(550e-9, 50e-3, 1e-3, 1)[0]
        self.assertAlmostEqual(U.to_um(r1), 33.540922, places=5)
        self.assertAlmostEqual(U.to_um(M.airy_first_dark_ring_radius(550e-9, 50e-3, 1e-3)),
                               33.55, places=2)


# ================================================================== 采样收敛
class TestSamplingConvergence(unittest.TestCase):
    def test_three_input_levels_fixed_pupil_and_window(self):
        """固定物理孔径与观察窗口，改变输入采样精度（审核要求 8）。"""
        D, f, lam = 0.1e-3, 50e-3, LAM
        W = 200e-6
        profiles = []
        for n in (200, 400, 800):
            dx = 2 * W / n
            x, line, _u, _r = focal_line_via_fft(D, f, lam, W, dx, pad=6)
            profiles.append((x, line))
        # 不同 n → 不同输出坐标；用共同坐标上的解析 Airy 作桥梁比较 L1
        l1 = []
        for x, line in profiles:
            a = M.airy_intensity(x, lam, f, D)
            l1.append(float(np.sum(np.abs(line - a)) / np.sum(a)))
        self.assertLess(abs(l1[-1] - l1[-2]), 5e-3,
                        "最后两级输入采样的 L1 差 %.3e" % abs(l1[-1] - l1[-2]))
        self.assertLess(max(l1), 0.01)

    def test_two_padding_factors(self):
        """S2 要求：``test_two_padding_factors`` 必须**实际执行 FFT**，且参数真的改变 M。

        旧版调用的辅助函数完全忽略 ``pad``，两次跑的是同一个直接积分，因此这个测试
        不能承担"输出采样/填充收敛"的验收。这里改为：
        固定同一离散输入，用两个**真正不同**的 FFT 长度 M，检查 (a) 原生输出间距按
        ``λz/(MΔx′)`` 改变，(b) 每次都与完整位移核在该原生网格上离散恒等，
        (c) 两个原生强度映射到同一公共探测器网格后对公共坐标完整核参考的原始 L1
        随 M 增大而不变差。
        """
        D, f, lam = 0.1e-3, 50e-3, LAM
        W, dx_in = 200e-6, 1e-6
        n = 2 * int(round(W / dx_in)) + 1
        grid = Grid2D(Axis1D(n, dx_in, "x'"), Axis1D(n, dx_in, "y'"))
        u1 = lens_aperture(D, f, lam, grid)
        r1 = M.airy_dark_ring_radii(lam, f, D, 1)[0]
        out_dx = 2e-6
        k_out = 2 * int(np.ceil(2.0 * r1 / out_dx)) + 1
        x_common = (np.arange(k_out, dtype=np.float64) - k_out // 2) * out_dx
        dA = out_dx ** 2
        u_ref = fresnel_kernel_separable(u1, grid, lam, f, x_common, x_common)
        I_ref = np.abs(u_ref) ** 2

        records = []
        for m_fft in (1024, 2048):
            res = fresnel_fft(u1, grid, FresnelConfig(lam, f, m_fft))
            # (a) 原生输出间距必须按公式改变
            self.assertAlmostEqual(res.grid.dx, lam * f / (m_fft * dx_in), places=18)
            self.assertNotAlmostEqual(res.grid.dx,
                                      lam * f / ((m_fft * 2) * dx_in), places=18)
            # (b) 原生网格上的离散恒等（归入算法检查）
            ref_native = fresnel_kernel_matrix(u1, grid, lam, f,
                                               res.grid.x.coords, res.grid.y.coords)
            self.assertLess(rel_l2(res.field, ref_native), 1e-6)
            # (c) 强度映射到公共网格后对公共参考的原始 L1
            I_map = M.resample_intensity_bilinear(np.abs(res.field) ** 2, res.grid,
                                                 x_common, x_common)
            mask = np.isfinite(I_map)
            a = np.where(mask, I_map, 0.0)
            b = np.where(mask, I_ref, 0.0)
            l1 = float(np.sum(np.abs(a - b)) * dA / (np.sum(b) * dA))
            records.append({"M": m_fft, "native_dx": res.grid.dx, "L1": l1,
                            "points": int(mask.sum())})
        self.assertLess(records[-1]["L1"], 0.02,
                        "最细 M 对公共参考的原始强度 L1 = %.4e" % records[-1]["L1"])
        self.assertLessEqual(records[-1]["L1"], records[0]["L1"] * 1.5,
                             "更细填充不应显著变差")


# ================================================================== 光程诊断
class TestParaxialDiagnostics(unittest.TestCase):
    def test_exact_vs_leading_path_difference(self):
        z = 50e-3
        rho = np.array([1e-4, 1e-3, 1.25e-3])
        exact = path_difference_exact_m(rho, z)
        lead = path_difference_leading_m(rho, z)
        self.assertTrue(np.all(exact < 0))
        self.assertLess(float(np.max(np.abs(exact - lead) / np.abs(exact))), 1e-2)

    def test_known_phase_error_value(self):
        """审核基准：ρ=1.25 mm、z=50 mm、λ=550 nm → 0.02788185 rad。"""
        self.assertAlmostEqual(third_order_phase_error_rad(1.25e-3, 50e-3, 550e-9),
                               0.02788184738292828, places=9)
        lead = float(abs(2 * np.pi / 550e-9 * path_difference_leading_m(1.25e-3, 50e-3)))
        self.assertAlmostEqual(lead, 0.02789055977973892, places=9)

    def test_sampling_diagnostics_covers_three_terms(self):
        grid = Grid2D(Axis1D(400, 1e-6, "x'"), Axis1D(400, 1e-6, "y'"))
        d = sampling_diagnostics(grid, LAM, 50e-3, 1.2e-3, 1.2e-3,
                                 dx_out=3e-6, dy_out=3e-6)
        for key in ("input_quadratic_step_rad", "cross_term_step_rad",
                    "output_quadratic_step_rad", "rho_max_m",
                    "third_order_phase_error_rad", "ok",
                    "input_kernel_step_x_rad", "input_kernel_step_y_rad",
                    "output_quadratic_step_x_rad", "output_quadratic_step_y_rad"):
            self.assertIn(key, d)
        self.assertGreater(d["cross_term_step_rad"], 0.0)
        self.assertAlmostEqual(d["rho_max_m"],
                               float(np.hypot(200e-6 + 1.2e-3, 200e-6 + 1.2e-3)), places=12)
        # 半宽必须按 (n//2)*d，不能额外乘 0.5
        self.assertAlmostEqual(d["input_half_width_m"], 200 * 1e-6, places=15)

    def test_sampling_diagnostics_uses_output_spacing_not_input(self):
        """S7：输出二次相位步进必须用**实际输出间距**，并且能标记"未评估"。"""
        grid = Grid2D(Axis1D(400, 1e-6, "x'"), Axis1D(400, 1e-6, "y'"))
        # 不给输出间距 → 输出方向必须报"未评估"，且不得用输入间距冒充
        d0 = sampling_diagnostics(grid, LAM, 50e-3, 1.2e-3, 1.2e-3)
        self.assertIsNone(d0["output_quadratic_step_x_rad"])
        self.assertIsNone(d0["output_quadratic_step_y_rad"])
        self.assertFalse(d0["output_quadratic_evaluated"])
        self.assertIn("未评估", d0["output_representation_formula"])
        # 改变输出间距 → 输出步进必须跟随改变
        d1 = sampling_diagnostics(grid, LAM, 50e-3, 1.2e-3, 1.2e-3,
                                  dx_out=2e-6, dy_out=2e-6)
        d2 = sampling_diagnostics(grid, LAM, 50e-3, 1.2e-3, 1.2e-3,
                                  dx_out=8e-6, dy_out=8e-6)
        self.assertGreater(d2["output_quadratic_step_x_rad"],
                           d1["output_quadratic_step_x_rad"] * 3.9)
        expect = np.pi * (2 * 1.2e-3 * 8e-6 + (8e-6) ** 2) / (LAM * 50e-3)
        self.assertAlmostEqual(d2["output_quadratic_step_x_rad"], float(expect), places=12)
        # 输入积分方向不得被输出间距影响
        self.assertAlmostEqual(d1["input_kernel_step_x_rad"],
                               d2["input_kernel_step_x_rad"], places=15)

    def test_sampling_diagnostics_rectangular_dy_larger(self):
        """S7：矩形网格且 dy > dx 时，y 方向必须被单独检出（不能漏）。"""
        g = Grid2D(Axis1D(100, 1e-6, "x'"), Axis1D(100, 8e-6, "y'"))
        d = sampling_diagnostics(g, LAM, 50e-3, 1.0e-3, 1.0e-3,
                                 dx_out=2e-6, dy_out=2e-6)
        self.assertGreater(d["input_kernel_step_y_rad"], d["input_kernel_step_x_rad"] * 5)
        self.assertAlmostEqual(d["input_kernel_step_worst_rad"],
                               d["input_kernel_step_y_rad"], places=15)
        self.assertEqual(d["ok"],
                         d["input_kernel_step_worst_rad"] <= d["threshold_rad"])
        # 保守核界公式：π[2(a+b)Δ+Δ²]/(λz)
        a = (100 // 2) * 8e-6
        expect_y = np.pi * (2 * (a + 1.0e-3) * 8e-6 + (8e-6) ** 2) / (LAM * 50e-3)
        self.assertAlmostEqual(d["input_kernel_step_y_rad"], float(expect_y), places=12)
        self.assertTrue(d["support_is_conservative_upper_bound"])

    def test_sampling_diagnostics_output_not_used_as_input_term(self):
        """S7：输出二次相位是常量项，绝不能当作输入求积的变化项。"""
        g = Grid2D(Axis1D(200, 2e-6, "x'"), Axis1D(200, 2e-6, "y'"))
        a = sampling_diagnostics(g, LAM, 50e-3, 0.5e-3, 0.5e-3, dx_out=5e-6, dy_out=5e-6)
        self.assertIn("对求和变量没有步进", a["output_quadratic_note"])
        # 输入项与输出项是两个独立字段，不相等（否则就是混用了）
        self.assertNotAlmostEqual(a["input_kernel_step_x_rad"],
                                  a["output_quadratic_step_x_rad"], places=9)


# ================================================================== 接口语义
class TestInterfaceSemantics(unittest.TestCase):
    def test_global_phase_switch_only_controls_exp_ikz(self):
        n, dx, M = 24, 10e-6, 48
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = asymmetric_field(grid, LAM)
        a = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M, include_global_phase=True))
        b = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M, include_global_phase=False))
        expect = np.exp(1j * 2 * np.pi / LAM * 50e-3)
        np.testing.assert_allclose(a.field / b.field, np.full(b.field.shape, expect),
                                   rtol=1e-12)

    def test_omit_output_quadratic_phase_semantics(self):
        """**reduced 场 = C·F**（未乘输出二次相位），与完整场相差**一次** ``exp(+iq)``。

        第二轮审核 S1 指出旧实现乘了 ``exp(−iq)``，使两者相差 ``exp(−2iq)``。
        本测试用**全场范数**断言准确关系，不逐点除场（避免暗点不稳定），
        也不只断言“比值模为 1”（任意错误的纯相位都能通过那种弱测试）。
        """
        n, dx, M = 24, 10e-6, 48
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = asymmetric_field(grid, LAM)
        full = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M))
        red = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M,
                                                  omit_output_quadratic_phase=True))
        self.assertIn("reduced", red.meta["representation"])
        self.assertFalse(red.meta["output_quadratic_phase_present"])
        x, y = red.grid.x.coords, red.grid.y.coords
        q = np.pi * (x[None, :] ** 2 + y[:, None] ** 2) / (LAM * 50e-3)
        self.assertLess(rel_l2(red.field * np.exp(1j * q), full.field), 1e-10)
        self.assertLess(rel_l2(red.field, full.field * np.exp(-1j * q)), 1e-10)
        # 纯相位变化不改变本平面强度（用强度比，不逐点除场）
        self.assertLess(rel_l2(np.abs(red.field) ** 2, np.abs(full.field) ** 2), 1e-12)
        # 反例：补两次相位（旧实现的口径）必须**不**成立
        self.assertGreater(rel_l2(red.field * np.exp(2j * q), full.field), 0.1)

    def test_reduced_full_non_square_and_global_phase_combo(self):
        """非方形 dx≠dy 网格 + 全局相位开关与本开关同时使用（S1 要求）。"""
        nx, ny = 20, 26
        dx, dy, M = 8e-6, 15e-6, (64, 48)
        grid = make_grid_xy(nx, ny, dx, dy)
        u1 = asymmetric_field(grid, LAM)
        for gp in (True, False):
            full = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M,
                                                       include_global_phase=gp))
            red = fresnel_fft(u1, grid, FresnelConfig(LAM, 50e-3, M,
                                                      include_global_phase=gp,
                                                      omit_output_quadratic_phase=True))
            x, y = red.grid.x.coords, red.grid.y.coords
            q = np.pi * (x[None, :] ** 2 + y[:, None] ** 2) / (LAM * 50e-3)
            self.assertLess(rel_l2(red.field * np.exp(1j * q), full.field), 1e-10)
            self.assertLess(rel_l2(np.abs(red.field), np.abs(full.field)), 1e-12)

    def test_spectral_global_phase_switch_both_methods(self):
        """S5：两种频域传播的 ``include_global_phase`` 都必须真的生效。

        验收：(a) 单位均匀平面波不填充时关开关应得到 1、开开关得到 exp(ikz)；
        (b) 非对称输入下 ``u_true ≈ exp(ikz)·u_false``（相对 L2 ≤1e−10）。
        """
        n, dx = 32, 20e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        pw = np.ones((n, n), dtype=np.complex128)
        expect = np.exp(1j * 2 * np.pi / LAM * 50e-3)
        for name, fn in (("Fresnel", fresnel_transfer_fresnel),
                         ("角谱", fresnel_angular_spectrum)):
            on = fn(pw, grid, LAM, 50e-3, include_global_phase=True)
            off = fn(pw, grid, LAM, 50e-3, include_global_phase=False)
            with self.subTest(method=name):
                self.assertLess(float(np.max(np.abs(off.field - 1.0))), 1e-10,
                                "%s：关开关后单位平面波应为 1" % name)
                self.assertLess(rel_l2(on.field, np.full((n, n), expect)), 1e-10,
                                "%s：开开关后应等于 exp(ikz)" % name)
                self.assertLess(rel_l2(on.field, expect * off.field), 1e-10)
                self.assertFalse(off.meta["include_global_phase"])
                # 强度不得改变
                self.assertLess(rel_l2(np.abs(on.field), np.abs(off.field)), 1e-12)

        u1 = asymmetric_field(grid, LAM)
        for name, fn in (("Fresnel", fresnel_transfer_fresnel),
                         ("角谱", fresnel_angular_spectrum)):
            on = fn(u1, grid, LAM, 50e-3, include_global_phase=True)
            off = fn(u1, grid, LAM, 50e-3, include_global_phase=False)
            with self.subTest(method=name, case="非对称输入"):
                self.assertLess(rel_l2(on.field, expect * off.field), 1e-10)

    def test_spectral_forms_are_distinct(self):
        """频域 Fresnel 近似与精确角谱必须区分名称。"""
        n, dx = 32, 20e-6
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = gaussian_field(grid, LAM, 100e-6)
        fr = fresnel_transfer_fresnel(u1, grid, LAM, 50e-3, padded_size=64)
        ang = fresnel_angular_spectrum(u1, grid, LAM, 50e-3, padded_size=64)
        self.assertIn("Fresnel", fr.meta["formulation"])
        self.assertIn("角谱", ang.meta["formulation"])
        self.assertNotEqual(fr.meta["formulation"], ang.meta["formulation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


