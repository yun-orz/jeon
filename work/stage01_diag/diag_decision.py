# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 3（临时）：判定 FFT 路径到底在算什么。

检验：
1. 直接求积在**同一个包含零填充的离散输入**上，是否与 FFT 逐点一致；
2. 在解析高斯解已经充分采样的参数下，FFT 路径是否与解析解一致；
3. 单次 FFT 与连续积分的差异随 Δx²/(λz) 的变化。
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as M                                     # noqa: E402
from optics.coordinates import Axis1D, Grid2D                       # noqa: E402
from optics.propagation import (FresnelConfig, fresnel_kernel_eval,  # noqa: E402
                                fresnel_propagate)

lam, z = 550e-9, 50e-3


def test1_padded_equivalence():
    print("=" * 78)
    print("检验 1：直接求积作用于同一零填充离散输入时，是否与单次 FFT 逐点一致")
    print("=" * 78)
    n, m, dx = 24, 40, 60e-6
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    rng = np.random.default_rng(7)
    u1 = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    res = fresnel_propagate(u1, g, FresnelConfig(lam, z, m))
    # 把输入放到"含零填充"的离散网格上，用同一公式直接求和
    gp = Grid2D(Axis1D(m, dx, "x"), Axis1D(m, dx, "y"))
    up = np.zeros((m, m), complex)
    up[(m - n) // 2:(m - n) // 2 + n, (m - n) // 2:(m - n) // 2 + n] = u1
    xs = res.grid.x.coords
    ys = res.grid.y.coords
    # 逐点直接求和（网格较大，用矩阵形式，只算少量点）
    picks = [(0, 0), (m // 2, m // 2), (m // 2 + 3, m // 2 - 5), (m - 1, m - 1)]
    chirp_p = np.exp(1j * np.pi * gp.radius() ** 2 / (lam * z))
    src = up * chirp_p
    pref = gp.cell_area / (1j * lam * z) * np.exp(1j * 2 * np.pi / lam * z)
    for (p, q) in picks:
        x, y = xs[p], ys[q]
        px = np.exp(1j * np.pi * (x - gp.x.coords) ** 2 / (lam * z))
        py = np.exp(1j * np.pi * (y - gp.y.coords) ** 2 / (lam * z))
        val = np.einsum("n,nm,m->", py, src, px) * pref
        print("  (p,q)=(%2d,%2d) 直接=%-28s FFT=%-28s rel=%.3e" % (
            p, q, val, res.field[q, p], abs(val - res.field[q, p]) / abs(val)))
    # 未填充输入的直接求积（对照）
    picks2 = [0, m // 2, m - 1]
    ke = fresnel_kernel_eval(u1, g, lam, z, xs[picks2], ys[picks2])
    print("  未填充输入的直接求积（对照，见检验 3 的物理解释）：")
    for i, p in enumerate(picks2):
        print("    fft=%s  direct(unpadded)=%s" % (res.field[p, p], ke[i, i]))


def test2_analytic(n, m, half, w0, tilt):
    print("=" * 78)
    print("检验 2：解析高斯解对照  n=%d M=%d half=%.3g mm w0=%.3g mm tilt=%.3g" % (
        n, m, half * 1e3, w0 * 1e3, tilt))
    print("=" * 78)
    dx = 2 * half / n
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    u1 = M.tilted_gaussian_field(g, lam, w0, (0.0, 0.0), (tilt, 0.0))
    res = fresnel_propagate(u1, g, FresnelConfig(lam, z, m))
    ana = M.tilted_gaussian_paraxial_solution(res.grid, lam, z, w0, (tilt, 0.0))
    print("  |u1|max=%.4g 边缘=%.3g  B_in=Δx²/(λz)=%.3g  Δx=%.4g um  输出Δx=%.4g um"
          % (np.abs(u1).max(), np.abs(u1[0, 0]), dx ** 2 / (lam * z), dx * 1e6,
             res.grid.dx * 1e6))
    print("  FFT  |u2|max=%.6g  解析 |u2|max=%.6g" % (
        np.abs(res.field).max(), np.abs(ana).max()))
    print("  relL2(FFT, 解析) = %.6e" % M.relative_l2_complex(res.field, ana))
    q = res.grid.y.n // 2
    p = res.grid.x.n // 2
    print("  中心点 FFT=%s  解析=%s" % (res.field[q, p], ana[q, p]))
    return res, ana


def test3_scaling():
    print("=" * 78)
    print("检验 3：固定物理问题、改变 Δx（是否满足 M ≥ 2N），看与解析解的一致性")
    print("=" * 78)
    half, w0, tilt = 1.5e-3, 4e-4, 1e-3
    for n, m in [(32, 128), (64, 128), (64, 256), (128, 256), (128, 512), (256, 1024)]:
        dx = 2 * half / n
        g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
        u1 = M.tilted_gaussian_field(g, lam, w0, (0.0, 0.0), (tilt, 0.0))
        res = fresnel_propagate(u1, g, FresnelConfig(lam, z, m))
        ana = M.tilted_gaussian_paraxial_solution(res.grid, lam, z, w0, (tilt, 0.0))
        print("  N=%3d M=%4d Δx=%7.3f um  M/N=%4.1f  B=Δx²/(λz)=%6.3g  relL2 vs 解析=%.4e"
              % (n, m, dx * 1e6, m / n, dx ** 2 / (lam * z),
                 M.relative_l2_complex(res.field, ana)))


if __name__ == "__main__":
    test1_padded_equivalence()
    test2_analytic(64, 256, 1.5e-3, 4e-4, 1e-3)
    test3_scaling()
