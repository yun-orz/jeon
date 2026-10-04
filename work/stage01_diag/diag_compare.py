# -*- coding: utf-8 -*-
"""阶段 01 调试脚本（临时）：并排比较 FFT 路径、卷积路径、直接求和与解析高斯解。

放在 work/ 下，不属于交付项目；交付项目只依赖 outputs/jeon2019_optics。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
sys.path.insert(0, str(ROOT))

from optics import metrics as M                       # noqa: E402
from optics.coordinates import Axis1D, Grid2D         # noqa: E402
from optics.propagation import (FresnelConfig, direct_fresnel_integral,  # noqa: E402
                                fresnel_kernel_eval, fresnel_propagate)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_case(n, dx, m, w0, tilt, z=50e-3, lam=550e-9):
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    u1 = M.tilted_gaussian_field(g, lam, w0, (0.0, 0.0), tilt)
    print("n=%d dx=%.3g um  M=%d  w0=%.3g mm  tilt=%.3g rad" % (n, dx * 1e6, m, w0 * 1e3, tilt[0]))
    print("  |u1|max=%.6g  输入边缘=%.3g  P_in=%.6e" % (
        np.abs(u1).max(), np.abs(u1[0, 0]), np.sum(np.abs(u1) ** 2) * dx * dx))

    ana_at = lambda gr: M.tilted_gaussian_paraxial_solution(gr, lam, z, w0, tilt)

    # 路径 1：单次 FFT
    res = fresnel_propagate(u1, g, FresnelConfig(lam, z, m))
    ana = ana_at(res.grid)
    print("  [FFT   ] |u2|max=%.6e  dx_out=%.4g um  relL2 vs 解析=%.4e  P_out=%.6e" % (
        np.abs(res.field).max(), res.grid.dx * 1e6,
        M.relative_l2_complex(res.field, ana), res.power))
    print("          解析 |u2|max=%.6e" % np.abs(ana).max())

    # 路径 2：卷积形式（冲激响应 FFT，无二次相位展开）
    P = 4 * n
    D = (P - n) // 2
    big = np.zeros((P, P), complex)
    big[D:D + n, D:D + n] = u1
    coord = (np.arange(P) - P // 2) * dx
    r2 = coord[:, None] ** 2 + coord[None, :] ** 2
    h = np.exp(1j * np.pi * r2 / (lam * z))
    Hk = np.fft.fft2(np.fft.ifftshift(h))
    Uk = np.fft.fft2(np.fft.ifftshift(big))
    conv = np.fft.fftshift(np.fft.ifft2(Uk * Hk))
    out = conv * (dx * dx / (1j * lam * z)) * np.exp(1j * 2 * np.pi / lam * z)
    sub = out[D:D + n, D:D + n]
    ana_native = ana_at(g)
    print("  [CONV  ] |u2|max=%.6e  relL2 vs 解析(原生网格)=%.4e" % (
        np.abs(sub).max(), M.relative_l2_complex(sub, ana_native)))

    # 路径 3：直接二维求积（同 FFT 输出网格）
    if n <= 64:
        d = direct_fresnel_integral(u1, g, lam, z, res.grid)
        print("  [DIRECT] |u2|max=%.6e  relL2 vs FFT=%.4e  relL2 vs 解析=%.4e" % (
            np.abs(d).max(), M.relative_l2_complex(res.field, d),
            M.relative_l2_complex(d, ana)))
    else:
        print("  [DIRECT] 跳过（n 过大）")

    # 点值比较（网格中心，x=0,y=0）
    q = res.grid.y.n // 2
    p = res.grid.x.n // 2
    print("  中心点：FFT=%s" % (res.field[q, p],))
    print("          解析=%s" % (ana[q, p],))
    ke = fresnel_kernel_eval(u1, g, lam, z,
                             np.array([res.grid.x.coords[p]]),
                             np.array([res.grid.y.coords[q]]))[0, 0]
    print("          直接=%s" % (ke,))


if __name__ == "__main__":
    section("情形 1：小窗口、粗采样（预期存在混叠/静止点问题）")
    run_case(n=128, dx=6e-3 / 128, m=512, w0=1e-3, tilt=(2e-3, 0.0))
    section("情形 2：小窗口、细采样")
    run_case(n=64, dx=2e-3 / 64, m=256, w0=0.4e-3, tilt=(1e-3, 0.0))
    section("情形 3：极小规模，直接求积可靠")
    run_case(n=32, dx=2e-3 / 32, m=128, w0=0.4e-3, tilt=(1e-3, 0.0))
