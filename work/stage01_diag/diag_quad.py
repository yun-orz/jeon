# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 12（临时）：用输入网格上的直接求积与解析解、传递函数对照。

关键：直接求积必须在输入采样足够细（相位步进小）时才有意义。
"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as M                                          # noqa: E402
from optics.coordinates import Axis1D, Grid2D                            # noqa: E402
from optics.propagation import fresnel_kernel_eval                       # noqa: E402

lam, z, w0, tilt = 550e-9, 50e-3, 0.39e-3, 1e-3
k = 2 * np.pi / lam
lamz = lam * z
zr = np.pi * w0 ** 2 / lam
print("z_R = %.4g m, z/z_R = %.6g, k z tilt^2 = %.3e rad" % (zr, z / zr, k * z * tilt ** 2))

# 参考：高精度数值积分（scipy quad，1D×1D 分离式）
from scipy import integrate  # noqa: E402


def quad_point(xe, ye):
    def f(real_part):
        return None
    fr = lambda c: np.exp(-c ** 2 / w0 ** 2) * np.exp(1j * k * tilt * c) \
        * np.exp(1j * np.pi * (xe - c) ** 2 / lamz)
    gi = lambda c: np.exp(-c ** 2 / w0 ** 2) * np.exp(1j * np.pi * (ye - c) ** 2 / lamz)
    ir = integrate.quad(lambda c: fr(c).real, -6e-3, 6e-3, limit=50000, epsabs=1e-14, epsrel=1e-13)[0] \
        + 1j * integrate.quad(lambda c: fr(c).imag, -6e-3, 6e-3, limit=50000, epsabs=1e-14, epsrel=1e-13)[0]
    # y 方向解析：∫exp(-c²/w0²)exp(iπ(y-c)²/λz)dc
    c0 = ye / (1 - 1j * lamz / (np.pi * w0 ** 2))
    Iy = np.sqrt(np.pi) * w0 / np.sqrt(1 - 1j * lamz / (np.pi * w0 ** 2)) \
        * np.exp(-ye ** 2 / (w0 ** 2 * (1 - 1j * lamz / (np.pi * w0 ** 2))))
    pref = 1 / (1j * lamz) * np.exp(1j * k * z)
    return pref * ir * Iy


def analytic_point(xe, ye):
    pref = np.exp(1j * k * z) / (1 + 1j * z / zr)
    dx = xe - z * tilt
    return pref * np.exp(-(dx ** 2 + ye ** 2) / (w0 ** 2 * (1 + 1j * z / zr)))

for (xe, ye) in [(0.0, 0.0), (z * tilt, 0.0), (0.5e-3, 0.0)]:
    print("x=%9.2f um  quad=%s  analytic=%s" % (xe * 1e6, quad_point(xe, ye), analytic_point(xe, ye)))
