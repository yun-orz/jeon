# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 17（临时）：从零对照 —— 单次 FFT(IR) 形式 vs 直接求和。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as MM                                       # noqa: E402
from optics.coordinates import Axis1D, Grid2D                          # noqa: E402
from optics.propagation import fresnel_kernel_eval                     # noqa: E402

lam, z = 550e-9, 50e-3
lamz = lam * z
k = 2 * np.pi / lam


def ir_from_scratch(u1, dx, M):
    n = u1.shape[0]
    o = (M - n) // 2
    big = np.zeros((M, M), complex)
    big[o:o + n, o:o + n] = u1
    c = (np.arange(M) - M // 2) * dx
    C = c[:, None] ** 2 + c[None, :] ** 2
    A = big * np.exp(1j * np.pi * C / lamz)
    U = np.fft.fft2(A)
    dxo = lamz / (M * dx)
    co = (np.arange(M) - M // 2) * dxo
    Co = co[:, None] ** 2 + co[None, :] ** 2
    return U * np.exp(1j * np.pi * Co / lamz) * np.exp(1j * k * z) / (1j * lamz), dxo


n, M, dx = 48, 80, 40e-6
g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
X, Y = g.meshgrid()
u1 = (np.exp(-(X / (0.25 * g.x.half_width)) ** 2 - (Y / (0.15 * g.y.half_width)) ** 2)
      * np.exp(1j * k * 1.0e-3 * X)).astype(complex)
print("边缘|u1| = %.3g" % abs(u1[0, 0]))

field_ir, dxo = ir_from_scratch(u1, dx, M)
print("IR 输出间距 = %.4g um" % (dxo * 1e6))

d = fresnel_kernel_eval(u1, g, lam, z, (np.arange(M) - M // 2) * dxo,
                        (np.arange(M) - M // 2) * dxo)
rel = MM.relative_l2_complex(field_ir, d)
print("rel L2 IR vs 直接 = %.4e" % rel)
print("中心点：IR=%s 直接=%s" % (field_ir[M // 2, M // 2], d[M // 2, M // 2]))
print("比值 =", field_ir[M // 2, M // 2] / d[M // 2, M // 2])
print("|IR|max=%.6g  |直接|max=%.6g" % (np.abs(field_ir).max(), np.abs(d).max()))
print("1/(lam z) = %.6g ; 1/(lam z)^2 = %.6g" % (1 / lamz, 1 / lamz ** 2))
