# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 4（临时）：轴向解析解 vs FFT vs 直接求和，逐步打印。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics.coordinates import (circular_aperture, grid_from_half_width,  # noqa: E402
                                plane_wave)
from optics.propagation import (FresnelConfig, fresnel_kernel_eval,       # noqa: E402
                                fresnel_propagate)

lam, z, D = 550e-9, 50e-3, 0.5e-3
N, half = 512, 2e-3
g = grid_from_half_width(half, N)
ap = circular_aperture(g, D)
u1 = plane_wave(g) * ap
dx = g.dx
print("dx = %.10g m   area = %.10g m^2   aperture samples = %d" % (dx, g.cell_area, ap.sum()))

# 解析：轴上精确式 u2(0,0) = exp(ikz)[1 - exp(i pi D^2/(4 lam z))]
exact = np.exp(1j * 2 * np.pi / lam * z) * (1 - np.exp(1j * np.pi * D ** 2 / (4 * lam * z)))
print("解析轴上 u2(0,0) =", exact)

# FFT 路径
res = fresnel_propagate(u1, g, FresnelConfig(lam, z, N))
q = res.grid.y.n // 2
p = res.grid.x.n // 2
print("输出中心坐标 = (%.6g, %.6g)" % (res.grid.x.coords[p], res.grid.y.coords[q]))
print("FFT  u2(0,0)  =", res.field[q, p])

# 直接求和（未填充输入）
ke = fresnel_kernel_eval(u1, g, lam, z, np.array([0.0]), np.array([0.0]))[0, 0]
print("直接 u2(0,0)  =", ke)

# 手工 FFT：同样输入、同样零填充、同样公式
padded = np.zeros((N, N), complex)
padded[:] = u1                       # N == M，无实际填充
chirp = np.exp(1j * np.pi * g.radius() ** 2 / (lam * z))
A = padded * chirp
print("sum(A)            =", A.sum())
U = np.fft.fft2(A)
print("U[0,0]            =", U[0, 0])
print("U[256,256]        =", U[N // 2, N // 2])
pref = g.cell_area / (1j * lam * z) * np.exp(1j * 2 * np.pi / lam * z)
print("pref              =", pref)
print("U[0,0]*pref       =", U[0, 0] * pref)
print("U[256,256]*pref   =", U[N // 2, N // 2] * pref)
