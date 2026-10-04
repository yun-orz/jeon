# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 6（临时）：从零重写单次 FFT 传播，逐项对照库函数。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics.coordinates import grid_from_half_width, plane_wave, circular_aperture  # noqa: E402
from optics.propagation import FresnelConfig, fresnel_propagate                     # noqa: E402

lam, z, D = 550e-9, 50e-3, 0.5e-3
N, half = 512, 2e-3
g = grid_from_half_width(half, N)
u1 = plane_wave(g) * circular_aperture(g, D)
dx = g.dx
lamz = lam * z
M = N
coords = (np.arange(M) - M // 2) * dx
R2_in = coords[:, None] ** 2 + coords[None, :] ** 2

A = u1 * np.exp(1j * np.pi * R2_in / lamz)
U = np.fft.fft2(A)

dxo = lamz / (M * dx)
cout = (np.arange(M) - M // 2) * dxo
R2_out = cout[:, None] ** 2 + cout[None, :] ** 2
field = U * np.exp(1j * np.pi * R2_out / lamz) * np.exp(1j * 2 * np.pi / lam * z) / (1j * lamz)

q = p = M // 2
exact = np.exp(1j * 2 * np.pi / lam * z) * (1 - np.exp(1j * np.pi * D ** 2 / (4 * lamz)))
print("自写实现      field(0,0) =", field[q, p])
print("解析轴上      u2(0,0)    =", exact)
print("相对差        =", abs(field[q, p] - exact) / abs(exact))
print("U[0,0]*pref   =", U[0, 0] * np.exp(1j * 2 * np.pi / lam * z) / (1j * lamz))
print("dx_out        = %.6g um（λz/(MΔx)=%.6g）" % (dxo * 1e6, lamz / (M * dx) * 1e6))

res = fresnel_propagate(u1, g, FresnelConfig(lam, z, M))
print("库函数        res.field(0,0) =", res.field[q, p])
print("库/自写 比值  =", res.field[q, p] / field[q, p])
print("库 grid_out dx= %.6g um" % (res.grid.dx * 1e6))
print("库 field[0,0] =", res.field[0, 0], " 自写 field[0,0] =", field[0, 0])
print("库 field[1,1] =", res.field[1, 1], " 自写 field[1,1] =", field[1, 1])
print("全网格最大相对差 =", np.abs(res.field - field).max() / np.abs(field).max())
