# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 18（临时）：传递函数形式的多种变体 vs 直接求和（裁判）。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics.coordinates import Axis1D, Grid2D                       # noqa: E402
from optics.propagation import fresnel_kernel_eval                  # noqa: E402

lam, z = 550e-9, 50e-3
k = 2 * np.pi / lam
lamz = lam * z
n, dx = 24, 10e-6
g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
X, Y = g.meshgrid()
u1 = np.exp(-(X ** 2 + Y ** 2) / (40e-6) ** 2).astype(complex)

# 裁判：高分辨率直接求和
nR, WR = 1200, 0.12e-3
dxR = 2 * WR / nR
gR = Grid2D(Axis1D(nR, dxR, "x"), Axis1D(nR, dxR, "y"))
XR, YR = gR.meshgrid()
uR = np.exp(-(XR ** 2 + YR ** 2) / (40e-6) ** 2).astype(complex)
ref = fresnel_kernel_eval(uR, gR, lam, z, np.array([0.0]), np.array([0.0]))[0, 0]
print("裁判（W=%.3gmm, n=%d, dx=%.3gum）: u2(0,0) = %s" % (WR * 1e3, nR, dxR * 1e6, ref))

d = fresnel_kernel_eval(u1, g, lam, z, np.array([0.0]), np.array([0.0]))[0, 0]
print("直接求和 (n=%d,dx=%.3gum)      : %s   相对裁判 %.3e"
      % (n, dx * 1e6, d, abs(d - ref) / abs(ref)))

for M in (n, 2 * n, 4 * n):
    o = (M - n) // 2
    big = np.zeros((M, M), complex)
    big[o:o + n, o:o + n] = u1
    fx = np.fft.fftfreq(M, d=dx)
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    fr2 = FX ** 2 + FY ** 2
    variants = {
        "fresnel": np.exp(1j * k * z) * np.exp(-1j * np.pi * lamz * fr2),
        "exact  ": np.exp(1j * k * z * np.sqrt((1 - lam ** 2 * fr2).astype(complex))),
        "no-global": np.exp(-1j * np.pi * lamz * fr2),
    }
    for name, H in variants.items():
        out = np.fft.ifft2(np.fft.fft2(big) * H)
        # fftshift 后中心
        c = out[o + n // 2, o + n // 2]
        print("  M=%3d %s : %s   相对裁判 %.3e" % (M, name, c, abs(c - ref) / abs(ref)))
    # 也试：fftshift 版（原点在数组中心）
    U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(big)))
    H2 = np.exp(1j * k * z) * np.exp(-1j * np.pi * lamz * (
        np.fft.fftshift(fx)[None, :] ** 2 + np.fft.fftshift(fx)[:, None] ** 2))
    out2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U * H2)))
    print("  M=%3d shifted: %s   相对裁判 %.3e"
          % (M, out2[M // 2, M // 2], abs(out2[M // 2, M // 2] - ref) / abs(ref)))
