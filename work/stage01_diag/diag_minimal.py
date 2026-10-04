# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 5（临时）：最小可判定实验 —— 平面波(无孔径)与孔径两种输入。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics.coordinates import (circular_aperture, grid_from_half_width,  # noqa: E402
                                plane_wave)
from optics.propagation import FresnelConfig, fresnel_propagate           # noqa: E402

lam, z = 550e-9, 50e-3
N, half = 512, 2e-3
g = grid_from_half_width(half, N)
lamz = lam * z
chirp = np.exp(1j * np.pi * g.radius() ** 2 / lamz)
pref = g.cell_area / (1j * lamz) * np.exp(1j * 2 * np.pi / lam * z)


def manual(label, u1, D=None):
    A = u1 * chirp
    U = np.fft.fft2(A)
    # 手工计算同一个 DC 求和，用高精度累加
    s_np = np.sum(A)
    s_fsum = complex(sum(A.real.ravel().tolist()), sum(A.imag.ravel().tolist()))
    print("[%s]" % label)
    print("   np.sum(A)      =", s_np)
    print("   U[0,0]         =", U[0, 0])
    print("   U[0,0]*pref    =", U[0, 0] * pref)
    res = fresnel_propagate(u1, g, FresnelConfig(lam, z, N))
    q = p = N // 2
    print("   res.field(0,0) =", res.field[q, p])
    print("   res.field/U00  =", res.field[q, p] / (U[0, 0] * pref))
    if D is not None:
        exact = np.exp(1j * 2 * np.pi / lam * z) * (1 - np.exp(1j * np.pi * D ** 2 / (4 * lamz)))
        print("   解析轴上       =", exact)
        print("   U00*pref/解析  =", U[0, 0] * pref / exact)


D = 0.5e-3
manual("平面波 + 圆孔 D=0.5mm", plane_wave(g) * circular_aperture(g, D), D)
manual("纯平面波（无孔径）", plane_wave(g))
