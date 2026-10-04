# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 16（临时）：以高分辨率直接求积为裁判，判定两种方法谁对。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as MM                                      # noqa: E402
from optics.coordinates import Axis1D, Grid2D                         # noqa: E402
from optics.propagation import (fresnel_kernel_eval,                   # noqa: E402
                                fresnel_propagate_transfer)

lam, z = 550e-9, 50e-3
lamz = lam * z
w0 = 0.03e-3
tilt = 0.5e-3


def field_on(W, n):
    dx = 2 * W / n
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    X, Y = g.meshgrid()
    u1 = (np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)
          * np.exp(1j * 2 * np.pi / lam * tilt * X)).astype(complex)
    return g, u1


print("裁判：输入网格极细、窗口足够大，使直接求和自身可信")
for W_ref, n_ref in [(0.3e-3, 3000), (0.6e-3, 6000)]:
    g_ref, u_ref = field_on(W_ref, n_ref)
    dx_ref = g_ref.dx
    print("  参考网格 W=%.3gmm n=%d dx=%.4gum : L*dx/(λz)=%.4g  核相位步进=%.4g rad"
          % (W_ref * 1e3, n_ref, dx_ref * 1e6, W_ref * dx_ref / lamz,
             np.pi * 2 * W_ref * dx_ref / lamz))
    pts = np.array([0.0])
    ref = fresnel_kernel_eval(u_ref, g_ref, lam, z, pts, pts)[0, 0]
    print("    参考 u2(0,0) =", ref)

    for W, n in [(0.15e-3, 64), (0.3e-3, 128), (0.45e-3, 256)]:
        g, u1 = field_on(W, n)
        d = fresnel_kernel_eval(u1, g, lam, z, pts, pts)[0, 0]
        res = fresnel_propagate_transfer(u1, g, lam, z, padded_size=8 * n)
        tf = res.field[g.shape[0] // 2, g.shape[1] // 2]
        print("    W=%.3gmm n=%4d  直接=%-32s 传递=%-32s" % (W * 1e3, n, d, tf))
        print("        相对裁判：直接 %.3e   传递 %.3e" % (
            abs(d - ref) / abs(ref), abs(tf - ref) / abs(ref)))
