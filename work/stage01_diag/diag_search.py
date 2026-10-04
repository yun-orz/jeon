# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 15（临时）：搜索同时满足两种方法适用条件的参数组合。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as MM                                          # noqa: E402
from optics.coordinates import Axis1D, Grid2D                             # noqa: E402
from optics.propagation import (fresnel_propagate_transfer,                # noqa: E402
                                direct_fresnel_integral)

lam = 550e-9


def trial(D_field, W, n, z, tag=""):
    """D_field: 场特征横向尺寸（高斯 w0）。返回各判据与两种方法的一致性。"""
    dx = 2 * W / n
    lamz = lam * z
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    X, Y = g.meshgrid()
    w0 = D_field
    u1 = (np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)
          * np.exp(1j * 2 * np.pi / lam * 0.5e-3 * X)).astype(complex)
    edge = abs(u1[0, 0])
    # 判据 1：传递函数的周期冲激响应 chirp 采样（在 Nyquist 频率处）
    step_TF = np.pi * lamz * (1.0 / (2 * dx)) ** 2 / 2.0 * 2  # 近似：πλz f_N²
    # 判据 2：直接求和的核相位步进 —— 输入最远点与评价点最远点
    step_DS = np.pi * 2 * (W / 2) * dx / lamz
    M = 8 * n
    res = fresnel_propagate_transfer(u1, g, lam, z, padded_size=M)
    d = direct_fresnel_integral(u1, g, lam, z, g)
    rel = MM.relative_l2_complex(res.field, d)
    print("%s w0=%6.3gmm W=%6.3gmm n=%5d z=%6.4gm 边缘=%8.1e  λz/Δx²=%9.1f  "
          "TF步进=%6.3f  DS步进=%6.3f  relL2=%.3e"
          % (tag, w0 * 1e3, W * 1e3, n, z, edge, lamz / dx ** 2, step_TF, step_DS, rel))
    return rel


print("目标：relL2 尽量小，且 边缘|u1| 足够小（场被网格容纳）")
for z in (1e-3, 5e-3, 2e-2, 5e-2):
    for n, W in ((64, 0.15e-3), (128, 0.3e-3), (256, 0.6e-3), (512, 1.2e-3)):
        w0 = W / 10.0
        trial(w0, W, n, z)
    print()
