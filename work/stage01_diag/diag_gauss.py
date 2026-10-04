# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 2（临时）：用手算的连续积分核对解析高斯解与离散求和。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as M                                    # noqa: E402
from optics.coordinates import Axis1D, Grid2D                      # noqa: E402
from optics.propagation import direct_fresnel_integral             # noqa: E402

lam, z, w0, tilt = 550e-9, 50e-3, 0.4e-3, 1e-3
k = 2 * np.pi / lam
lamz = lam * z


def integral_at(x_eval, n, half):
    """高分辨率离散 Fresnel 积分（输入解析式直接采样，不用项目函数）。"""
    dx = 2 * half / n
    c = (np.arange(n) - n // 2) * dx
    env = np.exp(-c ** 2 / w0 ** 2)
    u1 = env * np.exp(1j * k * tilt * c)
    ph = np.exp(1j * np.pi * (x_eval - c) ** 2 / lamz)
    return np.sum(u1 * ph) * dx * dx / (1j * lamz) * np.exp(1j * k * z)


def analytic_at(x_eval):
    return M.tilted_gaussian_paraxial_solution(
        Grid2D(Axis1D(1, 1.0, "x"), Axis1D(1, 1.0, "y")), lam, z, w0, (tilt, 0.0))


print("解析解（代码）在 x=0：", analytic_at(0.0).ravel()[0])
print("解析解（手算，含 Gouy）：",
      np.exp(1j * k * z) / (1 + 1j * z / (np.pi * w0 ** 2 / lam)))
print("解析解（手算，去 Gouy）：", np.exp(1j * k * z) / np.sqrt(1 + (z / (np.pi * w0 ** 2 / lam)) ** 2))
for n, half in [(128, 1e-3), (512, 1e-3), (2048, 2e-3), (8192, 4e-3), (8192, 8e-3)]:
    print("n=%5d half=%.1f mm  dx=%.4g um  积分@0 = %s"
          % (n, half * 1e3, 2 * half / n * 1e6, integral_at(0.0, n, half)))
print("解析解（代码）在 x=+50um：", analytic_at(50e-6).ravel()[0])
print("积分@50um (n=8192,half=8mm)：", integral_at(50e-6, 8192, 8e-3))
