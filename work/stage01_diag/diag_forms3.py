# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 11（临时）：按 λz/Δx² ≥ M/2 选择参数后的四方对照。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as M                                             # noqa: E402
from optics.coordinates import Axis1D, Grid2D                               # noqa: E402
from optics.propagation import (FresnelConfig, direct_fresnel_integral,      # noqa: E402
                                fresnel_propagate, fresnel_propagate_transfer,
                                ir_form_validity)

lam, z = 550e-9, 50e-3
lamz = lam * z


def case(n, half, w0, tilt, m_pad, tag=""):
    dx = 2 * half / n
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    u1 = M.tilted_gaussian_field(g, lam, w0, (0.0, 0.0), (tilt, 0.0))
    print("=" * 78)
    print("%s N=%d Δx=%.4g um 窗口=±%.4g mm M=%d 输出Δx=%.4g um 输出窗口=±%.4g mm"
          % (tag, n, dx * 1e6, half * 1e3, m_pad, lamz / (m_pad * dx) * 1e6,
             lamz / (m_pad * dx) * (m_pad // 2) * 1e3))
    print("   λz/Δx²=%.4g (需 ≥ M/2=%.4g)   边缘|u1|=%.3g"
          % (lamz / dx ** 2, m_pad / 2, np.abs(u1[0, 0])))
    iv = ir_form_validity(half, dx, lam, z)
    print("   IR 判据：最大相位步进=%.4g rad ok=%s；周期数=%.4g"
          % (iv["max_phase_step_rad"], iv["ok"], iv["phase_cycles_on_grid"]))

    res_t = fresnel_propagate_transfer(u1, g, lam, z, padded_size=m_pad)
    ana_g = M.tilted_gaussian_paraxial_solution(g, lam, z, w0, (tilt, 0.0))
    print("   [传递函数@输入网格] relL2 vs 解析 = %.6e" % M.relative_l2_complex(res_t.field, ana_g))

    res_i = fresnel_propagate(u1, g, FresnelConfig(lam, z, m_pad))
    ana_i = M.tilted_gaussian_paraxial_solution(res_i.grid, lam, z, w0, (tilt, 0.0))
    print("   [IR 形式@输出网格]  relL2 vs 解析 = %.6e" % M.relative_l2_complex(res_i.field, ana_i))

    d = direct_fresnel_integral(u1, g, lam, z, res_i.grid)
    print("   [直接求和@IR网格]   relL2 vs IR   = %.6e   vs 解析 = %.6e"
          % (M.relative_l2_complex(res_i.field, d), M.relative_l2_complex(d, ana_i)))

    d2 = direct_fresnel_integral(u1, g, lam, z, g)
    print("   [直接求和@输入网格] relL2 vs 传递函数 = %.6e   vs 解析 = %.6e"
          % (M.relative_l2_complex(res_t.field, d2), M.relative_l2_complex(d2, ana_g)))


case(n=128, half=1.17e-3, w0=0.39e-3, tilt=1e-3, m_pad=256, tag="A")
case(n=192, half=1.17e-3, w0=0.39e-3, tilt=1e-3, m_pad=512, tag="B")
