# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 9（临时）：两种形式 + 解析解 + 直接求积 的四方对照。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as M                                             # noqa: E402
from optics.coordinates import Axis1D, Grid2D                               # noqa: E402
from optics.propagation import (FresnelConfig, direct_fresnel_integral,      # noqa: E402
                                fresnel_propagate, fresnel_propagate_transfer,
                                ir_form_validity)

lam, z = 550e-9, 50e-3


def case(n, half, w0, tilt, m_pad):
    dx = 2 * half / n
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    u1 = M.tilted_gaussian_field(g, lam, w0, (0.0, 0.0), (tilt, 0.0))
    print("=" * 78)
    print("n=%d half=%.3g mm Δx=%.4g um w0=%.3g mm tilt=%.3g  最大|u1|=%.4g 边缘=%.3g"
          % (n, half * 1e3, dx * 1e6, w0 * 1e3, tilt, np.abs(u1).max(), np.abs(u1[0, 0])))
    iv = ir_form_validity(half, dx, lam, z)
    print("  IR 适用性：最大相位步进=%.4g rad（阈值 %.2f）→ ok=%s；网格上相位周期数=%.4g"
          % (iv["max_phase_step_rad"], iv["threshold_rad"], iv["ok"],
             iv["phase_cycles_on_grid"]))

    # (a) 传递函数形式（输入网格）
    res_t = fresnel_propagate_transfer(u1, g, lam, z, padded_size=m_pad)
    ana_g = M.tilted_gaussian_paraxial_solution(g, lam, z, w0, (tilt, 0.0))
    print("  [传递函数] relL2 vs 解析 = %.6e   |u|max=%.6g (解析 %.6g)"
          % (M.relative_l2_complex(res_t.field, ana_g),
             np.abs(res_t.field).max(), np.abs(ana_g).max()))

    # (b) IR 形式（λz/(MΔx′) 原生输出网格）
    res_i = fresnel_propagate(u1, g, FresnelConfig(lam, z, m_pad))
    ana_i = M.tilted_gaussian_paraxial_solution(res_i.grid, lam, z, w0, (tilt, 0.0))
    print("  [IR 形式 ] 输出Δx=%.4g um  网格 %d×%d  relL2 vs 解析 = %.6e"
          % (res_i.grid.dx * 1e6, res_i.grid.shape[0], res_i.grid.shape[1],
             M.relative_l2_complex(res_i.field, ana_i)))

    # (c) 直接求积（在 IR 输出网格上）
    d = direct_fresnel_integral(u1, g, lam, z, res_i.grid)
    print("  [直接求和] 在 IR 网格上：relL2 vs IR = %.6e    relL2 vs 解析 = %.6e"
          % (M.relative_l2_complex(res_i.field, d), M.relative_l2_complex(d, ana_i)))

    # (d) 直接求积（在输入网格上）vs 传递函数
    d2 = direct_fresnel_integral(u1, g, lam, z, g)
    print("  [直接求和] 在输入网格上：relL2 vs 传递函数 = %.6e   relL2 vs 解析 = %.6e"
          % (M.relative_l2_complex(res_t.field, d2), M.relative_l2_complex(d2, ana_g)))

    # (e) 中心区域限制比较
    r = res_i.grid.radius()
    mask = r <= 0.25 * half
    print("  [中心区 r≤%.3g mm] IR vs 解析 relL2 = %.6e ; IR vs 直接 = %.6e"
          % (0.25 * half * 1e3,
             M.relative_l2_complex(res_i.field[mask], ana_i[mask]),
             M.relative_l2_complex(res_i.field[mask], d[mask])))


case(n=16, half=0.5e-3, w0=0.2e-3, tilt=1e-3, m_pad=32)
case(n=32, half=1.0e-3, w0=0.4e-3, tilt=1e-3, m_pad=128)
