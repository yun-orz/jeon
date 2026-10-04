# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 10（临时）：在满足采样条件的状态下做四方对照。"""
import sys

import numpy as np

sys.path.insert(0, r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
from optics import metrics as M                                            # noqa: E402
from optics.coordinates import Axis1D, Grid2D                              # noqa: E402
from optics.propagation import (FresnelConfig, direct_fresnel_integral,     # noqa: E402
                                fresnel_propagate, fresnel_propagate_transfer,
                                ir_form_validity)

lam, z = 550e-9, 50e-3


def case(n, half, w0, tilt, m_pad, tag=""):
    dx = 2 * half / n
    g = Grid2D(Axis1D(n, dx, "x"), Axis1D(n, dx, "y"))
    u1 = M.tilted_gaussian_field(g, lam, w0, (0.0, 0.0), (tilt, 0.0))
    print("=" * 78)
    print("%s n=%d half=%.3g mm Δx=%.4g um w0=%.3g mm tilt=%.3g  λz/Δx²=%.4g"
          % (tag, n, half * 1e3, dx * 1e6, w0 * 1e3, tilt, lam * z / dx ** 2))
    iv = ir_form_validity(half, dx, lam, z)
    print("  IR 判据：最大相位步进=%.4g rad ok=%s；周期数=%.4g"
          % (iv["max_phase_step_rad"], iv["ok"], iv["phase_cycles_on_grid"]))

    res_t = fresnel_propagate_transfer(u1, g, lam, z, padded_size=m_pad)
    ana_g = M.tilted_gaussian_paraxial_solution(g, lam, z, w0, (tilt, 0.0))
    print("  [传递函数] relL2 vs 解析 = %.6e" % M.relative_l2_complex(res_t.field, ana_g))

    res_i = fresnel_propagate(u1, g, FresnelConfig(lam, z, m_pad))
    ana_i = M.tilted_gaussian_paraxial_solution(res_i.grid, lam, z, w0, (tilt, 0.0))
    print("  [IR 形式 ] 输出Δx=%.4g um 网格%d  relL2 vs 解析 = %.6e"
          % (res_i.grid.dx * 1e6, res_i.grid.shape[0],
             M.relative_l2_complex(res_i.field, ana_i)))

    d = direct_fresnel_integral(u1, g, lam, z, res_i.grid)
    print("  [直接求和@IR网格] relL2 vs IR = %.6e   vs 解析 = %.6e"
          % (M.relative_l2_complex(res_i.field, d), M.relative_l2_complex(d, ana_i)))

    d2 = direct_fresnel_integral(u1, g, lam, z, g)
    print("  [直接求和@输入网格] relL2 vs 传递函数 = %.6e   vs 解析 = %.6e"
          % (M.relative_l2_complex(res_t.field, d2), M.relative_l2_complex(d2, ana_g)))


case(n=512, half=5e-3, w0=1e-3, tilt=1e-3, m_pad=1024, tag="配置A")
case(n=512, half=5e-3, w0=1e-3, tilt=1e-3, m_pad=2048, tag="配置B")
