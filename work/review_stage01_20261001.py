# -*- coding: utf-8 -*-
"""阶段 01 独立审核脚本：只新增证据，不修改 DSH 提交的实现。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import j1, jn_zeros

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "outputs" / "jeon2019_optics"
BASE = ROOT / "outputs" / "审核_阶段01_20261001"
EVIDENCE = BASE
index = 2
while EVIDENCE.exists():
    EVIDENCE = BASE.with_name(BASE.name + f"_{index:02d}")
    index += 1
EVIDENCE.mkdir(parents=True)
sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")

from optics.coordinates import Axis1D, Grid2D
from optics.propagation import (
    FresnelConfig,
    fresnel_propagate,
    fresnel_kernel_eval,
    third_order_phase_error_rad,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source_paths = sorted(PROJECT.rglob("*.py")) + [PROJECT / "config.json"]
before = {str(p.relative_to(PROJECT)): digest(p) for p in source_paths}


def rel_l2(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def formula_reference(u, grid, wavelength, distance, x, y):
    """直接使用完整位移核；核内已包含输入坐标的平方项。"""
    px = np.exp(1j * np.pi * (x[:, None] - grid.x.coords[None, :]) ** 2
                / (wavelength * distance))
    py = np.exp(1j * np.pi * (y[:, None] - grid.y.coords[None, :]) ** 2
                / (wavelength * distance))
    return grid.cell_area / (1j * wavelength * distance) * (py @ u @ px.T)


def centered_fft_reference(u, grid, wavelength, distance, shape):
    """用展开后的核和中心化 DFT，验证与完整核离散求和的恒等性。"""
    my, mx = shape
    ny, nx = u.shape
    padded = np.zeros(shape, dtype=np.complex128)
    oy, ox = my // 2 - ny // 2, mx // 2 - nx // 2
    padded[oy:oy + ny, ox:ox + nx] = u
    xp = (np.arange(mx) - mx // 2) * grid.dx
    yp = (np.arange(my) - my // 2) * grid.dy
    chirp = np.exp(1j * np.pi * (yp[:, None] ** 2 + xp[None, :] ** 2)
                   / (wavelength * distance))
    transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded * chirp)))
    x = np.fft.fftshift(np.fft.fftfreq(mx, grid.dx)) * wavelength * distance
    y = np.fft.fftshift(np.fft.fftfreq(my, grid.dy)) * wavelength * distance
    output_phase = np.exp(1j * np.pi * (y[:, None] ** 2 + x[None, :] ** 2)
                          / (wavelength * distance))
    result = transformed * output_phase * grid.cell_area / (1j * wavelength * distance)
    return result, x, y


lam, z = 550e-9, 50e-3
results = {"审核性质": "独立诊断，未修复生产代码", "project": str(PROJECT), "cases": []}
case_arrays = {}
for n, dx, padding in [(48, 40e-6, (96, 96)), (24, 10e-6, (48, 48))]:
    grid = Grid2D(Axis1D(n, dx), Axis1D(n, dx))
    X, Y = grid.meshgrid()
    hw = grid.x.half_width
    u = (np.exp(-((X - 0.18 * hw) / (0.30 * hw)) ** 2
                - ((Y + 0.11 * hw) / (0.17 * hw)) ** 2)
         * np.exp(2j * np.pi / lam * (1.3e-3 * X - 0.9e-3 * Y))
         * np.exp(1j * 2e7 * (X ** 3 + 0.5 * Y ** 3)))
    current_fft = fresnel_propagate(u, grid, FresnelConfig(lam, z, padding,
                                                         include_global_phase=False))
    fft_ref, x, y = centered_fft_reference(u, grid, lam, z, padding)
    direct_ref = formula_reference(u, grid, lam, z, x, y)
    current_direct = fresnel_kernel_eval(u, grid, lam, z, x, y,
                                        include_global_phase=False)
    # 再用显式二维位移核逐点求和，独立核对矩阵求积。
    points = [(0, 0), (len(y) // 2, len(x) // 2), (len(y) // 3, len(x) // 4)]
    loop_values = []
    matrix_values = []
    for j, i in points:
        kernel = np.exp(1j * np.pi * ((x[i] - X) ** 2 + (y[j] - Y) ** 2) / (lam * z))
        loop_values.append(np.sum(u * kernel) * grid.cell_area / (1j * lam * z))
        matrix_values.append(direct_ref[j, i])
    pin = float(np.sum(np.abs(u) ** 2) * grid.cell_area)
    output_area = float((x[1] - x[0]) * (y[1] - y[0]))
    p_ref = float(np.sum(np.abs(fft_ref) ** 2) * output_area)
    case = {
        "n": n, "dx_m": dx,
        "DSH_FFT_vs_DSH_direct_rel_L2": rel_l2(current_fft.field, current_direct),
        "DSH_FFT_power_ratio": current_fft.power / pin,
        "inverse_input_cell_area": 1 / grid.cell_area,
        "inverse_input_cell_area_squared": 1 / grid.cell_area ** 2,
        "independent_FFT_vs_full_kernel_rel_L2": rel_l2(fft_ref, direct_ref),
        "independent_matrix_vs_explicit_2D_sum_rel_L2": rel_l2(np.array(matrix_values),
                                                              np.array(loop_values)),
        "DSH_direct_vs_full_kernel_rel_L2": rel_l2(current_direct, direct_ref),
        "independent_FFT_power_relative_error": abs(p_ref / pin - 1),
    }
    results["cases"].append(case)
    if n == 48:
        case_arrays = dict(input=u, reference_fft=fft_ref, reference_direct=direct_ref,
                           dsh_fft=current_fft.field, dsh_direct=current_direct,
                           x_m=x, y_m=y)

# 高斯解析解：直接比较传播器结果与连续解析解，不仅检查解析函数自身。
grid = Grid2D(Axis1D(300, 10e-6), Axis1D(300, 10e-6))
X, Y = grid.meshgrid()
w0 = 0.2e-3
u = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(complex)
x = np.linspace(-0.6e-3, 0.6e-3, 61)
Xout, Yout = np.meshgrid(x, x)
q = 1 + 1j * z / (np.pi * w0 ** 2 / lam)
analytic = np.exp(-(Xout ** 2 + Yout ** 2) / (w0 ** 2 * q)) / q
true_direct = formula_reference(u, grid, lam, z, x, x)
bad_direct = fresnel_kernel_eval(u, grid, lam, z, x, x, include_global_phase=False)
results["gaussian"] = {
    "independent_direct_vs_analytic_rel_L2": rel_l2(true_direct, analytic),
    "DSH_direct_vs_analytic_rel_L2": rel_l2(bad_direct, analytic),
}

# 焦平面输入二次相位应与传播核中的输入二次项相消，任何 Fresnel 数下均成立。
airy_rows = []
for diameter, focus in [(0.1e-3, 50e-3), (0.2e-3, 25e-3), (1e-3, 50e-3)]:
    spacing = 1e-6
    n = int(np.ceil(diameter / spacing)) + 40
    if n % 2:
        n += 1
    g = Grid2D(Axis1D(n, spacing), Axis1D(n, spacing))
    X, Y = g.meshgrid()
    pupil = (X ** 2 + Y ** 2 <= (diameter / 2) ** 2).astype(float)
    field = pupil * np.exp(-1j * np.pi * (X ** 2 + Y ** 2) / (lam * focus))
    first = float(jn_zeros(1, 1)[0] * lam * focus / (np.pi * diameter))
    line_x = np.linspace(0, 2 * first, 601)
    good = formula_reference(field, g, lam, focus, line_x, np.zeros(1))[0]
    bad = fresnel_kernel_eval(field, g, lam, focus, line_x, np.zeros(1),
                              include_global_phase=False)[0]
    rho = np.pi * diameter * line_x / (lam * focus)
    airy = np.ones_like(rho)
    airy[1:] = (2 * j1(rho[1:]) / rho[1:]) ** 2
    good_i, bad_i = np.abs(good) ** 2, np.abs(bad) ** 2
    good_n, bad_n = good_i / good_i[0], bad_i / bad_i[0]
    airy_rows.append({
        "D_m": diameter, "f_m": focus, "Fresnel_number": diameter ** 2 / (4 * lam * focus),
        "theory_first_ring_um": first * 1e6,
        "independent_profile_rel_L1": float(np.sum(abs(good_n - airy)) / np.sum(airy)),
        "DSH_profile_rel_L1": float(np.sum(abs(bad_n - airy)) / np.sum(airy)),
        "independent_peak_over_discrete_pupil_theory": float(good_i[0]
            / (np.sum(pupil) * g.cell_area / (lam * focus)) ** 2),
    })
    case_arrays[f"airy_{n}_x_m"] = line_x
    case_arrays[f"airy_{n}_reference_intensity"] = good_n
    case_arrays[f"airy_{n}_dsh_intensity"] = bad_n
    case_arrays[f"airy_{n}_analytic"] = airy
results["airy"] = airy_rows

# 四次光程项以实际横向距离 rho 为变量；与现有函数的额外高阶因子比较。
rho = 0.00125
exact_path_difference = -rho ** 4 / (4 * z ** 2 * (np.sqrt(z ** 2 + rho ** 2)
                          + z + rho ** 2 / (2 * z)))
results["paraxial_diagnostic"] = {
    "rho_m": rho,
    "exact_phase_error_rad": abs(2 * np.pi / lam * exact_path_difference),
    "leading_fourth_order_phase_error_rad": 2 * np.pi / lam * rho ** 4 / (8 * z ** 3),
    "DSH_diagnostic_rad": third_order_phase_error_rad(50e-6, z, lam, 1.2e-3),
}

# 原始结果是否真的与当前实现对应：只在少量已存坐标点复算。
stored = PROJECT / "results" / "stage01" / "run_20260930_201901" / "arrays"
with np.load(stored / "input_field.npz", allow_pickle=False) as src, \
     np.load(stored / "output_field.npz", allow_pickle=False) as out:
    xs, ys = src["x_m"], src["y_m"]
    grid = Grid2D(Axis1D(len(xs), float(xs[1] - xs[0])),
                  Axis1D(len(ys), float(ys[1] - ys[0])))
    chosen = np.array([0, 150, 300, 450, 600])
    field = fresnel_kernel_eval(src["u1"], grid, lam, z,
                                out["x_m"][chosen], out["y_m"][chosen])
    results["stored_result_vs_current_code_subset_rel_L2"] = rel_l2(
        field, out["u2"][np.ix_(chosen, chosen)])

env = dict(os.environ)
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONDONTWRITEBYTECODE"] = "1"
run = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                     cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     encoding="utf-8", errors="replace", timeout=180)
(EVIDENCE / "DSH原有测试_独立运行.log").write_text(run.stdout, encoding="utf-8")
results["existing_tests"] = {"command": run.args, "returncode": run.returncode,
                              "log": "DSH原有测试_独立运行.log"}

after = {str(p.relative_to(PROJECT)): digest(p) for p in source_paths}
results["source_hashes"] = before
results["production_source_unchanged"] = before == after
(EVIDENCE / "独立审核数值.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
np.savez_compressed(EVIDENCE / "独立参考数组.npz", **case_arrays)
print(json.dumps(results, ensure_ascii=False, indent=2))
print("EVIDENCE_DIRECTORY", EVIDENCE)
