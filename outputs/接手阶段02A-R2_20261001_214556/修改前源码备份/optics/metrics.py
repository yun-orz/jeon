# -*- coding: utf-8 -*-
"""评价指标与解析参考（阶段 01）。

包含三类内容：

1. 通用误差与相关量（复场相对 L2、归一化强度 L1、总面积加权 L1、最优全局相位）。
2. Airy 图样解析参考与暗环定位（径向平均 + 抛物插值 + Bessel 零点）。
3. 倾角高斯光束的**解析** Fresnel 传播解，用作独立物理参照。

所有长度单位均为米（m）；强度均为 |u|²。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import interpolate, special

from .coordinates import Axis1D, Grid2D

__all__ = [
    "relative_l2_complex",
    "remove_global_phase",
    "normalized_l1_intensity",
    "total_weighted_l1",
    "siec_radius",
    "azimuthal_average",
    "find_dark_rings",
    "find_local_minima",
    "airy_intensity",
    "airy_first_dark_ring_radius",
    "airy_dark_ring_radii",
    "airy_encircled_energy_analytic",
    "analytic_encircled_energy_from_profile",
    "tilted_gaussian_field",
    "tilted_gaussian_paraxial_solution",
    "make_centered_axis",
    "resample_intensity_bilinear",
    "relative_difference",
]


# --------------------------------------------------------------------------------------
# 通用误差量
# --------------------------------------------------------------------------------------
def relative_l2_complex(a: np.ndarray, b: np.ndarray) -> float:
    """复场相对 L2 误差 ``‖a − b‖₂ / ‖b‖₂``。

    ``a`` 为待检值、``b`` 为参考。若 ``‖b‖₂ = 0`` 则返回 ``nan``（排除零参考场除零）。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    denom = float(np.sqrt(np.sum(np.abs(b) ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2)) / denom)


def remove_global_phase(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, complex]:
    """去掉 ``a`` 相对 ``b`` 的**唯一**最优全局相位 ``exp(iθ)``，返回 ``(a·e^{−iθ}, e^{iθ})``。

    ``e^{iθ} = <b, a> / |<b, a>|``，其中 ``<b, a> = Σ conj(b)·a``。

    这是闭式解，不是拟合：除一个空间恒定相位外不改变任何东西，
    既不改变 ``|a|``，也不引入任何自由参数。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    ip = np.sum(np.conj(b) * a)
    if ip == 0.0:
        return a, 1.0 + 0.0j
    ph = ip / abs(ip)
    return a / ph, ph


def normalized_l1_intensity(i_a: np.ndarray, i_b: np.ndarray) -> float:
    """峰值归一化强度 L1 差 ``Σ|Î_a − Î_b| / Σ Î_b``，``Î = I / max(I)``。

    分式形式让指标**与整体光强尺度无关**（固定观察区里两幅图的总能量可能不同），
    但分母是参考图的归一化总量，所以仍是绝对（非形状相关系数）量度。
    """
    i_a = np.asarray(i_a, dtype=np.float64)
    i_b = np.asarray(i_b, dtype=np.float64)
    ma = float(np.max(i_a)) if i_a.size else 0.0
    mb = float(np.max(i_b)) if i_b.size else 0.0
    if ma <= 0 or mb <= 0:
        return float("nan")
    ia = i_a / ma
    ib = i_b / mb
    den = float(np.sum(ib))
    if den == 0.0:
        return float("nan")
    return float(np.sum(np.abs(ia - ib)) / den)


def total_weighted_l1(intensity: np.ndarray, cell_area: float) -> float:
    """面积加权总强度 ``Σ I·ΔA``（用于报告未归一化尺度）。"""
    return float(np.sum(np.asarray(intensity, dtype=np.float64)) * float(cell_area))


def relative_difference(value: float, reference: float) -> float:
    """相对差 ``(value − reference)/reference``；reference 为 0 时返回 nan。"""
    if reference == 0:
        return float("nan")
    return float((value - reference) / reference)


def siec_radius(power_fraction: float, r_axis: np.ndarray, encircled_fraction: np.ndarray) -> float:
    """给定包围能量比例，插值求对应半径。"""
    f = np.asarray(encircled_fraction, dtype=np.float64)
    r = np.asarray(r_axis, dtype=np.float64)
    if np.any(np.diff(f) < 0):
        f = np.maximum.accumulate(f)
    idx = int(np.searchsorted(f, float(power_fraction)))
    if idx <= 0:
        return float(r[0])
    if idx >= len(f):
        return float(r[-1])
    x0, x1 = f[idx - 1], f[idx]
    if x1 == x0:
        return float(r[idx])
    t = (float(power_fraction) - x0) / (x1 - x0)
    return float(r[idx - 1] + t * (r[idx] - r[idx - 1]))


# --------------------------------------------------------------------------------------
# 径向平均与暗环
# --------------------------------------------------------------------------------------
def azimuthal_average(
    intensity: np.ndarray,
    grid: Grid2D,
    n_bins: int,
    r_max: Optional[float] = None,
    drop_outside: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """方位角平均，直接在各网格**原生样点**上做，不做任何插值。

    Parameters
    ----------
    r_max : float or None
        None 时取网格最大半径。
    drop_outside : bool
        True（默认）：只统计 ``r <= r_max`` 的样点，返回的 ``count`` 总数会小于
        网格样点数。False：把 ``r > r_max`` 的样点也计入最后一个 bin。
        调用方若要用 ``count`` 核对样点总数，必须显式传 ``drop_outside=False``。

    Returns
    -------
    (r_center, mean_intensity, count)
        ``r_center`` 为 bin 中心半径（m）；``mean_intensity`` 为 bin 内样点的
        算术平均强度；``count`` 为落入该 bin 的样点数（为 0 的 bin 强度为 nan）。
    """
    r = grid.radius().ravel()
    I = np.asarray(intensity, dtype=np.float64).ravel()
    if r_max is None:
        r_max = float(r.max())
    edges = np.linspace(0.0, float(r_max), int(n_bins) + 1)
    idx = np.digitize(r, edges) - 1
    if drop_outside:
        valid = (idx >= 0) & (idx < int(n_bins))
    else:
        valid = idx >= 0
        idx = np.minimum(idx, int(n_bins) - 1)
    idx = idx[valid]
    Iv = I[valid]
    count = np.bincount(idx, minlength=int(n_bins)).astype(np.int64)
    ssum = np.bincount(idx, weights=Iv, minlength=int(n_bins))
    mean = np.full(int(n_bins), np.nan, dtype=np.float64)
    good = count > 0
    mean[good] = ssum[good] / count[good]
    r_center = 0.5 * (edges[:-1] + edges[1:])
    return r_center, mean, count


@dataclass
class DarkRing:
    """一个暗环的定位结果。"""

    order: int             # 第几个暗环（从 1 开始）
    r_m: float             # 定位半径（m）
    r_theory_m: float      # 理论半径（m）
    rel_err: float         # (r − r_theory)/r_theory
    r_bin_m: float         # 未插值的 bin 中心半径（m）
    intensity_at_min: float


def find_local_minima(r_center: np.ndarray, profile: np.ndarray,
                      max_radius: Optional[float] = None) -> List[Tuple[int, float, float]]:
    """找径向剖面的离散局部极小点，并用相邻三点抛物插值细化。

    判据（预先固定）：``I[k] < I[k−1]`` 且 ``I[k] <= I[k+1]``。

    Returns
    -------
    list of (k, r_refined_m, I[k])
        按半径从小到大排列。
    """
    r = np.asarray(r_center, dtype=np.float64)
    I = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(I)
    out: List[Tuple[int, float, float]] = []
    for k in range(1, len(I) - 1):
        if not (finite[k - 1] and finite[k] and finite[k + 1]):
            continue
        if max_radius is not None and r[k] > max_radius:
            continue
        if (I[k] < I[k - 1]) and (I[k] <= I[k + 1]):
            y0, y1, y2 = I[k - 1], I[k], I[k + 1]
            denom = (y0 - 2.0 * y1 + y2)
            delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            d = r[k + 1] - r[k]
            out.append((k, float(r[k] + delta * d), float(y1)))
    return out


def find_dark_rings(
    r_center: np.ndarray,
    profile: np.ndarray,
    theoretical_radii: Sequence[float],
    max_radius: Optional[float] = None,
    rel_tol: float = 0.35,
) -> List[DarkRing]:
    """在径向剖面中定位暗环，并与**最近的**理论暗环配对。

    方法（预先固定，不针对结果调整）：

    1. 用 :func:`find_local_minima` 找全部局部极小点（含亚样点抛物插值）；
    2. 对每个理论暗环半径 ``r_k``，只接受满足 ``|r − r_k|/r_k ≤ rel_tol`` 的极小点；
    3. 若同一理论暗环有多个候选，取最近的一个；每个极小点最多用一次。

    ``rel_tol`` 的存在是必要的：当网格欠采样时，方位角平均剖面会出现大量高频
    起伏（数值伪影），按"第 n 个极小点"配对会把伪影当成真实暗环——本阶段实测
    遇到过这一失败，见 ``docs/reproduction_log.md``。
    """
    r = np.asarray(r_center, dtype=np.float64)
    minima = find_local_minima(r, profile, max_radius=max_radius)
    used = [False] * len(minima)
    out: List[DarkRing] = []
    for order, rt in enumerate(theoretical_radii, start=1):
        rt = float(rt)
        best = None
        for idx, (k, r_min, i_min) in enumerate(minima):
            if used[idx]:
                continue
            if rt == 0:
                continue
            rel = abs(r_min - rt) / rt
            if rel > rel_tol:
                continue
            if best is None or rel < best[0]:
                best = (rel, idx, k, r_min, i_min)
        if best is None:
            continue
        rel, idx, k, r_min, i_min = best
        used[idx] = True
        out.append(DarkRing(order=order, r_m=float(r_min), r_theory_m=rt,
                            rel_err=float((r_min - rt) / rt), r_bin_m=float(r[k]),
                            intensity_at_min=float(i_min)))
    return out


# --------------------------------------------------------------------------------------
# Airy 解析参考
# --------------------------------------------------------------------------------------
def airy_intensity(radius: np.ndarray, wavelength: float, focal_length: float,
                   diameter: float, normalize: bool = True) -> np.ndarray:
    """理想圆孔 + 理想薄透镜的焦平面 Airy 强度 ``I(r) = [2J₁(x)/x]²``。

    ``x = π D r / (λ f)``，``I(0) = 1``。``D`` 为孔径直径、``f`` 为焦距。
    """
    r = np.asarray(radius, dtype=np.float64)
    x = np.pi * float(diameter) * r / (float(wavelength) * float(focal_length))
    small = np.abs(x) < 1e-8
    out = np.empty_like(x)
    xs = np.where(small, 1.0, x)
    out = (2.0 * special.j1(xs) / xs) ** 2
    if np.any(small):
        out = np.where(small, 1.0, out)
    return out if normalize else out


def airy_first_dark_ring_radius(wavelength: float, focal_length: float, diameter: float) -> float:
    """第一暗环半径 ``r₁ = 1.22 λ f / D``（常见近轴圆孔衍射基准）。"""
    return 1.22 * float(wavelength) * float(focal_length) / float(diameter)


def airy_dark_ring_radii(wavelength: float, focal_length: float, diameter: float,
                         n_rings: int) -> np.ndarray:
    """前 ``n_rings`` 个暗环半径，由 Bessel 函数 ``J₁`` 的零点精确给出：

    ``r_k = j_{1,k} · λ f / (π D)``；``k=1`` 时 ``j_{1,1}=3.8317…``，
    对应 ``1.2197 λf/D``（即常用的 1.22 近似）。
    """
    zeros = special.jn_zeros(1, int(n_rings))
    return zeros * float(wavelength) * float(focal_length) / (np.pi * float(diameter))


def airy_encircled_energy_analytic(radius: np.ndarray, wavelength: float,
                                   focal_length: float, diameter: float) -> np.ndarray:
    """Airy 图样的解析包围能量比例 ``1 − J₀²(x) − J₁²(x)``，``x = πDr/(λf)``。

    该式对总能量归一化（``r → ∞`` 时为 1），用于与 FFT 输出的离散包围能量对照。
    """
    r = np.asarray(radius, dtype=np.float64)
    x = np.pi * float(diameter) * r / (float(wavelength) * float(focal_length))
    return 1.0 - special.j0(x) ** 2 - special.j1(x) ** 2


def analytic_encircled_energy_from_profile(
    r_center: np.ndarray,
    profile: np.ndarray,
    radii: np.ndarray,
    r_zero: float = 0.0,
    i_zero: float = 1.0,
) -> np.ndarray:
    """对解析径向剖面做数值积分得到的包围能量比例（不给公式时用这个交叉检查）。

    使用复合梯形法；``r=0`` 处的值由 ``i_zero`` 给定（Airy 峰值 = 1）。
    """
    r = np.concatenate(([float(r_zero)], np.asarray(r_center, dtype=np.float64)))
    I = np.concatenate(([float(i_zero)], np.asarray(profile, dtype=np.float64)))
    good = np.isfinite(I)
    r, I = r[good], I[good]
    integ = np.concatenate(([0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(r))))
    return np.interp(np.asarray(radii, dtype=np.float64), r, integ)


# --------------------------------------------------------------------------------------
# 解析参照：倾角高斯光束的 Fresnel 传播
# --------------------------------------------------------------------------------------
def tilted_gaussian_field(
    grid: Grid2D,
    wavelength: float,
    waist: float,
    center: Tuple[float, float] = (0.0, 0.0),
    tilt: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """倾角高斯光束在**束腰平面**（z = 0）的解析复振幅::

        u(x, y) = exp[ −((x−c_x)² + (y−c_y)²)/w0² ] · exp[ i k ((x−c_x) sinθ_x + (y−c_y) sinθ_y) ]

    ``tilt`` 以弧度给出。它在 :func:`tilted_gaussian_paraxial_solution` 中取
    ``distance = 0`` 时严格复现，因此可直接作为解析传播解的输入一致性检查。
    """
    X, Y = grid.meshgrid()
    k = 2.0 * np.pi / float(wavelength)
    dx = X - float(center[0])
    dy = Y - float(center[1])
    env = np.exp(-(dx ** 2 + dy ** 2) / float(waist) ** 2)
    tilt_phase = np.exp(1j * k * (dx * float(tilt[0]) + dy * float(tilt[1])))
    return (env * tilt_phase).astype(np.complex128)


def tilted_gaussian_paraxial_solution(
    grid: Grid2D,
    wavelength: float,
    distance: float,
    waist: float,
    tilt: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """倾角高斯光束经距离 ``z`` 后的**解析**近轴解（连续解，不依赖任何离散求和）。

    取束腰位于 z = 0 平面、中心在原点、束宽 ``w0``，传播方向与 +z 成小角度
    ``θ_x, θ_y``。自由空间近轴（Fresnel）方程的标准解为

        u(x, y, z) = exp(ikz) / (1 + i z/z_R)
                     · exp[ −((x−x_c)² + (y−y_c)²) / (w0² (1 + i z/z_R)) ]
                     · exp[ ik ((x−x_c) sinθ_x + (y−y_c) sinθ_y) ]
        x_c = z·sinθ_x,  y_c = z·sinθ_y,  z_R = π w0²/λ

    它在 z = 0 处退化为输入定义 :func:`tilted_gaussian_field`
    （``u(x,y,0) = exp[−(x²+y²)/w0²]·exp[ik(x sinθ_x + y sinθ_y)]``），
    因此"解析解"与"数值传播的输入"是严格同一对象。

    与式(4) 的相位约定一致：保留 ``exp(ikz)``；横向倾斜相位落在光斑中心
    （Gouy 相位取在轴上，横向可忽略），斜率 ``k sinθ`` 与输入斜率严格相同。
    """
    X, Y = grid.meshgrid()
    lam = float(wavelength)
    z = float(distance)
    w0 = float(waist)
    k = 2.0 * np.pi / lam
    zr = np.pi * w0 ** 2 / lam
    cx = z * float(tilt[0])
    cy = z * float(tilt[1])
    dx = X - cx
    dy = Y - cy
    r2 = dx ** 2 + dy ** 2
    pref = np.exp(1j * k * z) / (1.0 + 1j * z / zr)
    return (pref * np.exp(-(r2 / w0 ** 2) / (1.0 + 1j * z / zr))
            * np.exp(1j * k * (dx * float(tilt[0]) + dy * float(tilt[1])))
            ).astype(np.complex128)


# --------------------------------------------------------------------------------------
# 网格重采样
# --------------------------------------------------------------------------------------
def resample_intensity_bilinear(
    intensity: np.ndarray,
    grid: Grid2D,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """双线性插值把强度映射到新坐标（**仅用于固定观察区的对照**）。

    越界点返回 ``nan``（不填 0、不外推），以便在指标中显式排除。
    """
    I = np.asarray(intensity, dtype=np.float64)
    f = interpolate.RegularGridInterpolator(
        (grid.y.coords, grid.x.coords), I, method="linear", bounds_error=False, fill_value=np.nan)
    Xn, Yn = np.meshgrid(np.asarray(x_new, dtype=np.float64),
                         np.asarray(y_new, dtype=np.float64), indexing="xy")
    pts = np.stack([Yn.ravel(), Xn.ravel()], axis=-1)
    return f(pts).reshape(Xn.shape)


def make_centered_axis(half_width: float, n: int) -> np.ndarray:
    """生成以 0 为中心、覆盖 ±half_width 的 ``n`` 点均匀轴（``n`` 为奇数时含 0）。"""
    n = int(n)
    if n < 2:
        raise ValueError("n 必须 >= 2")
    return np.linspace(-float(half_width), float(half_width), n)
