# -*- coding: utf-8 -*-
"""阶段 02B-1 的 PSF 定量分析：三翼旋转角、尺寸、能量、旁瓣口径。

本模块只做**分析**，不做传播、不改材料或高度。所有指标定义都写死在函数里并写明
单位与离散规则，避免同一名字在不同地方代表不同口径。

坐标与符号约定（与全项目一致）
------------------------------
* 数学坐标 x 向右、y 向上、z 朝探测器；数组 ``a[j, i]`` 对应 ``(x[i], y[j])``；
  绘图 ``origin='lower'``。
* 极角 ``theta = mod(atan2(y, x), 2*pi)``，**逆时针为正**。
* 观察面改变可能改变“顺/逆时针”的文字描述，报告必须先声明坐标。

三翼旋转角（第 8 节口径）
-------------------------
固定环带内::

    C3 = sum(Iraw * exp(+3i*theta)) * dA
    Pband = sum(Iraw) * dA
    A3 = |C3| / Pband
    alpha_wrapped = arg(C3) / 3

* 环带外权重为 0；光轴 ``r = 0`` 不计入环带。
* ``alpha`` 具有 **120° 周期**，不能按 360° 主轴解释。
* 仅对满足阈值 ``A3 >= A3_min`` 的**连续波长段**先 unwrap ``arg(C3)`` 再除以 3；
  低可靠值记 ``null``，不对 NaN 整列 unwrap，不跨不可靠间隙续接方向。

能量与尺寸（第 9 节口径）
-------------------------
::

    Pin = sum(|u1|^2) * dxin * dyin
    Pwindow = sum(Iraw) * dxout * dyout
    eta_window = Pwindow / Pin                       # 0 < eta <= 1.02 粗健康检查
    Eabs(R) = sum_{r<=R}(Iraw) * dA / Pin             # 绝对包围能量比例
    R50/R80 = Eabs 第一次达到 0.5/0.8 的半径；圆盘内未达到 -> None/not_reached

* ``R50/R80`` 的分母恒为 ``Pin``，**禁止**改用 ``Pwindow`` 或 ``Pdisk``。
* 不使用视场角落延长成不完整圆盘。
* 阈值面积：``Aq = sum_{ROI 且 I >= q*Ipeak} dA``，``req,q = sqrt(Aq/pi)``，
  ``q = 0.5 / 0.1``；它是**峰值相对形状指标**，不是通光效率。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "c3_rotation_metric",
    "unwrap_angles_over_reliable_segments",
    "absolute_encircled_energy",
    "find_first_radius_reaching",
    "threshold_area_radius",
    "fixed_window_centroid",
    "summarize_size_stability",
    "band_energy_fraction",
    "psf_metrics",
]

# 三翼角向谐波阶数（Jeon N=3）
C3_ORDER = 3


# --------------------------------------------------------------------------------------
# 三翼旋转角
# --------------------------------------------------------------------------------------
def c3_rotation_metric(intensity: np.ndarray, grid, r_min: float, r_max: float,
                       a3_min: float = 0.05,
                       ambiguity_wrapped_diff_rad: float = 0.9 * math.pi,
                       label: str = "") -> Dict[str, Any]:
    """固定环带内的三翼角向分量 C3、相对幅度 A3 与包裹角 ``arg(C3)/3``。

    Parameters
    ----------
    intensity : np.ndarray
        原始强度 ``Iraw``（未做任何峰值归一化），形状 ``(ny, nx)``。
    grid : Grid2D
        探测器物理网格（米）。
    r_min, r_max : float
        环带内外半径（米）。环带外权重为 0；``r = 0`` 不计入。
    a3_min : float
        角度可靠阈值（本项目暂定，非论文阈值）。
    ambiguity_wrapped_diff_rad : float
        相邻波长“三倍相位差”的包裹差告警阈值（默认 0.9π），
        接近 π 表示角度采样歧义，本批**不**自动加密波长。

    Returns
    -------
    dict
        含 ``C3_real``/``C3_imag``/``Pband``/``A3``/``alpha_wrapped_rad``/
        ``alpha_wrapped_deg``/``reliable``/``unreliable_reason``/``band_pixels`` 等。
    """
    I = np.asarray(intensity, dtype=np.float64)
    if I.shape != grid.shape:
        raise ValueError("强度形状 %r 与网格形状 %r 不一致" % (I.shape, grid.shape))
    if not np.all(np.isfinite(I)):
        raise ValueError("强度包含 NaN 或 Inf，拒绝计算角度指标")
    if not (0.0 <= float(r_min) < float(r_max)):
        raise ValueError("环带半径必须满足 0 <= r_min < r_max")
    if float(a3_min) < 0:
        raise ValueError("A3 阈值不能为负")

    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)
    theta = np.mod(np.arctan2(Y, X), 2.0 * np.pi)
    # 环带权重：环带外 0；光轴 r=0 不计入
    band = (r >= float(r_min)) & (r <= float(r_max)) & (r > 0.0)
    dA = float(grid.cell_area)

    band_pixels = int(np.count_nonzero(band))
    pband = float(np.sum(I[band]) * dA)
    if band_pixels == 0 or pband <= 0.0:
        return {
            "label": label, "r_min_m": float(r_min), "r_max_m": float(r_max),
            "band_pixels": band_pixels, "Pband": pband,
            "C3_real": None, "C3_imag": None, "C3_abs": None, "A3": None,
            "alpha_wrapped_rad": None, "alpha_wrapped_deg": None,
            "alpha_raw_rad": None, "alpha_raw_deg": None,
            "reliable": False, "unreliable_reason": "环带内无有效像素或总能量为零",
            "a3_min": float(a3_min), "order": C3_ORDER,
            "definition": "C3=sum(Iraw*exp(+3i*theta))*dA; A3=|C3|/Pband; alpha=arg(C3)/3",
        }

    phasor = np.exp(1j * C3_ORDER * theta)
    c3 = np.sum(I[band] * phasor[band]) * dA
    a3 = float(abs(c3)) / pband
    raw_alpha = float(np.angle(c3) / C3_ORDER)
    reliable = bool(a3 >= float(a3_min))
    reason = "" if reliable else (
        "A3=%.6g 低于阈值 %.6g，角度不可靠" % (a3, float(a3_min)))
    # R7 修正（2026-10-02）：不可靠测量**不得**对外发布角度数值。
    # 旧实现在 A3 未达阈值时仍输出 alpha（如轴对称高斯给出约 49.46° 的浮点噪声方向），
    # 极易被误当成方向测量。现在：alpha 字段为 null；原始相位另用 alpha_raw_* 命名，
    # 并在字段说明里明确它**不是有效方向**。
    return {
        "label": label, "r_min_m": float(r_min), "r_max_m": float(r_max),
        "band_pixels": band_pixels, "Pband": pband,
        "C3_real": float(c3.real), "C3_imag": float(c3.imag),
        "C3_abs": float(abs(c3)),
        "A3": a3,
        "alpha_wrapped_rad": (raw_alpha if reliable else None),
        "alpha_wrapped_deg": (math.degrees(raw_alpha) if reliable else None),
        "alpha_raw_rad": raw_alpha,
        "alpha_raw_deg": math.degrees(raw_alpha),
        "alpha_raw_note": ("arg(C3)/3 的原始数值；A3 未达阈值时它只是数值噪声，"
                           "**不是**有效方向，不得作为方向测量引用"),
        "alpha_period_deg": 360.0 / C3_ORDER,
        "reliable": reliable,
        "unreliable_reason": reason,
        "a3_min": float(a3_min), "order": C3_ORDER,
        "ambiguity_wrapped_diff_rad": float(ambiguity_wrapped_diff_rad),
        "definition": "C3=sum(Iraw*exp(+3i*theta))*dA; A3=|C3|/Pband; alpha=arg(C3)/3",
    }


def unwrap_angles_over_reliable_segments(
    wavelengths_m: Sequence[float],
    alphas_rad: Sequence[Optional[float]],
    reliables: Sequence[bool],
    max_abs_wrapped_diff_rad: float = 0.9 * math.pi,
) -> Dict[str, Any]:
    """仅在**连续可靠段**内展开 ``arg(C3)``，再除以 3 得到连续角。

    规则（第 8 节）：
    * 低可靠（``A3`` 未达阈值）或缺失值记 ``None``，**不**参与展开；
    * 不可靠间隙把序列切成若干段，段内独立展开，**不跨间隙续接方向**；
    * 记录相邻可靠样本的“三倍相位包裹差”：绝对值 ≥ 阈值时标记角度采样歧义。

    Returns
    -------
    dict
        ``segments``（每段的下标与展开后的三倍相位）、``alpha_unwrapped_rad``（与输入
        等长，不可靠处为 ``None``）、``ambiguous_pairs``、``has_ambiguity``。
    """
    lam = [float(x) for x in wavelengths_m]
    n = len(lam)
    if len(alphas_rad) != n or len(reliables) != n:
        raise ValueError("波长、角度与可靠标志长度必须一致")

    alpha_unwrapped: List[Optional[float]] = [None] * n
    segments: List[Dict[str, Any]] = []
    ambiguous_pairs: List[Dict[str, Any]] = []

    i = 0
    while i < n:
        if not reliables[i] or alphas_rad[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and reliables[j + 1] and alphas_rad[j + 1] is not None:
            j += 1
        idx = list(range(i, j + 1))
        triple = [C3_ORDER * float(alphas_rad[k]) for k in idx]
        unwrapped_triple = np.unwrap(np.asarray(triple, dtype=np.float64))
        for pos, k in enumerate(idx):
            alpha_unwrapped[k] = float(unwrapped_triple[pos] / C3_ORDER)
            if pos > 0:
                raw_a = C3_ORDER * float(alphas_rad[idx[pos - 1]])
                raw_b = C3_ORDER * float(alphas_rad[idx[pos]])
                d = abs(float(np.angle(np.exp(1j * (raw_b - raw_a)))))
                if d >= float(max_abs_wrapped_diff_rad):
                    ambiguous_pairs.append({
                        "from_index": idx[pos - 1], "to_index": idx[pos],
                        "from_wavelength_nm": round(lam[idx[pos - 1]] * 1e9),
                        "to_wavelength_nm": round(lam[idx[pos]] * 1e9),
                        "wrapped_triple_diff_rad": d,
                        "wrapped_triple_diff_over_pi": d / math.pi,
                        "note": "相邻三倍相位包裹差接近 π，角度采样存在歧义",
                    })
        segments.append({
            "indices": idx,
            "wavelengths_nm": [round(lam[k] * 1e9) for k in idx],
            "alpha_wrapped_rad": [float(alphas_rad[k]) for k in idx],
            "alpha_unwrapped_rad": [float(v) for v in (unwrapped_triple / C3_ORDER)],
            "span_deg": float(math.degrees(
                unwrapped_triple[-1] / C3_ORDER - unwrapped_triple[0] / C3_ORDER)),
            "direction": ("逆时针" if unwrapped_triple[-1] > unwrapped_triple[0]
                          else ("顺时针" if unwrapped_triple[-1] < unwrapped_triple[0]
                                else "无变化")),
        })
        i = j + 1

    return {
        "alpha_unwrapped_rad": alpha_unwrapped,
        "segments": segments,
        "ambiguous_pairs": ambiguous_pairs,
        "has_ambiguity": bool(ambiguous_pairs),
        "reliable_count": int(sum(1 for x in reliables if x)),
        "max_abs_wrapped_diff_rad": float(max_abs_wrapped_diff_rad),
        "note": ("alpha 具有 120° 周期；只在连续可靠段内展开，不跨不可靠间隙续接方向。"
                 "存在歧义时不能声称旋转方向已唯一确定。"),
    }


# --------------------------------------------------------------------------------------
# 绝对包围能量与 R50/R80
# --------------------------------------------------------------------------------------
def absolute_encircled_energy(intensity: np.ndarray, grid, p_in: float,
                              radii_m: Sequence[float]) -> List[Dict[str, Any]]:
    """``Eabs(R) = sum_{r<=R}(Iraw)*dA / Pin``（分母恒为输入功率）。

    ``Pin`` 必须由调用方按 ``sum(|u1|^2)*dxin*dyin`` 给出，**不接受** ``Pwindow``
    或 ``Pdisk`` 充当分母。
    """
    I = np.asarray(intensity, dtype=np.float64)
    if I.shape != grid.shape:
        raise ValueError("强度形状与网格不一致")
    if not np.all(np.isfinite(I)):
        raise ValueError("强度包含 NaN 或 Inf")
    pin = float(p_in)
    if not (math.isfinite(pin) and pin > 0):
        raise ValueError("Pin 必须为有限正数")
    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)
    dA = float(grid.cell_area)
    xmax, xmin = float(grid.x.coords.max()), float(grid.x.coords.min())
    ymax, ymin = float(grid.y.coords.max()), float(grid.y.coords.min())
    Rlimit = float(min(xmax, -xmin, ymax, -ymin))
    out = []
    for R in radii_m:
        R = float(R)
        if R <= 0:
            raise ValueError("评价半径必须为正")
        if R > Rlimit:
            # 超出完整圆盘上限的“圆盘”不是圆盘（会把方窗角落算进来），必须拒绝，
            # 不能给出一个看似有效的 Eabs。
            raise ValueError(
                "评价半径 R=%.6g m 超过完整圆盘上限 Rlimit=%.6g m；"
                "禁止把方窗角落当作圆盘能量" % (R, Rlimit))
        mask = r <= R
        pdisk = float(np.sum(I[mask]) * dA)
        out.append({
            "R_m": R, "R_um": R * 1e6,
            "pixels_in_disk": int(np.count_nonzero(mask)),
            "disk_within_full_circle": True,
            "Rlimit_m": Rlimit, "Rlimit_um": Rlimit * 1e6,
            "Pdisk": pdisk, "Pin": pin,
            "Eabs": pdisk / pin,
            "definition": ("Eabs(R)=sum_{r<=R}(Iraw)*dA/Pin，分母为输入功率；"
                           "R 必须 <= Rlimit=min(xmax,-xmin,ymax,-ymin) 才是完整圆盘"),
        })
    return out


def find_first_radius_reaching(intensity: np.ndarray, grid, p_in: float,
                               target: float) -> Dict[str, Any]:
    """``Eabs`` 第一次达到 ``target`` 的半径。

    实现规则（明确离散口径，**不插值**）：把 301² 各像素按**唯一半径**分组，得到
    严格递增的半径序列与累计能量，再取第一个 ``Eabs >= target`` 的半径。
    圆盘内未达到则返回 ``radius_m=None``、``status='not_reached'`` 并给出 ``Rlimit``。

    Returns
    -------
    dict
        含 ``radius_m``/``radius_um``/``status``/``Rlimit_m``/``radius_scale_m`` 等。

    Notes
    -----
    **R1 修正（2026-10-02）**：旧实现对整个方形窗排序累计，把方窗四个角落
    （``r`` 可超过 ``Rlimit``）也算进圆盘能量，导致

    * ``Eabs_at_Rlimit`` 取的是**方窗**累计终值而不是 150 μm 圆盘能量；
    * 角落只有能量时甚至会返回 ``reached`` 且半径 > ``Rlimit``（审核反例：
      强度只在 ``r>155 μm``，却返回 R80=185.31 μm）。

    现在先用**实际轴范围**求完整圆盘上限 ``Rlimit = min(xmax, −xmin, ymax, −ymin)``，
    **只保留 r ≤ Rlimit** 的像素再分组累计；``Eabs_at_Rlimit`` 因此对应真实圆盘。
    径向分辨率改用**明确的采样尺度**（网格间距与内切圆内最小非零半径），
    不再用浮点半径最小差（约 1e−20 m）冒充物理分辨率。
    """
    I = np.asarray(intensity, dtype=np.float64)
    if I.shape != grid.shape:
        raise ValueError("强度形状与网格不一致")
    if not np.all(np.isfinite(I)):
        raise ValueError("强度包含 NaN 或 Inf")
    pin = float(p_in)
    if not (math.isfinite(pin) and pin > 0):
        raise ValueError("Pin 必须为有限正数")
    tgt = float(target)
    if not (0.0 < tgt < 1.0):
        raise ValueError("目标包围能量比例必须介于 0 与 1 之间")

    # 完整圆盘上限必须由**实际轴范围**求，不能假设对称
    xmax, xmin = float(grid.x.coords.max()), float(grid.x.coords.min())
    ymax, ymin = float(grid.y.coords.max()), float(grid.y.coords.min())
    Rlimit = float(min(xmax, -xmin, ymax, -ymin))
    if not (Rlimit > 0):
        raise ValueError("由轴范围得到的完整圆盘上限必须为正")

    X, Y = grid.meshgrid()
    r_all = np.hypot(X, Y)
    # 只保留完整圆盘内（r <= Rlimit）的像素
    inside = r_all <= Rlimit
    r = r_all[inside].ravel()
    vals = I[inside].ravel()
    dA = float(grid.cell_area)
    order = np.argsort(r, kind="stable")
    r_sorted = r[order]
    cum = np.cumsum(vals[order]) * dA / pin
    # 相同半径的像素合并：取该半径的最后一个累计值
    last_of_radius = np.r_[r_sorted[1:] != r_sorted[:-1], True]
    r_unique = r_sorted[last_of_radius]
    e_unique = cum[last_of_radius]

    # 明确的采样尺度：不用浮点半径最小差当物理分辨率
    spacing = float(min(grid.dx, grid.dy))
    nonzero = r_unique[r_unique > 0]
    radius_scale = float(nonzero.min()) if nonzero.size else spacing

    base = {
        "target": tgt,
        "Rlimit_m": Rlimit, "Rlimit_um": Rlimit * 1e6,
        "pixels_inside_Rlimit": int(np.count_nonzero(inside)),
        "pixels_total": int(r_all.size),
        "radius_scale_m": radius_scale,
        "radius_scale_note": ("径向采样尺度：取网格间距与圆盘内最小非零半径中的较小者；"
                              "半径按像素唯一值离散分组，不插值"),
        "definition": ("先在 r<=Rlimit=min(xmax,-xmin,ymax,-ymin) 的完整圆盘内取像素，"
                       "再按唯一半径分组求累计，取第一个 Eabs>=target 的半径；"
                       "圆盘内未达到记 not_reached，不插值"),
    }
    hit = np.flatnonzero(e_unique >= tgt)
    if hit.size == 0:
        base.update({
            "radius_m": None, "radius_um": None,
            "status": "not_reached",
            "Eabs_at_radius": None,
            "Eabs_at_Rlimit": float(e_unique[-1]) if e_unique.size else 0.0,
        })
        return base
    k = int(hit[0])
    base.update({
        "radius_m": float(r_unique[k]),
        "radius_um": float(r_unique[k] * 1e6),
        "status": "reached",
        "Eabs_at_radius": float(e_unique[k]),
        "Eabs_at_Rlimit": float(e_unique[-1]),
    })
    return base


# --------------------------------------------------------------------------------------
# 尺寸（峰值阈值面积）与质心
# --------------------------------------------------------------------------------------
def threshold_area_radius(intensity: np.ndarray, grid, quantile: float,
                          peak: Optional[float] = None,
                          r_max: Optional[float] = None) -> Dict[str, Any]:
    """峰值阈值面积 ``Aq = sum_{ROI 且 I >= q*Ipeak} dA`` 与等效半径 ``sqrt(Aq/pi)``。

    ``Ipeak`` 取**同一完整方窗内**的峰值（不是 ROI 内峰值），以保证口径固定。
    """
    I = np.asarray(intensity, dtype=np.float64)
    if I.shape != grid.shape:
        raise ValueError("强度形状与网格不一致")
    if not np.all(np.isfinite(I)):
        raise ValueError("强度包含 NaN 或 Inf")
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError("阈值分位必须属于 (0, 1]")
    ipeak = float(np.max(I)) if peak is None else float(peak)
    if not (math.isfinite(ipeak) and ipeak > 0):
        raise ValueError("峰值强度必须为有限正数")

    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)
    roi = np.ones(I.shape, dtype=bool) if r_max is None else (r <= float(r_max))
    selected = roi & (I >= q * ipeak)
    dA = float(grid.cell_area)
    area = float(np.count_nonzero(selected) * dA)
    req = math.sqrt(area / math.pi) if area > 0 else 0.0

    # 峰值位置（在完整方窗内），用于发现峰值跑到 ROI 之外
    pk = np.unravel_index(int(np.argmax(I)), I.shape)
    peak_pos = (float(grid.x.coords[pk[1]]), float(grid.y.coords[pk[0]]))
    outside = bool(np.hypot(peak_pos[0], peak_pos[1]) > float(r_max)) if r_max else False
    components = _count_connected_components(selected)
    return {
        "quantile": q,
        "Ipeak": ipeak,
        "r_max_m": None if r_max is None else float(r_max),
        "selected_pixels": int(np.count_nonzero(selected)),
        "area_m2": area, "area_um2": area * 1e12,
        "r_equiv_m": req, "r_equiv_um": req * 1e6,
        "peak_position_m": peak_pos,
        "peak_position_um": (peak_pos[0] * 1e6, peak_pos[1] * 1e6),
        "peak_outside_roi": outside,
        "connected_components_8": components,
        "definition": ("Aq=sum_{ROI且I>=q*Ipeak}*dA；r_eq=sqrt(Aq/pi)；"
                       "Ipeak 取完整方窗内峰值；8 连通统计连通分量"),
    }


def _count_connected_components(mask: np.ndarray) -> int:
    """8 连通分量计数（不依赖 scipy.ndimage 的可选加速，纯 numpy 双扫描）。"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    ny, nx = m.shape
    labels = np.zeros(m.shape, dtype=np.int64)
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for j in range(ny):
        for i in range(nx):
            if not m[j, i]:
                continue
            neigh = []
            for dj, di in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and labels[jj, ii] > 0:
                    neigh.append(int(labels[jj, ii]))
            if not neigh:
                labels[j, i] = nxt
                parent[nxt] = nxt
                nxt += 1
            else:
                mn = min(neigh)
                labels[j, i] = mn
                for nb in neigh:
                    union(mn, nb)
    roots = {find(int(v)) for v in np.unique(labels) if v > 0}
    return len(roots)


def fixed_window_centroid(intensity: np.ndarray, grid) -> Dict[str, Any]:
    """固定方窗内的强度质心（裁切/不对称诊断）。

    **不是**无限平面质心；仅作诊断，不用它平移 PSF 使尺寸看起来稳定。
    """
    I = np.asarray(intensity, dtype=np.float64)
    X, Y = grid.meshgrid()
    total = float(np.sum(I))
    if not (math.isfinite(total) and total > 0):
        return {"cx_m": None, "cy_m": None, "cx_um": None, "cy_um": None,
                "note": "总强度非正，质心未定义"}
    cx = float(np.sum(I * X) / total)
    cy = float(np.sum(I * Y) / total)
    return {"cx_m": cx, "cy_m": cy, "cx_um": cx * 1e6, "cy_um": cy * 1e6,
            "note": "固定方窗内强度质心，不是无限平面质心；仅作裁切/不对称诊断"}


def band_energy_fraction(intensity: np.ndarray, grid, r_min: float, r_max: float,
                         p_in: float) -> Dict[str, Any]:
    """环带内能量占**输入功率**的比例（明确口径，不称为“旁瓣率”）。"""
    I = np.asarray(intensity, dtype=np.float64)
    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)
    band = (r >= float(r_min)) & (r <= float(r_max))
    pband = float(np.sum(I[band]) * grid.cell_area)
    return {"r_min_m": float(r_min), "r_max_m": float(r_max),
            "Pband": pband, "Pin": float(p_in),
            "band_energy_fraction_of_Pin": pband / float(p_in),
            "definition": "环带能量/Pin；不是旁瓣率，也不把未捕获能量称为旁瓣"}


def height_fingerprint_from_arrays(design_type: str, grid, delta_h: np.ndarray,
                                   mask: np.ndarray, params: Dict[str, Any]) -> str:
    """**独立重算**高度内容指纹（不依赖文件里保存的指纹字段）。

    R2 修正（2026-10-02）：保存后重读必须能自己算出指纹，否则“改高度 + 保留旧指纹字段”
    就能骗过核对。这里按与 ``DOEHeightProfile.compute_fingerprint`` **完全相同**的规范
    重建：设计类型 + 网格间距/形状 + 参数 JSON + 坐标轴 + 高度 + 掩膜，全部小端序。
    """
    hasher = hashlib.sha256()
    clean_params = {
        "design_type": str(design_type),
        "dx_m": float(grid.dx),
        "dy_m": float(grid.dy),
        "shape_yx": [int(grid.y.n), int(grid.x.n)],
        "params": {k: float(v) if isinstance(v, (int, float)) else str(v)
                   for k, v in sorted(params.items())},
    }
    hasher.update(json.dumps(clean_params, sort_keys=True).encode("utf-8"))
    hasher.update(np.ascontiguousarray(grid.x.coords, dtype="<f8").tobytes())
    hasher.update(np.ascontiguousarray(grid.y.coords, dtype="<f8").tobytes())
    hasher.update(np.ascontiguousarray(delta_h, dtype="<f8").tobytes())
    hasher.update(np.ascontiguousarray(mask, dtype="<f8").tobytes())
    return hasher.hexdigest()


def summarize_size_stability(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """报告尺寸指标的范围、均值与变异系数，并写明有效样本数。"""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n_valid = len(vals)
    if n_valid == 0:
        return {"n_valid": 0, "min": None, "max": None, "mean": None,
                "std": None, "cv": None, "range": None,
                "note": "无有效样本，不做稳定性结论"}
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    return {
        "n_valid": n_valid,
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": mean, "std": std,
        "cv": (std / mean) if mean != 0 else None,
        "range": float(arr.max() - arr.min()),
        "note": "不预设“变化小于 5%% 才算稳定”的判据；如实报告稳定或不稳定",
    }


# --------------------------------------------------------------------------------------
# 单场指标汇总
# --------------------------------------------------------------------------------------
def psf_metrics(intensity: np.ndarray, grid, p_in: float, p_window_raw: float,
                cfg_rotation: Dict[str, Any], cfg_size: Dict[str, Any],
                cfg_energy: Dict[str, Any], a3_min: float,
                ambiguity_rad: float) -> Dict[str, Any]:
    """把第 8、9 节的全部分析口径应用到一个 PSF 上，返回机器可读字典。"""
    I = np.asarray(intensity, dtype=np.float64)
    out: Dict[str, Any] = {
        "Pin": float(p_in),
        "Pwindow_recomputed": float(np.sum(I) * grid.cell_area),
        "Pwindow_saved": float(p_window_raw),
        "eta_window": float(p_window_raw) / float(p_in) if p_in > 0 else None,
    }
    bands = {}
    for key, band_cfg in (("primary", cfg_rotation["primary"]),
                          ("sensitivity", cfg_rotation["sensitivity"])):
        bands[key] = c3_rotation_metric(
            I, grid, band_cfg["r_min_m"], band_cfg["r_max_m"],
            a3_min=a3_min, ambiguity_wrapped_diff_rad=ambiguity_rad,
            label=band_cfg.get("label", key))
    out["rotation_bands"] = bands

    radii = [float(x) for x in cfg_energy["capture_radii_m"]]
    out["abs_encircled"] = absolute_encircled_energy(I, grid, p_in, radii)
    out["R50"] = find_first_radius_reaching(I, grid, p_in, 0.5)
    out["R80"] = find_first_radius_reaching(I, grid, p_in, 0.8)

    roi_rmax = float(cfg_size["r_max_m"])
    ipeak = float(np.max(I))
    shapes = {}
    for q in cfg_size["quantiles"]:
        shapes["q%g" % q] = threshold_area_radius(I, grid, q, peak=ipeak, r_max=roi_rmax)
    out["shape"] = shapes
    out["centroid"] = fixed_window_centroid(I, grid)
    out["band_energy"] = {
        "primary": band_energy_fraction(
            I, grid, cfg_rotation["primary"]["r_min_m"],
            cfg_rotation["primary"]["r_max_m"], p_in),
        "sensitivity": band_energy_fraction(
            I, grid, cfg_rotation["sensitivity"]["r_min_m"],
            cfg_rotation["sensitivity"]["r_max_m"], p_in),
    }
    out["peak_intensity"] = ipeak
    out["Rlimit_m"] = float(min(np.abs(grid.x.coords).max(), np.abs(grid.y.coords).max()))
    out["definitions"] = {
        "eta_window": "Pwindow/Pin；0<eta<=1.02 只作粗健康检查，不证明能量守恒精确成立",
        "Eabs": "sum_{r<=R}(Iraw)*dA/Pin，分母恒为 Pin",
        "R50_R80": "Eabs 第一次达到 0.5/0.8 的半径；未达到记 null/not_reached",
        "shape": "峰值相对形状指标 Aq 与 r_eq=sqrt(Aq/pi)，不是通光效率",
        "rotation": "C3=sum(Iraw*exp(+3i*theta))*dA；A3=|C3|/Pband；alpha=arg(C3)/3，120°周期",
    }
    return out
