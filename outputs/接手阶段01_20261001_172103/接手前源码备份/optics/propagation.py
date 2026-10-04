# -*- coding: utf-8 -*-
"""Fresnel 标量衍射传播（阶段 01 修正版）。

本模块提供**四条互相独立、可交叉验证**的计算路径，并在文档与元数据中明确区分：

1. :func:`fresnel_kernel_matrix` —— **完整位移核**的矩阵求积。
   这是式(4) 的离散形式，是交叉核验的基准::

       u₂(x,y) = exp(ikz)/(iλz) · Σ u₁(x′,y′)
                 · exp[iπ((x−x′)²+(y−y′)²)/(λz)] · Δx′Δy′

   核里已经含有 ``x′²+y′²``，因此**不得**再给 ``u₁`` 预乘一次输入二次相位
   （阶段 01 初审版正是这样做的，属于重复相位错误，见
   ``docs/revision_notes.md`` 与 ``docs/reproduction_log.md``）。

2. :func:`fresnel_fft` —— **单次 FFT**，用中心化频率坐标与面积权重::

       V  = 中心零填充后的 u₁ · exp[iπ(x′²+y′²)/(λz)]
       F  = fftshift(fft2(ifftshift(V)))
       u₂ = F · exp[iπ(x²+y²)/(λz)] · Δx′Δy′ · exp(ikz)/(iλz)
       x  = λz · fftshift(fftfreq(Mx, Δx′))，  y 同理

   由式(4) 把 ``(x−x′)²`` 展开得到（输入二次相位、交叉 Fourier 核、输出二次相位
   各出现一次）。1 与 2 在**同一离散输入、同一原生输出网格**上是离散代数恒等关系，
   实测相对 L2 ≈ 3e−15。

3. :func:`fresnel_transfer_fresnel` —— 频域 **Fresnel 近似**传递函数
   ``H = exp(ikz)·exp(−iπλz f²)``（输出与输入同网格）。
   **不要**把它与精确角谱混称。

4. :func:`fresnel_angular_spectrum` —— **精确角谱** ``H = exp(ikz√(1−λ²f²))``
   （含倏逝波指数衰减）。输出同样与输入同网格。

2/3/4 都受 FFT 周期性回绕影响；1 直接对指定评价点求和，没有回绕，
代价是 ``O(N_out · N_in)``。

## 相位与尺度约定

* Fourier 变换取**负指数**约定。
* ``include_global_phase`` **只**控制 ``exp(ikz)`` 一个因子：True（默认）保留，
  False 去掉；它从不涉及 ``1/(iλz)``。默认与直接积分一致。
  两种谱传播都遵循这一条：
  频域 Fresnel 近似为 ``H_true = exp(ikz)·exp(−iπλz f²)`` 与
  ``H_false = exp(−iπλz f²)``；
  精确角谱为 ``H_true = exp(ikz·s)`` 与 ``H_false = exp(ikz·s)·exp(−ikz)``，
  其中 ``s = √(1−λ²(fx²+fy²))``，对倏逝分量取使 z>0 衰减的复支路。
  因此两者都满足 ``u_true ≈ exp(ikz)·u_false``（相对 L2 ≤1e−10）。
* 默认复场**保留输出平面二次相位** ``exp[iπ(x²+y²)/(λz)]``，即
  ``u_full = C·F·exp(+i q_out)``。
  ``omit_output_quadratic_phase=True`` 返回 **reduced 场** ``u_red = C·F``：
  它就是完整场**除掉一次** ``exp(+i q_out)`` 的结果，因此
  ``u_full = u_red·exp(+i q_out)``、``u_red = u_full·exp(−i q_out)``。
  两种表示在**本平面**的强度完全相同（纯相位因子），但复场相位不同，
  再次传播时必须用 full 场（默认值即为 full）。
  **旧版这里错误地乘了 ``exp(−i q_out)``**，使结果与完整场相差 ``exp(−2i q_out)``，
  第二轮审核已定位并修正；回归测试用全场范数（不逐点除场）断言上述两条关系。
* ``Δx′Δy′`` 面积权重与 ``1/(λz)`` 尺度因子**任何情况下都不允许删除**。

## 采样判据

直接求和的被积函数含三部分相位：输入二次相位、交叉项、输出二次相位。
:func:`sampling_diagnostics` 按**实际输出窗口**给出各项的相位步进与保守和，
并同时报告精确高阶光程误差；不使用"只输入二次相位单项"作为通用精度保证。

所有长度单位均为 **米（m）**。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .coordinates import Axis1D, Grid2D, memory_estimate_mb

__all__ = [
    "FresnelConfig",
    "PropagationResult",
    "fresnel_kernel_matrix",
    "fresnel_kernel_eval",
    "fresnel_kernel_separable",
    "fresnel_fft",
    "fresnel_transfer_fresnel",
    "fresnel_angular_spectrum",
    "input_plane_quadratic_phase",
    "output_plane_quadratic_phase",
    "path_difference_exact_m",
    "path_difference_leading_m",
    "third_order_phase_error_rad",
    "quadratic_phase_max_step_rad",
    "sampling_diagnostics",
    "fresnel_number",
    "axial_fresnel_number",
    "pad_size_for",
]


# --------------------------------------------------------------------------------------
# 解析辅助量
# --------------------------------------------------------------------------------------
def fresnel_number(aperture_size: float, wavelength: float, distance: float) -> float:
    """Fresnel 数 ``N_F = a²/(λ z)``（``a`` 为特征横向尺度，m）。"""
    return float(aperture_size) ** 2 / (float(wavelength) * float(distance))


def axial_fresnel_number(diameter: float, wavelength: float, focal_length: float) -> float:
    """圆孔光束的轴向 Fresnel 数 ``D²/(4λf)``。

    **注意**：在"理想二次相位薄透镜 + z=f"的焦平面上，透镜相位
    ``exp(−iπr′²/(λf))`` 与传播展开式的输入二次相位 ``exp(+iπr′²/(λf))`` 相消，
    焦平面是孔径的 Fourier 强度，圆孔给出 Airy 图样。
    因此该量**不是** Airy 校验的适用条件，只用于报告与历史对照。
    """
    return float(diameter) ** 2 / (4.0 * float(wavelength) * float(focal_length))


def input_plane_quadratic_phase(grid: Grid2D, wavelength: float, distance: float) -> np.ndarray:
    """输入平面二次相位 ``exp[+iπ(x′²+y′²)/(λz)]``（展开核写法中的一部分）。"""
    r2 = grid.radius() ** 2
    return np.exp(1j * np.pi * r2 / (float(wavelength) * float(distance)))


def output_plane_quadratic_phase(grid: Grid2D, wavelength: float, distance: float) -> np.ndarray:
    """输出平面二次相位 ``exp[+iπ(x²+y²)/(λz)]``。"""
    r2 = grid.radius() ** 2
    return np.exp(1j * np.pi * r2 / (float(wavelength) * float(distance)))


def path_difference_exact_m(rho, z: float) -> np.ndarray:
    """Fresnel 近似的精确光程残差 ``√(z²+ρ²) − z − ρ²/(2z)``（m）。

    用稳定代数形式避免浮点抵消::

        √(z²+ρ²) − z = ρ² / (√(z²+ρ²) + z)
        ΔL = ρ²/(√(z²+ρ²)+z) − ρ²/(2z) = −ρ⁴ / [2z·(√(z²+ρ²)+z)²]

    返回值恒为负或零，量级为 ``−ρ⁴/(8z³)``。
    """
    rho2 = np.asarray(rho, dtype=np.float64) ** 2
    s = np.sqrt(z * z + rho2) + z
    return -rho2 ** 2 / (2.0 * z * s ** 2)


def path_difference_leading_m(rho, z: float) -> np.ndarray:
    """四次近似常数项 ``ΔL ≈ −ρ⁴/(8z³)``（m）。仅用于与精确式对照。"""
    rho2 = np.asarray(rho, dtype=np.float64) ** 2
    return -rho2 ** 2 / (8.0 * z ** 3)


def third_order_phase_error_rad(rho_max: float, distance: float, wavelength: float) -> float:
    """给定最大横向距离 ``ρ_max`` 时被忽略的高阶光程项带来的相位误差（rad）。

    ``相位误差 = |(2π/λ)·ΔL_exact(ρ_max)|``，用**精确**路径差而不是四次近似。
    本项目参考容忍度 0.1 rad（非论文数值）。
    """
    dl = path_difference_exact_m(rho_max, distance)
    return float(abs(2.0 * np.pi / wavelength * dl))


def quadratic_phase_max_step_rad(half_width: float, spacing: float, wavelength: float,
                                 distance: float) -> float:
    """二次相位 ``πr²/(λz)`` 在 ``|x| = half_width`` 处的一维最大相邻相位步进。

    ``|Δφ| ≈ π(2·x_max·Δ + Δ²)/(λz)``。**只描述该单项**，
    不能单独作为整个传播的通用精度保证（见 :func:`sampling_diagnostics`）。
    """
    lamz = float(wavelength) * float(distance)
    return float(np.pi * (2.0 * float(half_width) * float(spacing) + float(spacing) ** 2) / lamz)


def pad_size_for(n: int, factor: int) -> int:
    """按倍率算偶数零填充长度。"""
    m = int(n) * int(factor)
    return m + (m % 2)


def sampling_diagnostics(grid_in: Grid2D, wavelength: float, distance: float,
                         out_half_width_x: float, out_half_width_y: float,
                         dx_out: Optional[float] = None, dy_out: Optional[float] = None,
                         max_step_rad: float = 0.5) -> Dict[str, object]:
    """采样诊断：把**对积分变量的采样**与**输出表示的采样**彻底分开，且按 x/y 分别报告。

    设置口径（第二轮修正）
    ----------------------
    1. **输入积分方向**：判据要覆盖"完整被积函数"，即输入二次相位 + 交叉项。
       对 ``x′`` 求导，两项的步进直接相加::

           输入二次相位  d/dx′ [π(x′²+y′²)/(λz)]      = 2πx′/(λz)
           交叉项        d/dx′ [−2π(xx′+yy′)/(λz)]    = −2πx/(λz)

       在输入支撑 |x′| ≤ a_x 与输出窗口 |x| ≤ b_x 上取最坏组合，相邻样点 ``Δx′``
       的保守核界为::

           input_kernel_step_x = π[2(a_x + b_x)Δx′ + Δx′²]/(λz)

       ``y`` 方向同理，**分别报告**，避免粗的一方向被细的一方向掩盖。
    2. **输出表示方向**：输出二次相位 ``q_out = π(x²+y²)/(λz)`` 对**固定输出点**只是常量，
       它对求和变量**没有**步进，因此绝不能当作输入求积的变化项（旧版正是这样混用的）。
       它只决定"输出场本身在给定采样间距下是否被欠采样"::

           output_quadratic_step_x = π[2b_x·Δx_out + Δx_out²]/(λz)

       ``Δx_out`` 必须用**实际输出间距**；未提供时该方向标记为"未评估"
       （``None``），**不得用输入间距冒充**。
    3. 支撑范围默认取整张方形输入窗的半宽 ``(n//2)·Δ``，对非零圆孔而言是**保守上界**，
       在返回值里用 ``support_is_conservative_upper_bound=True`` 标注。
    4. 三项都只是**保守提示**，不是保证；最终以连续参考解与真实收敛研究为准。
       近轴高阶误差超阈值时只解释近轴条件，不能据此判定焦平面 Airy 不适用。

    Parameters
    ----------
    grid_in : Grid2D
        **实际参与求和**的输入网格（零填充区振幅为 0，应传未填充网格）。
    out_half_width_x, out_half_width_y : float
        输出评价窗口半宽（m）。
    dx_out, dy_out : float, optional
        实际输出网格间距（m）。缺省时对应方向的"输出表示"一项返回 ``None``（未评估）。
    max_step_rad : float
        输入积分方向的保守提示阈值，默认 0.5 rad（本项目建议值，非论文数值）。
    """
    lam = float(wavelength)
    z = float(distance)
    lamz = lam * z
    dx, dy = grid_in.dx, grid_in.dy
    ax = (grid_in.x.n // 2) * dx       # 输入支撑半宽（保守：方形窗上界）
    ay = (grid_in.y.n // 2) * dy
    bx = float(out_half_width_x)
    by = float(out_half_width_y)

    def _out_step(b: float, d_out: Optional[float]) -> Optional[float]:
        if d_out is None:
            return None
        return float(np.pi * (2.0 * b * float(d_out) + float(d_out) ** 2) / lamz)

    in_x = float(np.pi * (2.0 * (ax + bx) * dx + dx ** 2) / lamz)
    in_y = float(np.pi * (2.0 * (ay + by) * dy + dy ** 2) / lamz)
    out_x = _out_step(bx, dx_out)
    out_y = _out_step(by, dy_out)
    worst_in = max(in_x, in_y)

    # 只用于报告的单项分解（同一方向、同一最坏组合）
    input_only_x = float(np.pi * (2.0 * ax * dx + dx ** 2) / lamz)
    input_only_y = float(np.pi * (2.0 * ay * dy + dy ** 2) / lamz)
    cross_only_x = float(2.0 * np.pi * bx * dx / lamz)
    cross_only_y = float(2.0 * np.pi * by * dy / lamz)

    rho_max = float(np.hypot(ax + bx, ay + by))
    out_evaluated = out_x is not None and out_y is not None
    return {
        # ---- 输入支撑与输出窗口 ----
        "input_half_width_m": float(ax),
        "input_half_width_y_m": float(ay),
        "output_half_width_m": float(bx),
        "output_half_width_y_m": float(by),
        "support_is_conservative_upper_bound": True,
        "support_note": ("输入支撑按整张方形输入窗半宽 (n//2)·Δ 计；"
                         "对非零圆孔等受限孔径这是保守上界。"),
        # ---- 输入积分方向（x/y 分别） ----
        "input_kernel_step_x_rad": in_x,
        "input_kernel_step_y_rad": in_y,
        "input_kernel_step_worst_rad": worst_in,
        "input_quadratic_only_x_rad": input_only_x,
        "input_quadratic_only_y_rad": input_only_y,
        "cross_term_only_x_rad": cross_only_x,
        "cross_term_only_y_rad": cross_only_y,
        "input_integral_formula": "π[2(a+b)Δx′+Δx′²]/(λz)，a=输入支撑半宽，b=输出窗口半宽；y 同理",
        # ---- 输出表示方向（x/y 分别，需实际输出间距） ----
        "output_quadratic_step_x_rad": out_x,
        "output_quadratic_step_y_rad": out_y,
        "output_quadratic_evaluated": bool(out_evaluated),
        "output_spacing_x_m": None if dx_out is None else float(dx_out),
        "output_spacing_y_m": None if dy_out is None else float(dy_out),
        "output_representation_formula": "π[2b·Δout+b²]/(λz)，b=输出窗口半宽；未给输出间距时为未评估",
        "output_quadratic_note": ("输出二次相位对固定输出点只是常量，对求和变量没有步进；"
                                  "它只描述输出场本身的采样，不是输入求积误差。"),
        # ---- 旧字段名（向后兼容，含义已更正） ----
        "input_quadratic_step_rad": in_x,
        "cross_term_step_rad": cross_only_x,
        "output_quadratic_step_rad": out_x,
        "max_total_step_rad": worst_in,
        "threshold_rad": float(max_step_rad),
        "ok": bool(worst_in <= float(max_step_rad)),
        "rho_max_m": rho_max,
        "third_order_phase_error_rad": third_order_phase_error_rad(rho_max, z, lam),
        "leading_fourth_order_phase_error_rad": float(
            abs(2.0 * np.pi / lam * path_difference_leading_m(rho_max, z))),
        "paraxial_tolerance_rad": 0.1,
        "note": ("输入积分方向的核界已按 x/y 分别保守估计；ok 只是保守提示，"
                 "不是精度保证。输出表示方向在未给实际输出间距时标为未评估。"),
    }


# --------------------------------------------------------------------------------------
# 配置与结果容器
# --------------------------------------------------------------------------------------
@dataclass
class FresnelConfig:
    """传播配置（长度单位：米）。

    Parameters
    ----------
    wavelength, distance : float
        λ（m）与 z（m，沿 +z）。
    padded_size : int or (My, Mx)
        FFT 长度（含零填充），必须为偶数且 ≥ 输入样点数。
    include_global_phase : bool
        是否保留 ``exp(ikz)``。**只**控制这一个因子，默认 True（与直接积分一致）。
    omit_output_quadratic_phase : bool
        若为 True，返回 ``u₂·exp[−iπ(x²+y²)/(λz)]``：把输出二次相位解析地除掉后的
        场表示。**不是同一个物理复场**，只用于诊断；默认 False。
    """

    wavelength: float
    distance: float
    padded_size: object
    include_global_phase: bool = True
    omit_output_quadratic_phase: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength 必须是有限正数（m）")
        if not np.isfinite(self.distance) or self.distance <= 0:
            raise ValueError("distance 必须是有限正数（m）")

    @property
    def wavenumber(self) -> float:
        return 2.0 * np.pi / float(self.wavelength)

    @property
    def padded_shape(self) -> Tuple[int, int]:
        if isinstance(self.padded_size, (tuple, list, np.ndarray)):
            if len(self.padded_size) != 2:
                raise ValueError("padded_size 必须是 int 或 (My, Mx)")
            return (int(self.padded_size[0]), int(self.padded_size[1]))
        return (int(self.padded_size), int(self.padded_size))

    def to_metadata(self) -> Dict[str, object]:
        return {
            "wavelength_m": float(self.wavelength),
            "distance_m": float(self.distance),
            "padded_shape_yx": list(self.padded_shape),
            "include_global_phase": bool(self.include_global_phase),
            "omit_output_quadratic_phase": bool(self.omit_output_quadratic_phase),
            "phase_convention": ("Fourier 取负指数；include_global_phase 只控制 exp(ikz)；"
                                 "默认保留输出二次相位"),
        }


@dataclass
class PropagationResult:
    """传播结果（复振幅 + 物理坐标 + 元数据）。"""

    field: np.ndarray
    grid: Grid2D
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.field) ** 2

    @property
    def power(self) -> float:
        return float(np.sum(np.abs(self.field) ** 2) * self.grid.cell_area)

    def describe(self) -> Dict[str, object]:
        d = dict(self.meta)
        d["output_grid"] = self.grid.describe()
        return d


def _check_shapes(field_in: np.ndarray, grid_in: Grid2D) -> np.ndarray:
    f = np.asarray(field_in)
    if f.ndim != 2:
        raise ValueError("field_in 必须是二维数组")
    if f.shape != grid_in.shape:
        raise ValueError("field_in.shape=%r 与 grid_in.shape=%r 不一致" % (f.shape, grid_in.shape))
    return f


def _centered_coords(n: int, d: float) -> np.ndarray:
    """以 0 为中心的一维坐标（偶数 n 时 0 位于 n//2 号样点）。"""
    return (np.arange(n, dtype=np.float64) - (n // 2)) * float(d)


# --------------------------------------------------------------------------------------
# 路径 1：完整位移核的矩阵求积
# --------------------------------------------------------------------------------------
def fresnel_kernel_matrix(field_in: np.ndarray, grid_in: Grid2D, wavelength: float,
                          distance: float, x_out: np.ndarray, y_out: np.ndarray,
                          include_global_phase: bool = True,
                          block: int = 0) -> np.ndarray:
    """完整位移核的矩阵求积（式(4) 的离散形式；**不**预乘输入二次相位）。

    Returns
    -------
    ndarray，形状 ``(len(y_out), len(x_out))``。
    """
    f = _check_shapes(field_in, grid_in)
    lam = float(wavelength)
    z = float(distance)
    lamz = lam * z
    xp = grid_in.x.coords
    yp = grid_in.y.coords
    xo = np.atleast_1d(np.asarray(x_out, dtype=np.float64))
    yo = np.atleast_1d(np.asarray(y_out, dtype=np.float64))

    pref = (grid_in.dx * grid_in.dy) / (1j * lamz)
    if include_global_phase:
        pref = pref * np.exp(1j * 2.0 * np.pi / lam * z)

    px = np.exp(1j * np.pi * (xo[:, None] - xp[None, :]) ** 2 / lamz)
    if block and block > 0:
        out = np.empty((len(yo), len(xo)), dtype=np.complex128)
        for i0 in range(0, len(yo), int(block)):
            i1 = min(i0 + int(block), len(yo))
            py = np.exp(1j * np.pi * (yo[i0:i1, None] - yp[None, :]) ** 2 / lamz)
            out[i0:i1] = (py @ f) @ px.T
    else:
        py = np.exp(1j * np.pi * (yo[:, None] - yp[None, :]) ** 2 / lamz)
        out = (py @ f) @ px.T
    return out * pref


def fresnel_kernel_eval(field_in: np.ndarray, grid_in: Grid2D, wavelength: float,
                        distance: float, x_eval: np.ndarray, y_eval: np.ndarray,
                        include_global_phase: bool = True) -> np.ndarray:
    """:func:`fresnel_kernel_matrix` 的别名（保留旧名，便于对照历史结论）。"""
    return fresnel_kernel_matrix(field_in, grid_in, wavelength, distance,
                                 x_eval, y_eval, include_global_phase=include_global_phase)


# --------------------------------------------------------------------------------------
# 路径 2：单次 FFT
# --------------------------------------------------------------------------------------
def fresnel_kernel_separable(field_in: np.ndarray, grid_in: Grid2D, wavelength: float,
                             distance: float, x_out: np.ndarray, y_out: np.ndarray,
                             include_global_phase: bool = True) -> np.ndarray:
    """完整位移核的**可分离**求值（与 :func:`fresnel_kernel_matrix` 数学等价）。

    展开核后 ``(x−x′)²`` 对 ``x′`` 只有线性耦合，因此双重求和可写成两次一维矩阵乘法::

        u₂ = (Ey · u₁ · Exᵀ) · Δx′Δy′/(iλz) · exp(ikz)
        Ex[p,m] = exp[iπ(x_p² − 2x_p x′_m + x′_m²)/(λz)]
        Ey[q,n] = exp[iπ(y_q² − 2y_q y′_n + y′_n²)/(λz)]

    代价为 ``O(N_out · (Nx′ + Ny′))`` 而不是 ``O(N_out · Nx′Ny′)``，
    在评价点数很少、输入很大时（例如论文尺度的焦平面中心截线）是唯一可行的直接求值。
    与 :func:`fresnel_kernel_matrix` 的一致性由测试断言（相对 L2 ≤1e−12）。
    """
    f = _check_shapes(field_in, grid_in)
    lam = float(wavelength)
    z = float(distance)
    lamz = lam * z
    xp = grid_in.x.coords
    yp = grid_in.y.coords
    xo = np.atleast_1d(np.asarray(x_out, dtype=np.float64))
    yo = np.atleast_1d(np.asarray(y_out, dtype=np.float64))
    Ex = np.exp(1j * np.pi * (xo[:, None] ** 2 - 2.0 * xo[:, None] * xp[None, :]
                              + xp[None, :] ** 2) / lamz)
    Ey = np.exp(1j * np.pi * (yo[:, None] ** 2 - 2.0 * yo[:, None] * yp[None, :]
                              + yp[None, :] ** 2) / lamz)
    val = (Ey @ f) @ Ex.T
    pref = (grid_in.dx * grid_in.dy) / (1j * lamz)
    if include_global_phase:
        pref = pref * np.exp(1j * 2.0 * np.pi / lam * z)
    return val * pref


def fresnel_fft(field_in: np.ndarray, grid_in: Grid2D,
                config: FresnelConfig) -> PropagationResult:
    """单次 FFT 的 Fresnel 传播（中心化频率 + 面积权重 + 输出二次相位）。

    与 :func:`fresnel_kernel_matrix` 在同一离散输入、同一原生输出网格上是离散代数
    恒等关系（实测相对 L2 ≈ 3e−15，全输出功率相对误差 ≈ 1e−14）。

    **输入奇偶性**：``Mx/My`` 强制为偶数，``x = λz·fftshift(fftfreq(Mx, Δx′))``
    总是给出以 0 为中心、含 ``M//2`` 号零样点的坐标，与输入的奇偶性无关；
    输入奇偶性只决定 ``(n//2)·Δ`` 落点。奇数与偶数输入都支持，见
    ``tests/test_propagation.py::TestNonSquareAndOddGrid``。
    """
    t0 = time.perf_counter()
    f = _check_shapes(field_in, grid_in)
    my, mx = config.padded_shape
    ny, nx = f.shape
    if my < ny or mx < nx:
        raise ValueError("零填充长度 %r 不能小于输入样点数 %r" % ((my, mx), (ny, nx)))
    if my % 2 or mx % 2:
        raise ValueError("零填充长度必须为偶数，收到 %r" % ((my, mx),))

    lam = float(config.wavelength)
    z = float(config.distance)
    lamz = lam * z
    dx, dy = grid_in.dx, grid_in.dy

    # 零填充偏移：必须让输入的"0 号样点"落在填充后数组的 M//2 号样点上。
    # 偶数输入时 (M−n)//2 即可；**奇数输入时必须向上取整**，否则会整体错半个样点，
    # 导致 FFT 与完整核求和出现 O(0.1) 的差异（本阶段实测过该错误）。
    oy = (my - ny + 1) // 2
    ox = (mx - nx + 1) // 2
    if oy < 0 or ox < 0 or oy + ny > my or ox + nx > mx:
        raise ValueError("零填充长度不足以容纳输入网格")
    big = np.zeros((my, mx), dtype=np.complex128)
    big[oy:oy + ny, ox:ox + nx] = f
    # 记录实际使用的坐标（中心对齐后，填充数组的 0 号样点就是 input 的 0 号样点）
    xp = _centered_coords(mx, dx)
    yp = _centered_coords(my, dy)
    big *= np.exp(1j * np.pi * (xp[None, :] ** 2 + yp[:, None] ** 2) / lamz)

    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(big)))
    del big

    x = lamz * np.fft.fftshift(np.fft.fftfreq(mx, d=dx))
    y = lamz * np.fft.fftshift(np.fft.fftfreq(my, d=dy))
    grid_out = Grid2D(x=Axis1D(mx, float(x[1] - x[0]), "x"),
                      y=Axis1D(my, float(y[1] - y[0]), "y"), label="output@z")

    # 输出平面二次相位（以返回的物理坐标为准，与直接积分同一约定）
    q_out = np.pi * (x[None, :] ** 2 + y[:, None] ** 2) / lamz
    if config.omit_output_quadratic_phase:
        # 第二种场表示：**只**返回 C·F，不再乘任何输出二次相位。
        # （旧版这里乘了 exp(−iq)，使结果与完整场相差 exp(−2iq)，是明确错误；
        #  修正后满足 reduced·exp(+iq) == full，见 tests 与 phase_representation_check.npz。）
        representation = "reduced 场 u_red = C·F（未乘输出二次相位）"
        relation = "u_full = u_red · exp(+i·q_out)，q_out = π(x²+y²)/(λz)"
    else:
        F *= np.exp(1j * q_out)
        representation = "完整物理复场 u_full = C·F·exp(+i·q_out)"
        relation = "u_red = u_full · exp(−i·q_out)"

    scale = (dx * dy) / (1j * lamz)
    if config.include_global_phase:
        scale = scale * np.exp(1j * 2.0 * np.pi / lam * z)
    F *= scale

    p_in = float(np.sum(np.abs(f) ** 2) * grid_in.cell_area)
    p_out = float(np.sum(np.abs(F) ** 2) * grid_out.cell_area)
    hx_in = (nx // 2) * dx
    hy_in = (ny // 2) * dy
    hx_out = (mx // 2) * grid_out.dx
    hy_out = (my // 2) * grid_out.dy
    rho_max = float(np.hypot(hx_in + hx_out, hy_in + hy_out))

    meta: Dict[str, object] = {
        "formulation": "单次 FFT（中心化频率 + 面积权重 + 输出二次相位）",
        "representation": representation,
        "representation_relation": relation,
        "output_quadratic_phase_present": (not config.omit_output_quadratic_phase),
        "intensity_note": ("输出二次相位是纯相位因子，两种表示在**本平面**的强度完全相同；"
                           "它只影响复场相位，进而影响再次传播。"),
        "config": config.to_metadata(),
        "input_grid": grid_in.describe(),
        "output_grid": grid_out.describe(),
        "output_coordinate_rule": "x = λz·fftshift(fftfreq(Mx, Δx′))，y = λz·fftshift(fftfreq(My, Δy′))",
        "area_weight_m2": float(dx * dy),
        "scale_factor": "Δx′Δy′·exp(ikz)/(iλz)",
        "padding": {"input_shape_yx": [int(ny), int(nx)], "padded_shape_yx": [my, mx],
                    "pad_offset_yx": [int(oy), int(ox)]},
        "power_input_sum_abs2_cellarea": p_in,
        "power_output_native_sum_abs2_cellarea": p_out,
        "power_relative_error": (p_out - p_in) / p_in if p_in > 0 else float("nan"),
        "third_order_phase_error_rad": third_order_phase_error_rad(rho_max, z, lam),
        "leading_fourth_order_phase_error_rad": float(
            abs(2.0 * np.pi / lam * path_difference_leading_m(rho_max, z))),
        "paraxial_tolerance_rad": 0.1,
        "memory_estimate_mb": {"one_complex128_array": memory_estimate_mb((my, mx), 16)},
        "wall_time_s": float(time.perf_counter() - t0),
    }
    return PropagationResult(field=F, grid=grid_out, meta=meta)


# --------------------------------------------------------------------------------------
# 路径 3/4：谱传播（频域 Fresnel 近似 / 精确角谱）
# --------------------------------------------------------------------------------------
def _spectral_propagate(field_in: np.ndarray, grid_in: Grid2D, wavelength: float,
                        distance: float, padded_size: object, exact: bool,
                        include_global_phase: bool) -> PropagationResult:
    t0 = time.perf_counter()
    f = _check_shapes(field_in, grid_in)
    lam = float(wavelength)
    z = float(distance)
    k = 2.0 * np.pi / lam
    ny, nx = f.shape
    if padded_size is None:
        my, mx = ny, nx
    else:
        my, mx = FresnelConfig(lam, z, padded_size).padded_shape
    if my < ny or mx < nx:
        raise ValueError("零填充长度不能小于输入样点数")
    if my % 2 or mx % 2:
        raise ValueError("零填充长度必须为偶数")

    big = np.zeros((my, mx), dtype=np.complex128)
    oy = (my - ny) // 2
    ox = (mx - nx) // 2
    big[oy:oy + ny, ox:ox + nx] = f

    fx = np.fft.fftfreq(mx, d=grid_in.dx)
    fy = np.fft.fftfreq(my, d=grid_in.dy)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    fr2 = FX ** 2 + FY ** 2
    if exact:
        # 精确角谱：s = sqrt(1−λ²f²)；对 λ²f² > 1（倏逝分量）取复平方根，
        # 支路选择使 z>0 时指数衰减（Im(s) > 0 ⇒ exp(ikz·s) 的模为 exp(−kz·Im s)）。
        s = np.sqrt((1.0 - (lam ** 2) * fr2).astype(np.complex128))
        H_true = np.exp(1j * k * z * s)
        if include_global_phase:
            H = H_true
            tf = "exp(ikz·√(1−λ²(fx²+fy²)))（含倏逝波衰减；含全局相位 exp(ikz)）"
        else:
            # 只去掉 exp(ikz)：H_false = H_true · exp(−ikz)
            H = H_true * np.exp(-1j * k * z)
            tf = ("exp(ikz·√(1−λ²(fx²+fy²)))·exp(−ikz)"
                  "（已去掉全局相位 exp(ikz)，含倏逝波衰减）")
        name = ("精确角谱 H = exp(ikz·√(1−λ²f²))" if include_global_phase
                else "精确角谱 H = exp(ikz·√(1−λ²f²))·exp(−ikz)（无全局相位）")
    else:
        base = np.exp(-1j * np.pi * lam * z * fr2)
        if include_global_phase:
            H = base * np.exp(1j * k * z)
            tf = "exp(ikz)·exp(−iπλz(fx²+fy²))"
            name = "频域 Fresnel 近似 H = exp(ikz)·exp(−iπλz f²)"
        else:
            H = base
            tf = "exp(−iπλz(fx²+fy²))（已去掉全局相位 exp(ikz)）"
            name = "频域 Fresnel 近似 H = exp(−iπλz f²)（无全局相位）"

    out = np.fft.ifft2(np.fft.fft2(big) * H)
    field_out = out[oy:oy + ny, ox:ox + nx].copy()
    p_in = float(np.sum(np.abs(f) ** 2) * grid_in.cell_area)
    p_out = float(np.sum(np.abs(field_out) ** 2) * grid_in.cell_area)
    n_evanescent = int(np.count_nonzero((lam ** 2) * fr2 > 1.0)) if exact else 0
    meta: Dict[str, object] = {
        "formulation": name,
        "transfer_function": tf,
        "include_global_phase": bool(include_global_phase),
        "global_phase_factor": ("exp(ikz)" if include_global_phase else "1（已去掉 exp(ikz)）"),
        "output_grid": grid_in.describe(),
        "output_sampling_note": "输出与输入同一网格（Δx 不变）",
        "padded_shape_yx": [my, mx],
        "periodic_wrap_note": ("谱方法隐含周期边界：零填充只推迟回绕，不消除回绕；"
                               "输出从填充数组中按中心裁回原样点范围。"),
        "evanescent_bins": n_evanescent,
        "evanescent_note": ("λ²f²>1 的频点按复平方根指数衰减（z>0），"
                            "这部分功率会减少，因此**不能**宣称全输出严格无损；"
                            "Parseval 主检查只针对 Fresnel 频域形式单独执行。"),
        "power_input_sum_abs2_cellarea": p_in,
        "power_output_sum_abs2_cellarea": p_out,
        "power_relative_error": (p_out - p_in) / p_in if p_in > 0 else float("nan"),
        "wall_time_s": float(time.perf_counter() - t0),
    }
    return PropagationResult(field=field_out, grid=grid_in, meta=meta)


def fresnel_transfer_fresnel(field_in: np.ndarray, grid_in: Grid2D, wavelength: float,
                             distance: float, padded_size: object = None,
                             include_global_phase: bool = True) -> PropagationResult:
    """频域 **Fresnel 近似**传递函数传播（输出与输入同网格）。

    需要精确解时用 :func:`fresnel_angular_spectrum`。两者**不得**混称。
    """
    return _spectral_propagate(field_in, grid_in, wavelength, distance, padded_size,
                               exact=False, include_global_phase=include_global_phase)


def fresnel_angular_spectrum(field_in: np.ndarray, grid_in: Grid2D, wavelength: float,
                             distance: float, padded_size: object = None,
                             include_global_phase: bool = True) -> PropagationResult:
    """**精确角谱**传播 ``H = exp(ikz√(1−λ²f²))``（含倏逝波指数衰减）。"""
    return _spectral_propagate(field_in, grid_in, wavelength, distance, padded_size,
                               exact=True, include_global_phase=include_global_phase)
