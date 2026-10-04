# -*- coding: utf-8 -*-
"""坐标与采样约定（阶段 01 修正版）。

本模块只负责"网格怎么排、坐标怎么算"，不做任何光学计算。

坐标与数组约定
--------------
* 内部长度单位一律为 **米（m）**。
* 数学坐标：x 向右、y 向上、z 沿传播方向（+z 为 DOE → 传感器方向）。
* 二维数组 ``a`` 的形状为 ``(ny, nx)``：``a[j, i]`` 对应
  ``x = x_axis[i]``、``y = y_axis[j]``。
* 一维轴为**以 0 为中心的对称采样**：``coords = (arange(n) − n//2) * d``，
  因此**索引增加时坐标增加**。偶数 n 时 0 落在 ``n//2`` 号样点上。
  配套的显示约定是 matplotlib 的 ``origin="lower"``：屏幕向上 = +y、向右 = +x。
  **本项目的行指标随 y 递增。** 阶段 01 初审版的文档写成了"行增加 y 减小"，
  与实际实现相反，已在修正版统一（见 ``docs/revision_notes.md``）。
* ``x`` 与 ``y`` 允许不同的样点数与间距（非方形网格），所有函数都按
  ``grid.shape = (ny, nx)`` 与 ``grid.dx != grid.dy`` 处理。

图像 extent 的两种约定
----------------------
* :attr:`Grid2D.extent_center_um`：``[x[0], x[-1], y[0], y[-1]]``，**样点中心**范围。
* :attr:`Grid2D.extent_pixel_um`：向外各扩半个像元，**像素边界**范围。
  画图（``imshow``）与"每个样点代表一个像元"的积分式用法都应使用它，
  以避免半像元偏差。``extent_um`` 作为兼容名指向像素边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Axis1D:
    """一维均匀采样轴。

    Attributes
    ----------
    n : int
        样点数。
    d : float
        样点间距（m，恒为正）。
    label : str
        物理轴名（"x" / "y" / "x'" / "y'"），仅用于报告与图轴标签。
    """

    n: int
    d: float
    label: str = "x"

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("Axis1D.n 必须 >= 1，收到 %r" % (self.n,))
        if not np.isfinite(self.d) or self.d <= 0:
            raise ValueError("Axis1D.d 必须是有限正数，收到 %r" % (self.d,))

    @property
    def coords(self) -> np.ndarray:
        """坐标数组（m），长度 n，**随索引递增**，0 位于 ``n // 2`` 号样点。"""
        return (np.arange(self.n, dtype=np.float64) - (self.n // 2)) * self.d

    @property
    def half_width(self) -> float:
        """样点覆盖的半宽（m）＝ ``(n // 2) * d``（不额外乘 0.5）。"""
        return (self.n // 2) * self.d

    @property
    def span(self) -> float:
        """样点中心覆盖的全宽（m）＝ ``n * d``（首末样点之间为 ``(n−1)·d``）。"""
        return self.n * self.d

    @property
    def indices(self) -> np.ndarray:
        """整数样点序号 ``k - n // 2``，与 ``coords / d`` 相同。"""
        return np.arange(self.n, dtype=np.int64) - (self.n // 2)

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志
        return "Axis1D(%s: n=%d, d=%.6g m, half_width=%.6g m)" % (
            self.label, self.n, self.d, self.half_width)


@dataclass(frozen=True)
class Grid2D:
    """二维网格 = 两个一维轴的组合（允许非方形、x/y 间距不同）。"""

    x: Axis1D
    y: Axis1D
    label: str = ""

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.y.n, self.x.n)

    @property
    def dx(self) -> float:
        return self.x.d

    @property
    def dy(self) -> float:
        return self.y.d

    @property
    def cell_area(self) -> float:
        """单个样点代表的面积（m²），即求积权重 Δx·Δy。"""
        return self.x.d * self.y.d

    @property
    def extent_center_um(self) -> Tuple[float, float, float, float]:
        """**样点中心**范围 ``[x0, x1, y0, y1]``（μm）。"""
        xc, yc = self.x.coords, self.y.coords
        return (float(xc[0]) * 1e6, float(xc[-1]) * 1e6,
                float(yc[0]) * 1e6, float(yc[-1]) * 1e6)

    @property
    def extent_pixel_um(self) -> Tuple[float, float, float, float]:
        """**像素边界**范围（μm）：中心坐标向外各扩半个像元。"""
        x0, x1, y0, y1 = self.extent_center_um
        return (x0 - 0.5 * self.dx * 1e6, x1 + 0.5 * self.dx * 1e6,
                y0 - 0.5 * self.dy * 1e6, y1 + 0.5 * self.dy * 1e6)

    @property
    def extent_um(self) -> Tuple[float, float, float, float]:
        """兼容名：返回**像素边界**范围（画图与像元积分都应用它）。"""
        return self.extent_pixel_um

    @property
    def extent_m(self) -> Tuple[float, float, float, float]:
        """像素边界范围（m）。"""
        x0, x1, y0, y1 = self.extent_pixel_um
        return (x0 * 1e-6, x1 * 1e-6, y0 * 1e-6, y1 * 1e-6)

    def meshgrid(self, indexing: str = "xy") -> Tuple[np.ndarray, np.ndarray]:
        """返回 ``(X, Y)`` 物理坐标网格（m）。

        ``indexing="xy"`` 时形状为 ``(ny, nx)``，与 ``Grid2D.shape`` 一致：
        ``X[j, i] = x_i``、``Y[j, i] = y_j``。
        """
        return np.meshgrid(self.x.coords, self.y.coords, indexing=indexing)

    def radius(self) -> np.ndarray:
        """半径网格 ``r = sqrt(x² + y²)``（m），形状 ``(ny, nx)``。"""
        X, Y = self.meshgrid()
        return np.hypot(X, Y)

    def describe(self) -> dict:
        """给 metadata / JSON 用的网格描述（长度均为米）。"""
        return {
            "label": self.label,
            "shape_yx": [int(self.y.n), int(self.x.n)],
            "dx_m": float(self.x.d),
            "dy_m": float(self.y.d),
            "x_min_m": float(self.x.coords[0]),
            "x_max_m": float(self.x.coords[-1]),
            "y_min_m": float(self.y.coords[0]),
            "y_max_m": float(self.y.coords[-1]),
            "cell_area_m2": float(self.cell_area),
            "array_index_convention": ("a[j, i] -> x=x_axis[i], y=y_axis[j]; "
                                       "索引增加时 x、y 均增加；显示用 origin='lower'"),
            "extent_convention": ("extent_pixel_um = 像素边界（外扩半像元）；"
                                  "extent_center_um = 样点中心范围"),
        }


def make_axis(n: int, d: float, label: str = "x") -> Axis1D:
    """构造一维轴（``d`` 单位米）。"""
    return Axis1D(n=int(n), d=float(d), label=label)


def axis_from_half_width(half_width: float, n: int, label: str = "x") -> Axis1D:
    """按"给定半宽 + 样点数"构造轴：``d = half_width / (n // 2)``。

    半宽按 ``(n//2)·d`` 定义（**不**额外乘 0.5）。
    """
    n = int(n)
    if n < 2:
        raise ValueError("n 必须 >= 2")
    return Axis1D(n=n, d=float(half_width) / (n // 2), label=label)


def make_grid(n: int, d: float, label: str = "") -> Grid2D:
    """方形网格：x、y 用同样的 n 与 d。"""
    return Grid2D(x=make_axis(n, d, "x"), y=make_axis(n, d, "y"), label=label)


def make_grid_xy(nx: int, ny: int, dx: float, dy: float, label: str = "") -> Grid2D:
    """任意网格：可非方形，x/y 间距可不同。"""
    return Grid2D(x=make_axis(nx, dx, "x"), y=make_axis(ny, dy, "y"), label=label)


def grid_from_half_width(half_width: float, n: int, label: str = "") -> Grid2D:
    """方形网格：x、y 半宽均为 ``half_width``（m），样点数 ``n``。"""
    return Grid2D(x=axis_from_half_width(half_width, n, "x"),
                  y=axis_from_half_width(half_width, n, "y"), label=label)


def circular_aperture(grid: Grid2D, diameter: float) -> np.ndarray:
    """圆孔掩膜（1 在孔内、0 在孔外），形状 ``(ny, nx)``。硬边，无过渡带。"""
    r = grid.radius()
    return (r <= 0.5 * float(diameter)).astype(np.float64)


def ideal_thin_lens_phase(grid: Grid2D, wavelength: float, focal_length: float) -> np.ndarray:
    """理想薄透镜的二次相位 ``exp(−iπ r²/(λ f))``。

    这是**传播程序的验证用**理想元件，不是论文的包裹浮雕 DOE；
    阶段 01 不使用任何论文式(7)–(12) 的高度设计。

    符号约定：Fourier 取负指数、时间因子 e^{−iωt}，正透镜相位为
    ``φ = −π r²/(λ f)``。它与 :func:`optics.propagation.fresnel_fft` 中的输入二次相位
    ``exp(+iπr′²/(λz))`` 在 ``z = f`` 处相消，因此焦平面就是孔径的 Fourier 强度
    （圆孔给出 Airy 图样）。这一段相消是本阶段 Airy 校验的物理依据，
    **不需要** ``D²/(4λf) ≪ 1``。
    """
    r2 = grid.radius() ** 2
    return np.exp(-1j * np.pi * r2 / (float(wavelength) * float(focal_length)))


def plane_wave(grid: Grid2D, amplitude: float = 1.0, phase0: float = 0.0) -> np.ndarray:
    """均匀平面波复振幅 ``A e^{iφ0}``，形状 ``(ny, nx)``。"""
    return np.full(grid.shape, float(amplitude) * np.exp(1j * float(phase0)),
                   dtype=np.complex128)


def memory_estimate_mb(shape: Tuple[int, ...], itemsize: int = 16, factor: float = 1.0) -> float:
    """估算数组占用内存（MB）。complex128 的 itemsize = 16 字节。"""
    n = 1
    for s in shape:
        n *= int(s)
    return n * int(itemsize) * float(factor) / (1024.0 ** 2)
