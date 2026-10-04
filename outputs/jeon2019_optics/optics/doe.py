# -*- coding: utf-8 -*-
"""DOE 高度设计与固定器件波长相位响应（阶段 02A 新增）。

本模块实现：
1. 传统 Fresnel 透镜连续高度轮廓（正文式 (7)–(9)）；
2. Jeon et al. 2019 式 (9)–(12) 的 N=3 各向异性螺旋连续高度轮廓；
3. 固定高度器件在给定入射波长下的复振幅透过率场 u1（正文式 (2)–(3)）；
4. 固定器件的确定性内容哈希（指纹），确保多波长计算中器件绝对同一。

设计公式与约定
--------------
* 所有坐标与物理长度一律使用 **米（m）**。
* 径向光程差：使用数值稳定形式 delta = r^2 / (sqrt(r^2 + f^2) + f)。
* 极角范围：θ = mod(atan2(y, x), 2*pi)，逆时针为正。
* 角向周期：theta_period = mod(theta, 2*pi / N)。
* 设计波长匹配：
    lambda_design(theta) = lambda_min + (lambda_max - lambda_min) * N * theta_period / (2*pi)
* 设计折射率：实施假设 n_design = n_fused_silica(lambda_design)。
* 绕回阶数：wrap_order = floor(delta / lambda_design)。
* 相对高度差：
    delta_h = (wrap_order * lambda_design - delta) / (n_design - 1)
  孔径内满足：-lambda_design / (n_design - 1) <= delta_h <= 0。
* 孔径外：高度置 0，并配套掩膜 circular_mask。
* 原点定义：r=0 时 delta=0, wrap_order=0, delta_h=0。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np

from .coordinates import Grid2D, circular_aperture
from .materials import refractive_index_fused_silica, REFRACTIVE_INDEX_AIR


@dataclass(frozen=True)
class DOEHeightProfile:
    """DOE 高度轮廓结构体。

    Attributes
    ----------
    grid : Grid2D
        输入采样物理网格。
    delta_h : np.ndarray
        相对高度差 Δh(x, y)（单位：米），孔径外为 0。
    mask : np.ndarray
        有效圆形孔径掩膜（孔内 1.0，孔外 0.0）。
    lambda_design : Optional[np.ndarray]
        每个网格点的局部设计波长（单位：米），对传统透镜为单值标量广播数组。
    design_type : str
        设计类型名称："jeon2019_spiral" 或 "conventional_fresnel"。
    params : Dict[str, Any]
        设计参数字典。
    """
    grid: Grid2D
    delta_h: np.ndarray
    mask: np.ndarray
    lambda_design: Optional[np.ndarray]
    design_type: str
    params: Dict[str, Any]

    def compute_fingerprint(self) -> str:
        """计算确定性器件指纹（SHA256）。

        指纹基于小端序 float64/int64 二进制数据与标准化参数 JSON 计算，
        不受文件打包时间戳影响。
        """
        hasher = hashlib.sha256()

        # 1. 关键设计参数（按字典键排序的规范 JSON）
        clean_params = {
            "design_type": str(self.design_type),
            "dx_m": float(self.grid.dx),
            "dy_m": float(self.grid.dy),
            "shape_yx": [int(self.grid.y.n), int(self.grid.x.n)],
            "params": {k: float(v) if isinstance(v, (int, float)) else str(v)
                       for k, v in sorted(self.params.items())}
        }
        hasher.update(json.dumps(clean_params, sort_keys=True).encode("utf-8"))

        # 2. 坐标轴数据（确定性连续小端序 float64）
        x_bytes = np.ascontiguousarray(self.grid.x.coords, dtype="<f8").tobytes()
        y_bytes = np.ascontiguousarray(self.grid.y.coords, dtype="<f8").tobytes()
        hasher.update(x_bytes)
        hasher.update(y_bytes)

        # 3. 高度与掩膜数组（确定性连续小端序 float64）
        h_bytes = np.ascontiguousarray(self.delta_h, dtype="<f8").tobytes()
        m_bytes = np.ascontiguousarray(self.mask, dtype="<f8").tobytes()
        hasher.update(h_bytes)
        hasher.update(m_bytes)

        return hasher.hexdigest()


def optical_path_difference_delta(r: np.ndarray, focal_length: float) -> np.ndarray:
    """计算聚焦几何光程差 δ(r) = sqrt(r^2 + f^2) - f。

    采用数值稳定公式 r^2 / (sqrt(r^2 + f^2) + f)。
    """
    f = float(focal_length)
    r2 = r ** 2
    return r2 / (np.sqrt(r2 + f ** 2) + f)


def design_conventional_fresnel_height(
    grid: Grid2D,
    diameter: float,
    focal_length: float,
    wavelength_design: float = 550e-9,
) -> DOEHeightProfile:
    """设计传统单色 Fresnel 透镜连续高度轮廓（正文式 (7)–(9)）。

    Parameters
    ----------
    grid : Grid2D
        输入网格。
    diameter : float
        孔径直径 D（米）。
    focal_length : float
        设计焦距 f（米）。
    wavelength_design : float
        设计单色波长 λ0（米），默认 550 nm。

    Returns
    -------
    DOEHeightProfile
        传统 Fresnel 透镜高度轮廓对象。
    """
    D = float(diameter)
    f = float(focal_length)
    lam0 = float(wavelength_design)

    mask = circular_aperture(grid, D)
    r = grid.radius()

    # 几何光程差
    delta = optical_path_difference_delta(r, f)

    # 传统 Fresnel 透镜设计折射率
    n_design = float(refractive_index_fused_silica(lam0))
    dn = n_design - REFRACTIVE_INDEX_AIR

    # 绕回阶数与相对高度差
    wrap_order = np.floor(delta / lam0)
    delta_h = (wrap_order * lam0 - delta) / dn

    # 原点 r=0 精确置 0
    delta_h = np.where(r == 0.0, 0.0, delta_h)

    # 孔径外高度强制归零
    delta_h = delta_h * mask

    params = {
        "diameter_m": D,
        "focal_length_m": f,
        "wavelength_design_m": lam0,
        "refractive_index_design": n_design,
    }

    lam_design_arr = np.full(grid.shape, lam0, dtype=np.float64)

    return DOEHeightProfile(
        grid=grid,
        delta_h=delta_h,
        mask=mask,
        lambda_design=lam_design_arr,
        design_type="conventional_fresnel",
        params=params,
    )


def design_jeon2019_spiral_height(
    grid: Grid2D,
    diameter: float,
    focal_length: float,
    wings_N: int = 3,
    lambda_min: float = 420e-9,
    lambda_max: float = 660e-9,
) -> DOEHeightProfile:
    """设计 Jeon et al. 2019 各向异性螺旋连续 DOE 高度轮廓（正文式 (9)–(12)）。

    Parameters
    ----------
    grid : Grid2D
        输入平面物理网格。
    diameter : float
        孔径直径 D（米）。
    focal_length : float
        设计焦距 f（米）。
    wings_N : int
        周期数/翼数 N，默认为 3。
    lambda_min : float
        设计波长下限（米），默认 420 nm。
    lambda_max : float
        设计波长上限（米），默认 660 nm。

    Returns
    -------
    DOEHeightProfile
        Jeon2019 螺旋连续 DOE 高度轮廓对象。
    """
    D = float(diameter)
    f = float(focal_length)
    N = int(wings_N)
    l_min = float(lambda_min)
    l_max = float(lambda_max)

    if N <= 0:
        raise ValueError(f"翼数 N 必须为正整数，收到 {N}")
    if l_min <= 0 or l_max <= l_min:
        raise ValueError(f"设计波长范围非法：[{l_min}, {l_max}]")

    mask = circular_aperture(grid, D)
    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)

    # 极角 θ ∈ [0, 2π)
    theta = np.mod(np.arctan2(Y, X), 2.0 * np.pi)

    # 角向周期 θ_period ∈ [0, 2π / N)
    period_angle = 2.0 * np.pi / N
    theta_period = np.mod(theta, period_angle)

    # 各向异性局部设计波长 λ_design(θ)
    lambda_design = l_min + (l_max - l_min) * (theta_period / period_angle)

    # 原点特殊处理：r=0 处指定 θ=0, lambda_design = lambda_min
    lambda_design = np.where(r == 0.0, l_min, lambda_design)

    # 材料色散假设：n_design = n_fused_silica(lambda_design)
    n_design = refractive_index_fused_silica(lambda_design)
    dn_design = n_design - REFRACTIVE_INDEX_AIR

    # 几何光程差 δ(r)
    delta = optical_path_difference_delta(r, f)

    # 绕回阶数 wrap_order = floor(delta / lambda_design)
    wrap_order = np.floor(delta / lambda_design)

    # 相对高度差 Δh(r, θ) = (wrap_order * lambda_design - delta) / (n_design - 1)
    delta_h = (wrap_order * lambda_design - delta) / dn_design

    # 原点 r=0 精确置 0
    delta_h = np.where(r == 0.0, 0.0, delta_h)

    # 孔径外高度强制置零
    delta_h = delta_h * mask

    params = {
        "diameter_m": D,
        "focal_length_m": f,
        "wings_N": N,
        "lambda_min_m": l_min,
        "lambda_max_m": l_max,
        "material": "fused_silica_malitson1965",
    }

    return DOEHeightProfile(
        grid=grid,
        delta_h=delta_h,
        mask=mask,
        lambda_design=lambda_design,
        design_type="jeon2019_spiral",
        params=params,
    )


def compute_doe_transmission_field(
    profile: DOEHeightProfile,
    wavelength_in: float,
    amplitude: float = 1.0,
    h_offset: float = 0.0,
) -> np.ndarray:
    """计算固定高度 DOE 在指定波长入射光下的复振幅透过率场 u1（正文式 (2)–(3)）。

    严禁在此重新设计高度！输入必须是已固定生成的 profile。

    Parameters
    ----------
    profile : DOEHeightProfile
        预先计算好的固定高度器件。
    wavelength_in : float
        当前实际入射单色光波长（单位：米）。
    amplitude : float
        轴上平面波振幅，默认 1.0。
    h_offset : float
        全局物理高度偏置常数（米），默认 0.0。仅引入全局常数相位，不改变光斑强度。

    Returns
    -------
    np.ndarray
        形状与 profile.grid 相同的 complex128 透过率场 u1。
    """
    lam_in = float(wavelength_in)
    n_in = float(refractive_index_fused_silica(lam_in))
    dn = n_in - REFRACTIVE_INDEX_AIR

    # 物理高度 h(x', y') = delta_h + h_offset
    # 相位差 phi_h = 2*pi/lambda_in * (n_in - n_air) * (delta_h + h_offset)
    h_total = profile.delta_h + float(h_offset)
    phase_h = (2.0 * np.pi / lam_in) * dn * h_total

    # 孔径内出射场：u1 = mask * A * exp(i * phase_h)
    u1 = profile.mask * (float(amplitude) * np.exp(1j * phase_h))
    return u1
