# -*- coding: utf-8 -*-
"""材料色散模型（阶段 02A 新增）。

本模块提供熔融石英（Fused Silica）等光学材料的折射率计算。
默认采用 Malitson (1965) 三项 Sellmeier 公式，对应室温（20°C）常压环境。

参考文献
--------
I. H. Malitson, "Interspecimen Comparison of the Refractive Index of Fused Silica,"
J. Opt. Soc. Am. 55, 1205-1209 (1965).
DOI: 10.1364/JOSA.55.001205
URL: https://opg.optica.org/josa/abstract.cfm?URI=josa-55-10-1205

单位约定
--------
* 外部接口输入波长一律为 **米（m）**。
* Sellmeier 公式内部转换成 **微米（μm）** 计算。
* 有效波长范围：0.21 μm 至 3.71 μm（即 2.1e-7 m 至 3.71e-6 m）。
* 超出范围、非正数、NaN 或 Inf 均抛出 ValueError。
"""

from __future__ import annotations

from typing import Any, Dict, Union
import numpy as np

# Malitson (1965) 熔融石英 Sellmeier 系数 (λ 以 μm 计)
# n^2 - 1 = B1*L^2/(L^2 - C1) + B2*L^2/(L^2 - C2) + B3*L^2/(L^2 - C3)
FUSED_SILICA_SELLMEIER_COEFFS = {
    "B1": 0.6961663,
    "B2": 0.4079426,
    "B3": 0.8974794,
    "C1": 0.0684043 ** 2,  # μm^2
    "C2": 0.1162414 ** 2,  # μm^2
    "C3": 9.896161 ** 2,   # μm^2
}

# 有效波长范围（米）
FUSED_SILICA_WAVELENGTH_MIN_M = 0.21e-6
FUSED_SILICA_WAVELENGTH_MAX_M = 3.71e-6

# 空气折射率常数（实施假设：标准近轴近似下取 1.0）
REFRACTIVE_INDEX_AIR = 1.0


def refractive_index_fused_silica(
    wavelength_m: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """计算熔融石英在指定波长下的折射率（Malitson 1965 Sellmeier 模型）。

    Parameters
    ----------
    wavelength_m : float or np.ndarray
        真空中入射光波长（单位：米）。

    Returns
    -------
    float or np.ndarray
        无量纲折射率 n(λ)。

    Raises
    ------
    ValueError
        当波长 <= 0、包含 NaN/Inf 或超出 [0.21, 3.71] μm 有效范围时。
    """
    arr = np.asarray(wavelength_m, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("波长必须是有限数值，包含 NaN 或 Inf")
    if np.any(arr <= 0.0):
        raise ValueError("波长必须为严格正数（米）")

    # 范围检查（严禁静默把 nm 当作 m，如 550 将大于 3.71e-6 报错）
    if np.any(arr < FUSED_SILICA_WAVELENGTH_MIN_M) or np.any(arr > FUSED_SILICA_WAVELENGTH_MAX_M):
        raise ValueError(
            f"波长超出 Malitson (1965) 熔融石英有效范围 "
            f"[{FUSED_SILICA_WAVELENGTH_MIN_M*1e6:.2f}, {FUSED_SILICA_WAVELENGTH_MAX_M*1e6:.2f}] μm"
        )

    # 转换为 μm
    L_um = arr * 1e6
    L2 = L_um ** 2

    c = FUSED_SILICA_SELLMEIER_COEFFS
    term1 = c["B1"] * L2 / (L2 - c["C1"])
    term2 = c["B2"] * L2 / (L2 - c["C2"])
    term3 = c["B3"] * L2 / (L2 - c["C3"])

    n2 = 1.0 + term1 + term2 + term3
    if np.any(n2 <= 0.0):
        raise ValueError("Sellmeier 公式计算出非物理的 n^2 <= 0")

    n = np.sqrt(n2)
    if np.isscalar(wavelength_m):
        return float(n)
    return n


def delta_refractive_index(
    wavelength_m: Union[float, np.ndarray],
    n_air: float = REFRACTIVE_INDEX_AIR
) -> Union[float, np.ndarray]:
    """计算材料与空气的折射率差 Δn = n_material - n_air。

    Parameters
    ----------
    wavelength_m : float or np.ndarray
        波长（单位：米）。
    n_air : float
        环境空气折射率，默认 1.0。

    Returns
    -------
    float or np.ndarray
        折射率差 Δn。
    """
    return refractive_index_fused_silica(wavelength_m) - float(n_air)


def get_material_model_metadata() -> Dict[str, Any]:
    """返回材料模型元数据，便于记录到实验报告与清单中。"""
    return {
        "material_name": "fused_silica",
        "dispersion_model": "Sellmeier_3term",
        "reference": "Malitson (1965) J. Opt. Soc. Am. 55(10), 1205-1209",
        "doi": "10.1364/JOSA.55.001205",
        "assumed_temperature_celsius": 20.0,
        "valid_wavelength_range_um": [
            FUSED_SILICA_WAVELENGTH_MIN_M * 1e6,
            FUSED_SILICA_WAVELENGTH_MAX_M * 1e6,
        ],
        "coefficients_um2": dict(FUSED_SILICA_SELLMEIER_COEFFS),
        "n_air_assumed": REFRACTIVE_INDEX_AIR,
    }
