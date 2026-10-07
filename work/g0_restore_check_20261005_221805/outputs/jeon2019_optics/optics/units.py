# -*- coding: utf-8 -*-
"""单位换算与配置解析。

约定：整个 ``optics`` 包内部长度单位 **一律为米（m）**；
nm / μm / mm 只允许出现在配置解析层（本模块）与报告文字中。
任何 optics 模块内部若出现 1e-9、1e-6、1e-3 之类的量纲因子，都属于错误。
"""

from __future__ import annotations

import numpy as np

# ---- 单位换算系数（乘上数值即得米）----
NM_TO_M = 1.0e-9
UM_TO_M = 1.0e-6
MM_TO_M = 1.0e-3

# ---- 反向换算（乘上米制数值即得目标单位）----
M_TO_NM = 1.0e9
M_TO_UM = 1.0e6
M_TO_MM = 1.0e3


def nm(value: float) -> float:
    """纳米 → 米。"""
    return float(value) * NM_TO_M


def um(value: float) -> float:
    """微米 → 米。"""
    return float(value) * UM_TO_M


def mm(value: float) -> float:
    """毫米 → 米。"""
    return float(value) * MM_TO_M


def to_nm(value_m):
    """米 → 纳米。支持标量与 ndarray。"""
    return np.asarray(value_m) * M_TO_NM if _is_array(value_m) else float(value_m) * M_TO_NM


def to_um(value_m):
    """米 → 微米。支持标量与 ndarray。"""
    return np.asarray(value_m) * M_TO_UM if _is_array(value_m) else float(value_m) * M_TO_UM


def to_mm(value_m):
    """米 → 毫米。支持标量与 ndarray。"""
    return np.asarray(value_m) * M_TO_MM if _is_array(value_m) else float(value_m) * M_TO_MM


def _is_array(value) -> bool:
    """判断是否为 ndarray / 序列（用于让换算函数同时支持标量与数组）。"""
    return isinstance(value, np.ndarray) or (
        hasattr(value, "__len__") and not isinstance(value, (str, bytes)))
