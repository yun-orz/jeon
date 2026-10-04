# -*- coding: utf-8 -*-
"""``optics`` 包 —— Jeon2019 光学复现（阶段 01：传播基线）。

本包内部长度单位**一律为米（m）**；nm / μm / mm 只在 :mod:`optics.units`
与配置解析层出现。

阶段 01 只包含：
* :mod:`optics.coordinates` —— 坐标与采样约定；
* :mod:`optics.propagation` —— 论文式(4) 的 Fresnel 传播（单次 FFT）与直接求积参考；
* :mod:`optics.metrics` —— 误差、暗环、包围能量与解析参考；
* :mod:`optics.units` —— 单位换算。

DOE 高度设计（式(7)–(12)）、成像（式(13)–(15)）与重建**不在本阶段范围内**，
因此本包中不存在对应模块。
"""

from . import coordinates, metrics, propagation, units  # noqa: F401

__all__ = ["coordinates", "propagation", "metrics", "units"]
__version__ = "0.1.0-stage01"
