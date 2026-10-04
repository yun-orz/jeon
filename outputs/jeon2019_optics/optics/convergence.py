# -*- coding: utf-8 -*-
"""原始强度采样比较；插值仅用于测量误差，不作为新PSF生成方法。"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def common_indices(source_axis, target_axis):
    """只允许提取重合的原生坐标点，不做近邻代替或插值。"""
    source = np.asarray(source_axis)
    target = np.asarray(target_axis)
    step = source[1] - source[0]
    indices = np.rint((target - source[0]) / step).astype(int)
    if np.any(indices < 0) or np.any(indices >= source.size):
        raise ValueError("目标坐标超过源网格")
    if not np.allclose(source[indices], target, rtol=0, atol=abs(step)*1e-9):
        raise ValueError("目标坐标不是源场的原生采样点")
    return indices


def compare_intensities(coarse, coarse_grid, fine, fine_grid):
    """双线性强度插值后的相对误差，同时记录共同原生点误差。"""
    if coarse.shape != coarse_grid.shape or fine.shape != fine_grid.shape:
        raise ValueError("强度与网格形状不一致")
    if not (np.all(np.isfinite(coarse)) and np.all(np.isfinite(fine))):
        raise ValueError("强度必须有限")
    if np.any(coarse < 0) or np.any(fine < 0) or fine.sum() <= 0:
        raise ValueError("强度必须非负且参考能量非零")
    if np.array_equal(coarse_grid.x.coords, fine_grid.x.coords) and \
            np.array_equal(coarse_grid.y.coords, fine_grid.y.coords):
        sampled = coarse
        common_error = None
        method = "同一物理网格，直接比较原始强度"
    else:
        X, Y = fine_grid.meshgrid()
        interp = RegularGridInterpolator((coarse_grid.y.coords, coarse_grid.x.coords),
                                         coarse, bounds_error=True)
        sampled = interp(np.column_stack((Y.ravel(), X.ravel()))).reshape(fine.shape)
        ix = common_indices(fine_grid.x.coords, coarse_grid.x.coords)
        iy = common_indices(fine_grid.y.coords, coarse_grid.y.coords)
        common = fine[np.ix_(iy, ix)]
        common_error = float(np.max(np.abs(common-coarse))/max(float(common.max()),1e-300))
        method = "粗强度双线性插值，仅为误差诊断；共同点另核对"
    diff = sampled - fine
    return {"intensity_l1": float(np.abs(diff).sum()/fine.sum()),
            "intensity_l2": float(np.linalg.norm(diff)/np.linalg.norm(fine)),
            "common_point_relative_max": common_error, "method": method}


def periodic_angle_difference_deg(a, b):
    """N=3方向按120度周期求最小角差；缺失方向不伪造数值。"""
    if a is None or b is None:
        return None
    return float(abs((a-b+60.0) % 120.0-60.0))


def compare_metrics(a, b, thresholds):
    """检查功率、两个环带方向、绝对能量半径与峰值阈值面积。"""
    values = {"eta_absolute": abs(a["eta_window"]-b["eta_window"])}
    checks = {"eta": values["eta_absolute"] <= thresholds["eta_absolute"]}
    for band in ("primary", "sensitivity"):
        aa, bb = a["rotation_bands"][band], b["rotation_bands"][band]
        d = periodic_angle_difference_deg(aa["alpha_wrapped_deg"], bb["alpha_wrapped_deg"])
        values[band+"_angle_difference_deg"] = d
        checks[band+"_angle"] = (d <= thresholds["angle_deg"]) if d is not None \
            else not aa["reliable"] and not bb["reliable"]
    for radius in ("R50", "R80"):
        aa, bb = a[radius]["radius_um"], b[radius]["radius_um"]
        d = abs(aa-bb) if aa is not None and bb is not None else None
        values[radius+"_difference_um"] = d
        checks[radius] = (d <= thresholds["radius_um"]) if d is not None else aa is None and bb is None
    for q in ("q0.5", "q0.1"):
        aa, bb = a["shape"][q]["area_m2"], b["shape"][q]["area_m2"]
        d = abs(aa-bb)/bb if bb>0 else (0.0 if aa==0 else None)
        values[q+"_area_relative"] = d
        checks[q+"_area"] = d is not None and d <= thresholds["area_relative"]
    return {"values": values, "checks": checks, "metrics_pass": all(checks.values())}
