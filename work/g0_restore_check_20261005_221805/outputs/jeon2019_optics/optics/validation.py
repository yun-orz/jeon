"""有限物理窗口的衍射验证，保留绝对复场与输入功率标定。"""
from typing import Any, Dict
import time
import numpy as np
from .coordinates import Axis1D, Grid2D
from .metrics import airy_dark_ring_radii, airy_encircled_energy_analytic


def finite_window_energy(field_in: np.ndarray, grid_in: Grid2D,
                         wavelength: float, distance: float, diameter: float,
                         radius: float, window_fraction: float,
                         samples_per_dark_ring: float = 40.0,
                         label: str = "energy", logger=None) -> Dict[str, Any]:
    """在覆盖评价圆盘的小窗口计算式(4)，分母直接取输入功率。

    只省略输入振幅严格为零的外围区域，不重采样或改变相位。
    这是完整位移核的两次矩阵乘法；有限输出窗口无需包含全平面，
    因为绝对包围能量的分母不是输出窗口功率。
    """
    start = time.perf_counter()
    lam, z, D, R = map(float, (wavelength, distance, diameter, radius))
    u = np.asarray(field_in, dtype=np.complex128)
    if u.shape != grid_in.shape or not np.isfinite(u).all():
        raise ValueError("输入复场形状不符或含非有限值")
    if min(lam, z, D, R, samples_per_dark_ring) <= 0:
        raise ValueError("波长、距离、直径、评价半径与采样数必须为正")
    if not 0 < window_fraction < 1:
        raise ValueError("评价圆盘占窗口半宽的上限必须介于0和1之间")
    pin = float(np.sum(np.abs(u) ** 2) * grid_in.cell_area)
    if pin <= 0:
        raise ValueError("能量检查要求非零输入功率")
    r1 = float(airy_dark_ring_radii(lam, z, D, 1)[0])
    spacing = r1 / float(samples_per_dark_ring)
    half = R / float(window_fraction) * 1.025
    count = 2 * int(np.ceil(half / spacing)) + 1
    grid = Grid2D(Axis1D(count, spacing, "x"), Axis1D(count, spacing, "y"),
                  label="finite energy evaluation window")
    x, y = grid.x.coords, grid.y.coords

    # 精确裁去全零输入行列；原始坐标仍来自输入网格，不重新居中。
    rows = np.flatnonzero(np.any(u != 0, axis=1))
    cols = np.flatnonzero(np.any(u != 0, axis=0))
    ys, xs = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
    source = u[ys, xs]
    xp, yp = grid_in.x.coords[xs], grid_in.y.coords[ys]
    coefficient = np.pi / (lam * z)
    kx = np.exp(1j * coefficient * (x[:, None] - xp[None, :]) ** 2)
    ky = np.exp(1j * coefficient * (y[:, None] - yp[None, :]) ** 2)
    prefactor = np.exp(2j * np.pi / lam * z) * grid_in.cell_area / (1j * lam * z)
    field = prefactor * (ky @ source @ kx.T)
    intensity = np.abs(field) ** 2
    rr = np.hypot(x[None, :], y[:, None])
    disk = rr <= R
    pdisk = float(np.sum(intensity[disk]) * grid.cell_area)
    psquare = float(np.sum(intensity) * grid.cell_area)
    analytic = float(airy_encircled_energy_analytic(np.array([R]), lam, z, D)[0])
    order = np.argsort(rr.ravel())
    radial = rr.ravel()[order]
    cumulative = np.cumsum(intensity.ravel()[order]) * grid.cell_area
    # 相同半径取最后一个累计值；压缩重复坐标，不改变任何积分。
    end = np.r_[radial[1:] != radial[:-1], True]
    record = {
        "label": label, "method": "direct_finite_window",
        "formulation": "论文式(4)完整位移核，严格全零输入支撑裁剪，两次矩阵乘法",
        "R_m": R, "R_um": R * 1e6, "R_over_r1": R / r1,
        "fft_size": None, "native_output_points": count, "output_points": count,
        "samples_per_dark_ring": float(samples_per_dark_ring),
        "output_spacing_m": spacing, "output_spacing_um": spacing * 1e6,
        "window_half_m": float(x[-1]), "window_half_um": float(x[-1] * 1e6),
        "radial_analytic_comparison_max_radius_m": float(min(x[-1], y[-1])),
        "R_over_window_half": R / float(x[-1]),
        "window_covers_disk": bool(R < min(x[-1], y[-1])),
        "window_ok": bool(R / min(x[-1], y[-1]) <= window_fraction),
        "disk_does_not_select_whole_window": bool(0 < disk.sum() < disk.size),
        "input_spacing_m": grid_in.dx, "input_spacing_y_m": grid_in.dy,
        "input_shape_yx": list(u.shape), "nonzero_support_shape_yx": list(source.shape),
        "aperture_area_discrete_m2": float(np.count_nonzero(u) * grid_in.cell_area),
        "aperture_area_ideal_m2": float(np.pi * (D / 2) ** 2),
        "P_in": pin, "P_disk": pdisk, "P_square_window": psquare,
        "P_total_native_fft": None, "native_fft_power_rel_err": None,
        "E_numeric_disk_over_Pin": pdisk / pin, "E_analytic": analytic,
        "absolute_difference": abs(pdisk / pin - analytic),
        "conditional_fraction_disk_over_square": pdisk / psquare,
        "definition": "E_numeric=P_disk(R)/P_in；未用有限输出窗口归一化",
        "v_pi_D_R_over_lam_f": float(np.pi * D * R / (lam * z)),
        "wall_time_s": time.perf_counter() - start,
        "field": field, "intensity": intensity, "grid": grid,
        "radial_r_m": radial[end], "radial_P_cum": cumulative[end],
        "radial_available": True,
    }
    if logger is not None:
        logger.info("【能量·%s】完整位移核有限窗口 %d×%d；非零输入 %s；"
                    "Δout=%.4g μm；Pdisk/Pin=%.9f，解析=%.9f，差=%.3e（%.2f s）",
                    label, count, count, source.shape, spacing * 1e6,
                    pdisk / pin, analytic, record["absolute_difference"], record["wall_time_s"])
    return record
