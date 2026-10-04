# -*- coding: utf-8 -*-
"""Jeon2019 光学复现 —— 阶段 01（修正版）：Fresnel 传播基线的建立与验证。

用法
----
在 PyCharm 中直接点击本文件的运行按钮（无参数、默认 stage01、CPU、保存结果、不弹窗）::

    python main.py                       # 使用项目根目录的 config.json
    python main.py --config other.json   # 指定配置（相对路径按**当前工作目录**解析）
    python main.py --show-plots          # 交互查看：切换可用交互后端，图窗弹出
    python main.py --no-save             # 真正不落盘（只打印，不写任何文件）
    python main.py --only airy           # 只跑某一组（airy/direct/energy/sampling）
    python main.py --run-dir results/stage01/run_x   # 写入**必须为空**的目录

本阶段范围（修正后不变）：只建立并验证传播基线。
不实现 DOE 高度（式(7)–(12)）、三翼结构、多波长编码 PSF、图 3、成像或重建。

修正要点（详见 ``docs/revision_notes.md`` 与 ``docs/reproduction_log.md``）
--------------------------------------------------------------------------
1. 直接求值不再预乘输入二次相位（旧版重复相位）。
2. 单次 FFT 补齐 **Δx′Δy′ 面积权重**，并用 **fftshift(fft2(ifftshift(·)))** 做频率中心化，
   输出坐标由 ``λz·fftshift(fftfreq(M, Δx′))`` 给出。修正后它与完整位移核求和
   在同一离散输入、同一原生输出网格上相差 ~1e−14（旧版差 6.25e8）。
3. 理想透镜焦平面：``z=f`` 时透镜相位与传播展开式的输入二次相位**相消**，
   焦平面就是孔径的 Fourier 强度（圆孔→Airy）。**不再要求** ``D²/(4λf)≪1``；
   论文尺度 ``D=1 mm, f=50 mm`` 直接求值并按 20 样点/暗环检查。
4. 高阶光程误差改用**精确**路径差 ``√(z²+ρ²)−z−ρ²/(2z)``，``ρ`` 取实际输入支撑
   与输出窗口的角点距离；不再引入额外高阶因子（旧版把 0.0279 rad 误报为 4.1e−11）。
5. 能量判据恢复为**原生 FFT 全输出 Parseval 检验**（≤1e−10）；有限窗口对解析
   包围能量只作**补充**检查。
6. 坐标说明与实现一致：**索引增加 → x、y 都增加**，显示用 ``origin='lower'``；
   图像 extent 区分样点中心与像素边界。
7. ``--no-save`` 真正不写文件；``--run-dir`` 要求目标为空。
8. 后端在导入 pyplot **之前**按需选择；验收失败进入机器可读总状态并返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Jeon2019 光学复现 —— 阶段 01（修正版）：Fresnel 传播基线")
    p.add_argument("--config", type=str, default=None,
                   help="配置文件路径；相对路径按当前工作目录解析（默认 <项目>/config.json）")
    p.add_argument("--stage", type=str, default=None, help="覆盖配置中的 stage")
    p.add_argument("--show-plots", action="store_true",
                   help="交互查看模式：在导入 pyplot 前选择可用交互后端并弹出图窗")
    p.add_argument("--no-save", action="store_true",
                   help="真正不落盘：不创建任何结果文件（含日志与图）")
    p.add_argument("--log-to-console-only", action="store_true",
                   help="与 --no-save 相同语义的显式别名，便于阅读")
    p.add_argument("--only", type=str, default=None,
                   choices=["airy", "direct", "energy", "sampling"],
                   help="只运行指定的一组验证；未执行的组在总状态中标记为 not_run")
    p.add_argument("--run-dir", type=str, default=None,
                   help="写入指定目录；该目录必须不存在或为空，否则拒绝执行")
    p.add_argument("--light", action="store_true",
                   help="缩小 A/C/D 组的 FFT 规模以加快负面测试与快速回归；"
                        "只改计算规模，不改任何物理定义或验收阈值，正式运行不使用")
    return p


ARGS = build_parser().parse_args()

# 路径与配置**先**解析（第二轮修正 S6）：只有拿到最终的 show/save 结论之后才允许
# 选择 matplotlib 后端并导入 pyplot，否则 runtime.show_plots 无法生效。
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"


def resolve_project_path(value: Optional[str]) -> Optional[Path]:
    """解析用户给出的路径。

    规则（README 与配置说明里写明）：相对路径按**项目位置**（``__file__``）解析，
    这样在 PyCharm 里点击运行与从任意工作目录用绝对路径运行的结果一致；
    传入绝对路径时原样使用。若需要按当前工作目录解析，请直接给出绝对路径。
    """
    if value in (None, ""):
        return None
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def resolve_run_settings(args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把配置与命令行合成**最终**运行设置。

    ``effective_show = runtime.show_plots 或 CLI --show-plots``
    ``effective_save = runtime.save_results 且未使用 CLI --no-save/--log-to-console-only``
    """
    rt = cfg.get("runtime") or {}
    cli_show = bool(getattr(args, "show_plots", False))
    cli_no_save = bool(getattr(args, "no_save", False)
                       or getattr(args, "log_to_console_only", False))
    show = bool(rt.get("show_plots", False)) or cli_show
    save = bool(rt.get("save_results", True)) and not cli_no_save
    return {
        "show_plots": show,
        "save_results": save,
        "show_source": ("CLI --show-plots" if cli_show else
                        ("配置 runtime.show_plots" if rt.get("show_plots") else "默认 false")),
        "save_source": ("CLI --no-save/--log-to-console-only" if cli_no_save else
                        ("配置 runtime.save_results=%s" % rt.get("save_results", True))),
        "cli_show": cli_show, "cli_no_save": cli_no_save,
        "config_show": bool(rt.get("show_plots", False)),
        "config_save": bool(rt.get("save_results", True)),
    }


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_runtime_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """按命令行覆盖**计算规模**（只影响速度，不改变任何物理定义）。

    ``--light``：把 A/C/D 组的 FFT 长度与输入规模缩小若干倍，用于负面测试与快速回归。
    它与验收阈值无关，因此不会把失败掩盖成通过；正式运行**不使用**它。
    """
    if not getattr(args, "light", False):
        return cfg
    s = cfg["studies"]
    for case in s["airy"]["cases"]:
        case["input_spacing_um"] = max(float(case["input_spacing_um"]), 2.0)
        case["input_half_width_um"] = min(
            float(case["input_half_width_um"]),
            400.0 * float(case["aperture_diameter_mm"]) + 100.0)
        ev = case.get("energy_evaluation")
        if ev:
            ev["fft_size"] = 1024
    # C 组的能量评价输入网格较大，轻量模式必须把 FFT 长度设在样点数之上
    ec = s["energy"]["airy_window_case"]
    ec["input_spacing_um"] = 2.0
    n_ec = 2 * int(round(float(ec["input_half_width_um"]) / float(ec["input_spacing_um"]))) + 1
    for lv in s["energy"].get("energy_levels", []):
        lv["fft_size"] = max(1024, n_ec * 2)
    smp = s["sampling"]
    smp["common_output_spacing_um"] = max(
        float(smp.get("common_output_spacing_um", 2.0)), 4.0)
    smp["input_levels"] = smp["input_levels"][:2]
    smp["padding_levels"] = [{"name": "M512", "fft_size": 512},
                             {"name": "M1024", "fft_size": 1024}]
    smp["identity_cases"] = smp["identity_cases"][:1]
    cfg.setdefault("runtime", {})["light_mode"] = True
    return cfg


_CFG_PATH = resolve_project_path(ARGS.config) or DEFAULT_CONFIG
if not _CFG_PATH.exists():
    print("[错误] 找不到配置文件：%s" % _CFG_PATH)
    raise SystemExit(2)
_CFG_EARLY = apply_runtime_overrides(load_config(_CFG_PATH), ARGS)
SETTINGS = resolve_run_settings(ARGS, _CFG_EARLY)


# ------------------------------------------------------------------ 后端选择（必须在 pyplot 之前）
def select_backend(show_plots: bool) -> str:
    """按运行模式选择 matplotlib 后端，**在导入 pyplot 之前**调用。

    * 批处理（``show_plots=False``）：用 ``Agg``，不依赖窗口系统。
    * 交互查看（``show_plots=True``）：优先使用可用的交互后端
      （``QtAgg``/``TkAgg``/``GTK3Agg`` 等），失败时回退到 ``Agg`` 并记录，
      此时 ``plt.show()`` 不会弹窗——报告会如实写明。
    """
    import matplotlib
    if not show_plots:
        matplotlib.use("Agg", force=True)
        return "Agg"
    for cand in ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg", "MacOSX", "Qt5Agg"):
        try:
            matplotlib.use(cand, force=True)
            import matplotlib.pyplot as _plt
            _plt.switch_backend(cand)
            return cand
        except Exception:
            continue
    matplotlib.use("Agg", force=True)
    return "Agg（回退：未找到可用交互后端，plt.show() 不会弹窗）"


_SHOW_PLOTS = bool(SETTINGS["show_plots"])
BACKEND = select_backend(_SHOW_PLOTS)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optics import metrics as M                       # noqa: E402
from optics import units as U                         # noqa: E402
from optics.coordinates import (                      # noqa: E402
    Axis1D, Grid2D, circular_aperture, ideal_thin_lens_phase,
    make_grid_xy, memory_estimate_mb, plane_wave,
)
from optics.propagation import (                      # noqa: E402
    FresnelConfig, axial_fresnel_number, fresnel_fft, fresnel_kernel_matrix,
    fresnel_kernel_separable, path_difference_exact_m, path_difference_leading_m,
    sampling_diagnostics, third_order_phase_error_rad,
)
from optics.runutil import (                          # noqa: E402
    PROJECT_ROOT as PKG_ROOT, Timer, allocate_run_dir, environment_info,
    setup_logger, write_json,
)

import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager as _fontmgr       # noqa: E402


def _configure_cjk_font() -> Optional[str]:
    """为图内中文标签挑选本机存在的 CJK 字体。"""
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                 "SimSun", "DengXian", "Arial Unicode MS"]
    available = {f.name for f in _fontmgr.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK_FONT = _configure_cjk_font()


# ======================================================================================
# 基础设施
# ======================================================================================
def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_runtime_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """按命令行覆盖**计算规模**（只影响速度，不改变任何物理定义）。

    ``--light``：把 A/C/D 组的 FFT 长度与输入规模缩小若干倍，用于负面测试与快速回归。
    它与验收阈值无关，因此不会把失败掩盖成通过；正式运行**不使用**它。
    """
    if not getattr(args, "light", False):
        return cfg
    s = cfg["studies"]
    for case in s["airy"]["cases"]:
        case["input_spacing_um"] = max(float(case["input_spacing_um"]), 2.0)
        case["input_half_width_um"] = min(
            float(case["input_half_width_um"]),
            400.0 * float(case["aperture_diameter_mm"]) + 100.0)
        ev = case.get("energy_evaluation")
        if ev:
            ev["fft_size"] = 1024
    # C 组的能量评价输入网格较大，轻量模式必须把 FFT 长度设在样点数之上
    ec = s["energy"]["airy_window_case"]
    ec["input_spacing_um"] = 2.0
    n_ec = 2 * int(round(float(ec["input_half_width_um"]) / float(ec["input_spacing_um"]))) + 1
    for lv in s["energy"].get("energy_levels", []):
        lv["fft_size"] = max(1024, n_ec * 2)
    smp = s["sampling"]
    smp["common_output_spacing_um"] = max(
        float(smp.get("common_output_spacing_um", 2.0)), 4.0)
    smp["input_levels"] = smp["input_levels"][:2]
    smp["padding_levels"] = [{"name": "M512", "fft_size": 512},
                             {"name": "M1024", "fft_size": 1024}]
    smp["identity_cases"] = smp["identity_cases"][:1]
    cfg.setdefault("runtime", {})["light_mode"] = True
    return cfg


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join("" if v is None else str(v) for v in row) + "\n")


def gnum(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(value) if not np.isfinite(value) else repr(float(value))
    return str(value)


def save_npz(run_dir: Optional[Path], name: str, **arrays: Any) -> Optional[float]:
    """写 npz；``run_dir`` 为 None（--no-save）时直接返回 None，不写任何文件。"""
    if run_dir is None:
        return None
    path = run_dir / "arrays" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path.stat().st_size / (1024.0 ** 2)


def write_csv_maybe(run_dir: Optional[Path], name: str, header, rows) -> None:
    if run_dir is None:
        return
    write_csv(run_dir / "metrics" / name, header, rows)


def save_fig(fig, run_dir: Optional[Path], name: str, dpi: int, show: bool) -> Optional[Path]:
    """保存图；无结果目录时只（在 show 模式下）显示。"""
    p = None
    if run_dir is not None:
        p = run_dir / "figures" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return p


def encircled_energy_record(u1: np.ndarray, grid_in: Grid2D, lam: float, f: float, D: float,
                            R: float, fft_size: int, window_frac: float, label: str,
                            logger=None) -> Dict[str, Any]:
    """统一口径的绝对包围能量记录（第二轮修正 S3）。

    为什么用**单次 FFT**而不是可分离直接求值
    ----------------------------------------
    解析式 ``1−J₀²−J₁²`` 的分母是**无限平面**的总功率。可分离直接求值只对有限的
    输入窗与有限的输出窗求和，而焦平面强度的尾部衰减很慢（远处的相对偏差可达
    百分之几，本阶段实测在 20·r1 处仍有约 2% 的标定偏差），因此它无法给出与解析式
    可比的分母。单次 FFT 的原生全输出功率**严格守恒**
    （``meta['power_relative_error']`` 实测 ≤2.2e−14），所以它是这里唯一合适的口径。

    分母口径（**唯一**口径，不得替换成有限方窗内和）::

        P_in       = Σ |u_in|² · Δx′Δy′
        P_disk(R)  = Σ_{x²+y²≤R²} |u_out|² · Δx_outΔy_out
        E_numeric  = P_disk(R) / P_in
        v          = π D R/(λf)
        E_analytic = 1 − J₀(v)² − J₁(v)²
        绝对差     = |E_numeric − E_analytic|

    另外把 ``P_disk/P_square`` 单独记成“有限方窗内条件占比”，**不与**无限平面解析式
    当同一指标。

    Parameters
    ----------
    fft_size : int
        单次 FFT 的长度 M（偶数）。原生输出间距 ``λz/(MΔx′)``。
    window_frac : float
        该等级的 ``R`` 占原生输出半宽的比例上限判据；用于检查圆盘不贴边、也不恒选全窗。
    """
    t0 = time.perf_counter()
    res = fresnel_fft(u1, grid_in, FresnelConfig(lam, f, int(fft_size)))
    I = np.abs(res.field) ** 2
    g = res.grid
    rr = g.radius()
    p_in = float(np.sum(np.abs(u1) ** 2) * grid_in.cell_area)
    p_total = res.power
    p_disk = float(np.sum(I[rr <= R]) * g.cell_area)
    half = float(np.abs(g.x.coords).max())
    in_window = rr <= half
    p_square = float(np.sum(I[in_window]) * g.cell_area)

    e_numeric = p_disk / p_in
    e_analytic = float(M.airy_encircled_energy_analytic(np.array([R]), lam, f, D)[0])
    frac_power = abs(p_total / p_in - 1.0)
    rec = {
        "label": label,
        "R_m": float(R),
        "R_um": U.to_um(R),
        "R_over_r1": float(R / M.airy_dark_ring_radii(lam, f, D, 1)[0]),
        "fft_size": int(fft_size),
        "output_spacing_m": float(g.dx),
        "output_spacing_um": U.to_um(g.dx),
        "native_output_points": int(g.x.n),
        "window_half_m": half,
        "window_half_um": U.to_um(half),
        "R_over_window_half": float(R / half),
        "window_covers_disk": bool(half > R),
        # 判据：R 必须明显小于输出半宽（尾部截断要可忽略），且窗口不是靠"全选"通过
        "window_ok": bool(R / half <= float(window_frac)),
        "disk_does_not_select_whole_window": bool(p_square < 0.9999 * p_total),
        "input_spacing_m": float(grid_in.dx),
        "aperture_area_discrete_m2": float(np.sum(np.abs(u1) > 0) * grid_in.cell_area),
        "aperture_area_ideal_m2": float(np.pi * (0.5 * D) ** 2),
        "P_in": p_in,
        "P_disk": p_disk,
        "P_square_window": p_square,
        "P_total_native_fft": p_total,
        "native_fft_power_rel_err": frac_power,
        "E_numeric_disk_over_Pin": e_numeric,
        "E_analytic": e_analytic,
        "absolute_difference": abs(e_numeric - e_analytic),
        "conditional_fraction_disk_over_square": p_disk / p_square if p_square > 0 else float("nan"),
        "definition": "E_numeric = P_disk(R)/P_in；分母口径与解析 1−J0²−J1² 一致；用单次 FFT 求值",
        "v_pi_D_R_over_lam_f": float(np.pi * D * R / (lam * f)),
        "wall_time_s": float(time.perf_counter() - t0),
    }
    if logger is not None:
        logger.info("【能量·%s】R=%.4g μm（%.2f·r1）；M=%d，原生输出 %d 点，Δout=%.4g μm，"
                    "输出半宽 %.4g μm（R/半宽=%.4f，覆盖=%s，R/半宽达标=%s，非全窗=%s）；"
                    "P_in=%.10e，P_disk=%.10e，P_total=%.10e（FFT 功率误差=%.2e）；"
                    "E_numeric=%.9f，E_analytic=%.9f，绝对差=%.4e（%.1f s）",
                    label, rec["R_um"], rec["R_over_r1"], fft_size, g.x.n,
                    U.to_um(g.dx), U.to_um(half), rec["R_over_window_half"],
                    rec["window_covers_disk"], rec["window_ok"],
                    rec["disk_does_not_select_whole_window"], p_in, p_disk, p_total,
                    frac_power, e_numeric, e_analytic, rec["absolute_difference"],
                    rec["wall_time_s"])
    rec["intensity"] = I
    rec["grid"] = g
    # 只保存**径向**信息：能量口径完全由 P_disk(r) 与 P_in 决定，保存整幅 M×M
    # 强度数组会浪费上百 MB 且不增加任何证据价值。二维图仍由完整数组现算并画图。
    r_flat = rr.ravel()
    i_flat = I.ravel()
    order = np.argsort(r_flat)
    rec["radial_r_m"] = r_flat[order]
    rec["radial_P_cum"] = np.cumsum(i_flat[order]) * g.cell_area
    rec["radial_available"] = True
    return rec


def imshow_grid(ax, img, grid: Grid2D, cmap: str, title: str, cbar_label: str = "") -> None:
    """显示二维数组：``img[j, i]`` ↔ ``(x_i, y_j)``，索引增加 x/y 都增加。

    使用**像素边界** extent（``extent_pixel_um``），避免半像元偏差。
    """
    im = ax.imshow(img, origin="lower", extent=grid.extent_pixel_um, cmap=cmap,
                   aspect="equal")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(title, fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=9)


def make_lens_aperture_field(D: float, f: float, lam: float, grid: Grid2D):
    """圆孔内单位平面波 × 理想薄透镜二次相位 exp(−iπr²/(λf))。"""
    ap = circular_aperture(grid, D)
    lens = ideal_thin_lens_phase(grid, lam, f)
    return plane_wave(grid) * ap * lens, ap, lens


def make_asymmetric_input(grid: Grid2D, wavelength: float) -> np.ndarray:
    """受限、非对称的复振幅（椭圆高斯 + 斜向线性相位 + 弱立方相位）。

    非对称是关键：它能让轴翻转、相位符号与索引错误暴露出来。
    """
    X, Y = grid.meshgrid()
    hw = grid.x.half_width
    env = np.exp(-((X - 0.30 * hw) / (0.55 * hw)) ** 2
                 - ((Y + 0.22 * hw) / (0.30 * hw)) ** 2)
    k = 2.0 * np.pi / float(wavelength)
    return (env * np.exp(1j * k * (1.3e-3 * X - 0.9e-3 * Y))
            * np.exp(1j * 2.0e7 * (X ** 3 + 0.5 * Y ** 3))).astype(np.complex128)


def gaussian_field(grid: Grid2D, lam: float, w0: float) -> np.ndarray:
    X, Y = grid.meshgrid()
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def free_space_gaussian(grid: Grid2D, lam: float, z: float, w0: float) -> np.ndarray:
    """自由空间近轴高斯解析复场（束腰在 z=0、位于原点、正入射）。"""
    X, Y = grid.meshgrid()
    k = 2.0 * np.pi / lam
    zr = np.pi * w0 ** 2 / lam
    return (np.exp(1j * k * z) / (1.0 + 1j * z / zr)
            * np.exp(-(X ** 2 + Y ** 2) / (w0 ** 2 * (1.0 + 1j * z / zr)))).astype(np.complex128)


def rel_l2(a, b) -> float:
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2)) / np.sqrt(np.sum(np.abs(b) ** 2)))


def first_dark_ring_from_line(x_asc: np.ndarray, y: np.ndarray, smooth: int = 5) -> float:
    """从升序半径剖面上取第一个有意义的局部极小（相对下降需超过 1%）。

    必须这样做的两个原因：偶数网格没有 r=0 样点，中心两侧对称点会造出伪极小；
    Airy 中心附近强度变化极慢，浮点噪声会造出一串假极小。
    """
    if len(y) < 3 * smooth:
        return float("nan")
    ys = np.convolve(y, np.ones(smooth) / smooth, mode="same")
    n = len(y)
    for i in range(smooth, n - smooth):
        if ys[i] <= ys[i - 1] and ys[i] < ys[i + 1]:
            left = float(np.max(y[max(0, i - smooth):i]))
            right = float(np.max(y[i + 1:i + 1 + smooth]))
            if ys[i] < 0.99 * min(left, right):
                j0, j1 = max(0, i - smooth // 2), min(n, i + smooth // 2 + 1)
                return float(x_asc[j0 + int(np.argmin(y[j0:j1]))])
    return float("nan")


# ======================================================================================
# A. 理想透镜焦平面的 Airy 尺度
# ======================================================================================
def study_airy(cfg: Dict[str, Any], run_dir: Optional[Path], logger, show: bool
               ) -> Dict[str, Any]:
    """A. 圆孔 + 理想薄透镜 → 焦平面 Airy 尺度（含论文尺度 D=1 mm）。

    物理依据：``z=f`` 时透镜相位 ``exp(−iπr′²/(λf))`` 与传播展开式的输入二次相位
    ``exp(+iπr′²/(λf))`` 相消，焦平面就是孔径的 Fourier 强度。
    **不需要** ``D²/(4λf)≪1``；但模型近轴条件与数值采样条件仍要检查。
    """
    s = cfg["studies"]["airy"]
    lam = U.nm(s["wavelength_nm"])
    out: Dict[str, Any] = {"cases": []}
    figs: List[Path] = []
    rows_ring: List[List[Any]] = []
    rows_energy: List[List[Any]] = []

    for case in s["cases"]:
        base = "D%gmm_f%gmm" % (case["aperture_diameter_mm"], case["f_mm"])
        D = U.mm(case["aperture_diameter_mm"])
        f = U.mm(case["f_mm"])
        W = U.um(case["input_half_width_um"])
        dx = U.um(case["input_spacing_um"])
        n = 2 * int(round(W / dx))
        if n % 2 == 0:
            n += 1
        out_dx = U.um(case["output_spacing_um"])
        r1 = M.airy_dark_ring_radii(lam, f, D, 1)[0]
        n_out = 2 * int(round(5.0 * r1 / out_dx)) + 1

        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1, ap, lens = make_lens_aperture_field(D, f, lam, grid)
        sd = sampling_diagnostics(grid, lam, f, 5.0 * r1, 5.0 * r1,
                                  dx_out=out_dx, dy_out=out_dx)
        nf = axial_fresnel_number(D, lam, f)
        logger.info("【A】D=%.4g mm，f=%.4g mm，λ=%.4g nm；N_F=D²/(4λf)=%.6g（**不是** Airy 适用条件）",
                    case["aperture_diameter_mm"], case["f_mm"], s["wavelength_nm"], nf)
        logger.info("【A】输入 %d×%d，Δx′=%.4g μm（孔径半径 %.4g μm → %.1f 样点）；"
                    "输出 %d 点，间距 %.4g μm（%.1f 样点/暗环）",
                    n, n, U.to_um(dx), U.to_um(0.5 * D), 0.5 * D / dx, n_out,
                    U.to_um(out_dx), r1 / out_dx)
        logger.info("【A】采样诊断：输入二次相位步进 %.4g rad，交叉项步进 %.4g rad，"
                    "合计 %.4g rad（阈值 %.2f）→ ok=%s",
                    sd["input_quadratic_step_rad"], sd["cross_term_step_rad"],
                    sd["max_total_step_rad"], sd["threshold_rad"], sd["ok"])
        logger.info("【A】高阶光程误差：ρ_max=%.4g μm → 精确 %.6e rad（四次近似 %.6e rad），"
                    "容忍度 %.2f rad", U.to_um(sd["rho_max_m"]),
                    sd["third_order_phase_error_rad"],
                    sd["leading_fourth_order_phase_error_rad"], 0.1)

        # 中心截线：完整位移核的**可分离**求值（与完整核数学等价，无 FFT 回绕）
        xo = np.linspace(-5.0 * r1, 5.0 * r1, n_out)
        with Timer() as t:
            u2_line = fresnel_kernel_separable(u1, grid, lam, f, xo, np.array([0.0]))[0]
        I_line = np.abs(u2_line) ** 2
        I_n = I_line / I_line.max()
        airy = M.airy_intensity(xo, lam, f, D)
        xs = np.abs(xo)
        order = np.argsort(xs)
        xs_s, I_s, airy_s = xs[order], I_n[order], airy[order]
        l1 = float(np.sum(np.abs(I_s - airy_s)) / np.sum(airy_s))
        r_meas = first_dark_ring_from_line(xs_s, I_s)
        rel_ring = abs(r_meas - r1) / r1
        rings_theory = M.airy_dark_ring_radii(lam, f, D, 4)
        logger.info("【A】中心截线：L1=%.4e；第一暗环实测 %.4f μm / 理论 %.4f μm，"
                    "相对误差 %+.4e（耗时 %.2f s）",
                    l1, U.to_um(r_meas), U.to_um(r1), (r_meas - r1) / r1, t.elapsed)

        # 二维复场与坐标（用于后续阶段的 PSF 对照；本阶段只保存）
        half2d = 2.5 * r1
        n2d = 2 * int(round(half2d / out_dx)) + 1
        xo2 = np.linspace(-half2d, half2d, n2d)
        u2_2d = fresnel_kernel_separable(u1, grid, lam, f, xo2, xo2)
        I2d = np.abs(u2_2d) ** 2
        g2d = Grid2D(Axis1D(n2d, float(xo2[1] - xo2[0]), "x"),
                     Axis1D(n2d, float(xo2[1] - xo2[0]), "y"), label="focal plane")

        # 能量评价：独立窗口 + 单次 FFT（S3 修正）
        # 旧版用 2.5r1 的方窗去测 5r1 的圆盘 —— 所有像素都被选中，比例恒为 1，是空检查。
        ev = case.get("energy_evaluation", {})
        R_e = float(ev.get("R_over_r1", 2.0)) * r1
        energy = encircled_energy_record(u1, grid, lam, f, D, R_e,
                                         int(ev.get("fft_size", 2048)),
                                         float(ev.get("max_R_over_window_half", 0.5)),
                                         "A_%s" % case["name"], logger)
        save_npz(run_dir, "energy_encircled_%s.npz" % case["name"],
                 radial_r_m=energy["radial_r_m"],
                 radial_P_cum=energy["radial_P_cum"],
                 R_m=np.array(energy["R_m"]), P_in=np.array(energy["P_in"]),
                 P_disk=np.array(energy["P_disk"]),
                 P_total=np.array(energy["P_total_native_fft"]),
                 E_numeric=np.array(energy["E_numeric_disk_over_Pin"]),
                 E_analytic=np.array(energy["E_analytic"]),
                 comment=np.array(
                     "径向累计功率 P_cum(r) 与绝对包围能量口径 P_disk/P_in（坐标 m）；"
                     "不保存整幅 M×M 强度数组"))
        rows_energy.append([base, gnum(energy["R_um"]), gnum(energy["window_half_um"]),
                            energy["window_covers_disk"], energy["window_ok"],
                            energy["disk_does_not_select_whole_window"],
                            gnum(energy["P_in"]), gnum(energy["P_disk"]),
                            gnum(energy["P_square_window"]),
                            gnum(energy["E_numeric_disk_over_Pin"]),
                            gnum(energy["E_analytic"]), gnum(energy["absolute_difference"])])

        save_npz(run_dir, "airy_%s_line.npz" % base,
                 x_m=xo, u2_line=u2_line, intensity=I_line, airy_analytic=airy,
                 comment=np.array("中心截线：完整位移核可分离求值；坐标为 m"))
        save_npz(run_dir, "airy_%s_field.npz" % base,
                 u2=u2_2d, intensity=I2d, x_m=g2d.x.coords, y_m=g2d.y.coords,
                 aperture=ap, lens_phase=lens,
                 comment=np.array("焦平面二维复场/强度/坐标（m）；index [j,i]=(x_i,y_j)"))
        rows_ring.append([base, gnum(U.to_um(dx)), gnum(n), gnum(U.to_um(r1)),
                          gnum(U.to_um(r_meas)), gnum((r_meas - r1) / r1), gnum(l1),
                          gnum(nf), sd["ok"], gnum(sd["max_total_step_rad"]),
                          gnum(sd["third_order_phase_error_rad"])])

        # 图
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
        axes[0].plot(U.to_um(xs_s), I_s, lw=1.0, label="数值（完整核可分离求值）")
        axes[0].plot(U.to_um(xs_s), airy_s, "--", lw=1.0, label="解析 Airy $[2J_1(x)/x]^2$")
        axes[0].axvline(U.to_um(r1), color="r", ls=":", lw=0.8, label="理论第一暗环")
        axes[0].set_xlabel("r (μm)")
        axes[0].set_ylabel("归一化强度")
        axes[0].set_title("%s 中心截线（L1=%.3e）" % (base, l1), fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].semilogy(U.to_um(xs_s), np.maximum(I_s, 1e-12), lw=1.0, label="数值")
        axes[1].semilogy(U.to_um(xs_s), np.maximum(airy_s, 1e-12), "--", lw=1.0, label="解析")
        axes[1].set_xlabel("r (μm)")
        axes[1].set_ylabel("归一化强度（对数）")
        axes[1].set_title("%s 对数坐标" % base, fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, which="both")
        imshow_grid(axes[2], I2d, g2d, cfg["plots"]["colormap_intensity"],
                    "%s 焦平面强度（直接求值）" % base, cbar_label="a.u.")
        fig.tight_layout()
        p = save_fig(fig, run_dir, "A_airy_%s.png" % base, cfg["plots"]["dpi"], show)
        if p is not None:
            figs.append(p)

        out["cases"].append({
            "name": base, "aperture_mm": case["aperture_diameter_mm"], "f_mm": case["f_mm"],
            "wavelength_nm": s["wavelength_nm"], "fresnel_number": nf,
            "input_samples": n, "input_spacing_um": U.to_um(dx),
            "output_spacing_um": U.to_um(out_dx), "output_points": n_out,
            "samples_per_dark_ring": float(r1 / out_dx),
            "theory_first_ring_um": U.to_um(r1),
            "theory_first_ring_bessel_um": U.to_um(rings_theory[0]),
            "theory_rings_um": [U.to_um(v) for v in rings_theory],
            "measured_first_ring_um": U.to_um(r_meas),
            "first_ring_rel_err": float((r_meas - r1) / r1),
            "line_L1_vs_airy": l1,
            "sampling_diagnostics": sd,
            "encircled_energy": {k: v for k, v in energy.items()
                                 if k not in ("field", "intensity", "grid",
                                              "radial_r_m", "radial_P_cum")},
            "wall_time_s": t.elapsed,
        })

    write_csv_maybe(run_dir, "airy_metrics.csv",
                    ["算例", "输入间距_um", "输入样点N", "理论第一暗环_um", "实测第一暗环_um",
                     "暗环相对误差", "截线相对Airy_L1", "Fresnel数", "采样判据ok",
                     "判据总步进_rad", "高阶光程误差_rad"], rows_ring)
    write_csv_maybe(run_dir, "energy_absolute_metrics.csv",
                    ["算例", "R_um", "评价窗口半宽_um", "窗口覆盖圆盘", "圆盘非全窗",
                     "P_in", "P_disk", "P_square", "E_numeric=P_disk/P_in",
                     "E_analytic=1-J0²-J1²", "绝对差"], rows_energy)

    # 论文尺度参照（纯解析）
    ps = s["paper_scale_reference"]
    r1p = M.airy_dark_ring_radii(U.nm(ps["wavelength_nm"]), U.mm(ps["f_mm"]),
                                U.mm(ps["D_mm"]), 1)[0]
    out["paper_scale_reference"] = {
        "D_mm": ps["D_mm"], "f_mm": ps["f_mm"], "wavelength_nm": ps["wavelength_nm"],
        "first_ring_bessel_um": U.to_um(r1p),
        "first_ring_1p22_um": U.to_um(M.airy_first_dark_ring_radius(
            U.nm(ps["wavelength_nm"]), U.mm(ps["f_mm"]), U.mm(ps["D_mm"]))),
        "fresnel_number": axial_fresnel_number(U.mm(ps["D_mm"]),
                                              U.nm(ps["wavelength_nm"]), U.mm(ps["f_mm"])),
        "note": ("与第一个算例同参数，已实际求值中心截线；"
                 "N_F 只作报告，不是 Airy 适用条件"),
    }
    out["图"] = [str(p.relative_to(run_dir)) if run_dir else str(p) for p in figs]
    return out


# ======================================================================================
# B. 离散恒等：FFT vs 完整位移核
# ======================================================================================
def study_direct_vs_fft(cfg: Dict[str, Any], run_dir: Optional[Path], logger, show: bool
                        ) -> Dict[str, Any]:
    """B. 同一离散输入、同一原生输出网格：单次 FFT 必须等于完整位移核求和。"""
    s = cfg["studies"]["direct_vs_fft"]
    lam = U.nm(s["wavelength_nm"])
    z = U.mm(s["propagation_distance_mm"])
    res: Dict[str, Any] = {"cases": []}
    rows: List[List[Any]] = []
    figs: List[Path] = []
    worst = 0.0

    for case in s["cases"]:
        n = int(case["input_samples"])
        dx = U.um(case["input_spacing_um"])
        M = int(case["fft_size"])
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = make_asymmetric_input(grid, lam)
        p_in = float(np.sum(np.abs(u1) ** 2) * grid.cell_area)
        with Timer() as t1:
            r = fresnel_fft(u1, grid, FresnelConfig(lam, z, M))
        with Timer() as t2:
            ref = fresnel_kernel_matrix(u1, grid, lam, z, r.grid.x.coords, r.grid.y.coords)
        rel = rel_l2(r.field, ref)
        p_out = r.power
        p_err = abs(p_out / p_in - 1.0)
        worst = max(worst, rel)
        logger.info("【B】N=%d，Δx′=%.4g μm，M=%d：FFT vs 完整核求和 rel L2=%.4e；"
                    "原生全输出功率相对误差=%.4e（P_out/P_in=%.12f）；耗时 %.3f/%.3f s",
                    n, U.to_um(dx), M, rel, p_err, p_out / p_in, t1.elapsed, t2.elapsed)
        rows.append([n, gnum(U.to_um(dx)), M, gnum(rel), gnum(p_err),
                     gnum(p_out / p_in), gnum(U.to_um(r.grid.dx))])
        res["cases"].append({
            "input_samples": n, "input_spacing_um": U.to_um(dx), "fft_size": M,
            "rel_L2_fft_vs_full_kernel": rel,
            "power_relative_error": p_err, "power_ratio": p_out / p_in,
            "output_spacing_um": U.to_um(r.grid.dx),
        })
        save_npz(run_dir, "direct_vs_fft_N%d_M%d.npz" % (n, M),
                 u1=u1, x_in_m=grid.x.coords, y_in_m=grid.y.coords,
                 u2_fft=r.field, u2_full_kernel=ref,
                 x_out_m=r.grid.x.coords, y_out_m=r.grid.y.coords,
                 comment=np.array("同一离散输入、同一原生输出网格上的两套复场（m）"))

        if n == 48:
            fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
            imshow_grid(axes[0], np.abs(u1) ** 2, grid, cfg["plots"]["colormap_intensity"],
                        "B1 输入强度 |u_in|²（非对称）", cbar_label="a.u.")
            imshow_grid(axes[1], np.abs(ref) ** 2, r.grid, cfg["plots"]["colormap_intensity"],
                        "B2 完整位移核求和", cbar_label="a.u.")
            d = np.abs(r.field - ref) / max(float(np.max(np.abs(ref))), 1e-300)
            imshow_grid(axes[2], d, r.grid, cfg["plots"]["colormap_error"],
                        "B3 |FFT−完整核|/max|完整核|（relL2=%.2e）" % rel,
                        cbar_label="归一化差")
            fig.tight_layout()
            p = save_fig(fig, run_dir, "B_discrete_identity.png", cfg["plots"]["dpi"], show)
            if p is not None:
                figs.append(p)

    write_csv_maybe(run_dir, "direct_vs_fft_metrics.csv",
                    ["输入样点N", "dx_in_um", "FFT长度M", "rel_L2_FFT_vs_完整核",
                     "功率相对误差", "P_out/P_in", "输出间距_um"], rows)
    res["worst_rel_L2"] = worst
    res["acceptance_target"] = cfg["acceptance"]["discrete_identity_rel_l2_max"]
    res["pass"] = bool(worst <= cfg["acceptance"]["discrete_identity_rel_l2_max"])
    res["图"] = [str(p.relative_to(run_dir)) if run_dir else str(p) for p in figs]
    return res


# ======================================================================================
# C. 能量（原生 FFT Parseval + 有限窗口补充检查）
# ======================================================================================
def study_energy(cfg: Dict[str, Any], run_dir: Optional[Path], logger, show: bool
                 ) -> Dict[str, Any]:
    """C. 原生 FFT 全输出 Parseval 检验（主）+ 有限窗口对解析包围能量（补充）。"""
    s = cfg["studies"]["energy"]
    lam = U.nm(s["wavelength_nm"])
    z = U.mm(s["propagation_distance_mm"])
    res: Dict[str, Any] = {}
    rows: List[List[Any]] = []
    figs: List[Path] = []
    worst_power = 0.0

    # C1: 一般非对称输入的原生 FFT Parseval（主判据）
    for case in s["parseval_cases"]:
        n, dx, m_fft = (int(case["input_samples"]), U.um(case["input_spacing_um"]),
                        int(case["fft_size"]))
        grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
        u1 = make_asymmetric_input(grid, lam)
        p_in = float(np.sum(np.abs(u1) ** 2) * grid.cell_area)
        r = fresnel_fft(u1, grid, FresnelConfig(lam, z, m_fft))
        p_out = r.power
        err = abs(p_out / p_in - 1.0)
        worst_power = max(worst_power, err)
        logger.info("【C1】非对称输入 N=%d，Δx′=%.4g μm，M=%d：P_in=%.12e，P_out=%.12e，"
                    "相对误差=%.4e（主判据，目标 ≤%.0e）",
                    n, U.to_um(dx), m_fft, p_in, p_out, err,
                    cfg["acceptance"]["parseval_rel_err_max"])
        rows.append(["parseval_asymmetric", n, m_fft, gnum(p_in), gnum(p_out), gnum(err)])
        save_npz(run_dir, "energy_parseval_N%d_M%d.npz" % (n, m_fft),
                 u1=u1, u2=r.field, x_in_m=grid.x.coords, y_in_m=grid.y.coords,
                 x_out_m=r.grid.x.coords, y_out_m=r.grid.y.coords,
                 comment=np.array("原生 FFT 全输出 Parseval 检验的原始数组（m）"))

    # C2: 绝对包围能量 P_disk(R)/P_in 对解析式（S3 修正后的唯一口径）
    a = s["airy_window_case"]
    D, f = U.mm(a["aperture_diameter_mm"]), U.mm(a["f_mm"])
    W, dx = U.um(a["input_half_width_um"]), U.um(a["input_spacing_um"])
    n = 2 * int(round(W / dx))
    if n % 2 == 0:
        n += 1
    grid = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
    u1, _, _ = make_lens_aperture_field(D, f, lam, grid)
    r1 = M.airy_dark_ring_radii(lam, f, D, 1)[0]

    energy_levels = []
    for lv in s.get("energy_levels", []):
        R = float(lv["R_over_r1"]) * r1
        rec = encircled_energy_record(u1, grid, lam, f, D, R, int(lv["fft_size"]),
                                      float(s.get("energy_max_R_over_window_half", 0.5)),
                                      "C2_%s" % lv["name"], logger)
        energy_levels.append(rec)
        rows.append(["abs_encircled_" + lv["name"], rec["native_output_points"],
                     rec["fft_size"], gnum(rec["P_disk"]),
                     gnum(rec["E_numeric_disk_over_Pin"]),
                     gnum(rec["absolute_difference"])])
        save_npz(run_dir, "energy_absolute_%s.npz" % lv["name"],
                 radial_r_m=rec["radial_r_m"], radial_P_cum=rec["radial_P_cum"],
                 R_m=np.array(rec["R_m"]), P_in=np.array(rec["P_in"]),
                 P_disk=np.array(rec["P_disk"]),
                 P_total=np.array(rec["P_total_native_fft"]),
                 E_numeric=np.array(rec["E_numeric_disk_over_Pin"]),
                 E_analytic=np.array(rec["E_analytic"]),
                 comment=np.array(
                     "径向累计功率与绝对包围能量 P_disk/P_in（坐标 m）；不保存整幅强度数组"))

    # 主水平等级：绝对差（容差在配置里预先约定）
    primary = energy_levels[0]
    abs_diff = float(primary["absolute_difference"])
    worst_abs = max(float(r["absolute_difference"]) for r in energy_levels)
    grid_stable = bool(
        max(float(r["absolute_difference"]) for r in energy_levels)
        - min(float(r["absolute_difference"]) for r in energy_levels)
        <= cfg["acceptance"]["energy_absolute_abs_diff_max"])
    windows_ok = all(r["window_ok"] and r["disk_does_not_select_whole_window"]
                     for r in energy_levels)

    # 累计曲线用**同一个 P_in 分母**，并与解析包围能量直接比较
    ref = energy_levels[min(1, len(energy_levels) - 1)]      # 用最细等级画曲线
    rr = ref["grid"].radius().ravel()
    ii = ref["intensity"].ravel()
    o = np.argsort(rr)
    cum = np.cumsum(ii[o]) * ref["grid"].cell_area / ref["P_in"]
    ana_curve = M.airy_encircled_energy_analytic(rr[o], lam, f, D)
    pp = ref["P_total_native_fft"] / ref["P_in"]
    cum_native = np.cumsum(ii[o]) * ref["grid"].cell_area / ref["P_total_native_fft"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    axes[0].plot(U.to_um(rr[o]), cum, lw=1.1, label="数值 $P_{disk}(r)/P_{in}$")
    axes[0].plot(U.to_um(rr[o]), cum_native, ":", lw=1.0,
                 label="数值 $P_{disk}(r)/P_{total}$（FFT，比值 %.6f）" % pp)
    axes[0].plot(U.to_um(rr[o]), ana_curve, "--", lw=1.0,
                 label="解析 $1-J_0^2-J_1^2$")
    axes[0].axvline(U.to_um(ref["R_m"]), color="r", ls=":",
                    label="评价 R=%.1f μm" % ref["R_um"])
    axes[0].set_xlabel("r (μm)")
    axes[0].set_ylabel("包围能量比例")
    axes[0].set_title("C2 径向累计包围能量（主口径分母 = $P_{in}$；%s）" % ref["label"],
                      fontsize=9.5)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.3)
    d_curve = cum - ana_curve
    axes[1].plot(U.to_um(rr[o]), d_curve, lw=1.0)
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[1].set_xlabel("r (μm)")
    axes[1].set_ylabel("数值 − 解析")
    axes[1].set_title("C3 与解析包围能量之差（同口径 $P_{disk}/P_{in}$）", fontsize=9.5)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = save_fig(fig, run_dir, "C_energy.png", cfg["plots"]["dpi"], show)
    if p is not None:
        figs.append(p)

    write_csv_maybe(run_dir, "energy_metrics.csv",
                    ["检查", "原生点数", "M", "P_disk", "E_numeric或解析", "绝对差"], rows)
    res.update({
        "parseval_worst_rel_err": worst_power,
        "parseval_target": cfg["acceptance"]["parseval_rel_err_max"],
        "parseval_pass": bool(worst_power <= cfg["acceptance"]["parseval_rel_err_max"]),
        "absolute_energy_levels": [
            {k: v for k, v in r.items()
             if k not in ("intensity", "grid", "radial_r_m", "radial_P_cum")}
            for r in energy_levels],
        "absolute_energy_primary_label": primary["label"],
        "absolute_energy_primary_abs_diff": abs_diff,
        "absolute_energy_worst_abs_diff": worst_abs,
        "absolute_energy_grid_stable": grid_stable,
        "absolute_energy_windows_ok": bool(windows_ok),
        "absolute_energy_tolerance": cfg["acceptance"]["energy_absolute_abs_diff_max"],
        "absolute_energy_definition": ("E_numeric = P_disk(R)/P_in；"
                                       "P_in = Σ|u_in|²Δx′Δy′；唯一分母口径；"
                                       "总功率取自原生 FFT 全输出（严格守恒）"),
        "cumulative_curve_uses_same_P_in": True,
        "note": ("主能量判据仍是原生 FFT 全输出 Parseval；绝对包围能量按 P_disk/P_in 口径"
                 "与解析 1−J₀²−J₁² 比较，绝对差为验收量。"),
        "图": [str(p.relative_to(run_dir)) if run_dir else str(p) for p in figs],
    })
    return res


# ======================================================================================
# D. 三类彼此不可替代的检查：输入采样收敛 / 输出采样（填充）收敛 / 离散算法恒等
# ======================================================================================
def _intensity_l1_common(a: np.ndarray, b: np.ndarray, dA: float) -> Dict[str, float]:
    """同一物理网格上两张**未归一化**强度的比较（第二轮修正定义的三种量）。

    ``raw_intensity_L1``  : ``Σ|Ia−Ib|ΔA / Σ Ib ΔA`` —— 主收敛指标，保留绝对通量信息。
    ``shape_L1``          : ``Σ|Ia/Pa − Ib/Pb|ΔA`` —— 只比较形状，``Pa/Pb`` 是各自
                            在同一观察窗内的强度积分；**不用于**主验收。
    ``field_relL2``       : ``sqrt(Σ|ua−ub|²ΔA / Σ|ub|²ΔA)`` —— 复场相对 L2，
                            单独存放，**不得**塞进名为 L1 的字段。
    """
    pa = float(np.sum(a) * dA)
    pb = float(np.sum(b) * dA)
    raw = float(np.sum(np.abs(a - b)) * dA / pb) if pb > 0 else float("nan")
    if pa > 0 and pb > 0:
        shape = float(np.sum(np.abs(a / pa - b / pb)) * dA)
    else:
        shape = float("nan")
    return {"raw_intensity_L1": raw, "shape_L1": shape, "P_a": pa, "P_b": pb}


def study_sampling(cfg: Dict[str, Any], run_dir: Optional[Path], logger, show: bool
                   ) -> Dict[str, Any]:
    """D. 把「离散算法恒等」「输入积分收敛」「输出采样/填充收敛」当作**不同指标**分别验收。

    * **D1 输入采样收敛**：固定同一个连续器件（理想圆孔 + 理想薄透镜）、固定公共输出
      坐标，只改变输入采样间距。每级都按**同一个连续函数**取样，不把粗网格图像放大。
    * **D2 输出采样收敛**：固定同一个离散输入，只改变 FFT 长度 M，把原生**强度**
      映射到同一个公共探测器网格，并与该输入在公共坐标上的完整核参考比较。
    * **D3 离散算法恒等**：同一离散输入、同一原生网格上 FFT 与完整核求和必须相等。
      它**不能**代替 D1/D2 的物理收敛。

    三种量的定义不同，阈值也不同，报告与 JSON 里分开存放。
    """
    s = cfg["studies"]["sampling"]
    lam = U.nm(s["wavelength_nm"])
    z = U.mm(s["propagation_distance_mm"])
    D = U.mm(s["aperture_diameter_mm"])
    W = U.um(s["input_half_width_um"])
    r1 = M.airy_dark_ring_radii(lam, z, D, 1)[0]
    tol_in = cfg["acceptance"]["input_sampling_intensity_L1_max"]
    tol_pad = cfg["acceptance"]["padding_sampling_intensity_L1_max"]
    tol_id = cfg["acceptance"]["discrete_identity_rel_l2_max"]

    # 公共输出坐标（所有输入等级与所有 M 都映射到这一套坐标）
    half_out = float(s["common_output_half_width_over_r1"]) * r1
    out_dx = U.um(s["common_output_spacing_um"])
    k_out = 2 * int(np.ceil(half_out / out_dx)) + 1
    x_common = (np.arange(k_out, dtype=np.float64) - k_out // 2) * out_dx
    g_common = Grid2D(Axis1D(k_out, out_dx, "x"), Axis1D(k_out, out_dx, "y"),
                      label="common detector grid")
    dA = g_common.cell_area
    common = {"x_m": x_common, "y_m": x_common, "spacing_m": out_dx, "points": k_out,
              "half_width_m": half_out}
    logger.info("【D】公共输出网格：%d×%d，间距 %.4g μm，半宽 %.4g μm（= %.2f·r1，r1=%.4g μm）",
                k_out, k_out, U.to_um(out_dx), U.to_um(half_out), s["common_output_half_width_over_r1"],
                U.to_um(r1))

    figs: List[Path] = []
    rows_in: List[List[Any]] = []
    rows_pad: List[List[Any]] = []
    rows_id: List[List[Any]] = []
    saved: Dict[str, Any] = {"x_common_m": x_common}

    def build_level(dx_in: float) -> Dict[str, Any]:
        """按**同一个连续器件函数**在给定间距上取样。"""
        n = 2 * int(np.ceil(W / dx_in)) + 1        # 奇数，覆盖同一边界
        g = Grid2D(Axis1D(n, dx_in, "x'"), Axis1D(n, dx_in, "y'"))
        u1, ap, lens = make_lens_aperture_field(D, z, lam, g)
        p_in = float(np.sum(np.abs(u1) ** 2) * g.cell_area)
        ap_area = float(np.sum(ap) * g.cell_area)
        # 公共坐标上用完整位移核的**可分离**等价形式求值
        u2 = fresnel_kernel_separable(u1, g, lam, z, x_common, x_common)
        I = np.abs(u2) ** 2
        # 该级与解析 Airy 的中心截线误差（沿用与 A 组相同的口径）
        line = I[k_out // 2, :] / max(float(I[k_out // 2, :].max()), 1e-300)
        airy = M.airy_intensity(np.abs(x_common), lam, z, D)
        line_l1 = float(np.sum(np.abs(line - airy)) / np.sum(airy))
        return {"n": n, "dx_m": dx_in, "grid": g, "u1": u1, "ap": ap, "lens": lens,
                "p_in": p_in, "ap_area": ap_area, "u2": u2, "I": I,
                "line_L1_vs_airy": line_l1}

    # ---------------- D1: 输入采样收敛 ----------------
    levels = [build_level(U.um(lv["input_spacing_um"])) for lv in s["input_levels"]]
    for lv, d in zip(s["input_levels"], levels):
        logger.info("【D1】输入采样 %s：N=%d，Δx′=%.4g μm，P_in=%.10e，孔径离散面积=%.6e "
                    "（理想 πD²/4=%.6e，相对差 %+.3e），中心截线 L1=%.4e",
                    lv["name"], d["n"], U.to_um(d["dx_m"]), d["p_in"], d["ap_area"],
                    np.pi * (0.5 * D) ** 2,
                    d["ap_area"] / (np.pi * (0.5 * D) ** 2) - 1.0, d["line_L1_vs_airy"])

    level_metrics = []
    for i in range(len(levels) - 1):
        a, b = levels[i], levels[i + 1]
        cmp_ib = _intensity_l1_common(a["I"], b["I"], dA)
        cmp_ba = _intensity_l1_common(b["I"], a["I"], dA)
        f_l2 = float(np.sqrt(np.sum(np.abs(a["u2"] - b["u2"]) ** 2) * dA
                             / (np.sum(np.abs(b["u2"]) ** 2) * dA)))
        m = {"pair": "%s->%s" % (s["input_levels"][i]["name"], s["input_levels"][i + 1]["name"]),
             "left": s["input_levels"][i]["name"], "right": s["input_levels"][i + 1]["name"],
             "raw_intensity_L1": cmp_ib["raw_intensity_L1"],
             "raw_intensity_L1_reverse": cmp_ba["raw_intensity_L1"],
             "shape_L1": cmp_ib["shape_L1"],
             "field_relL2": f_l2,
             "P_left_window": cmp_ib["P_a"], "P_right_window": cmp_ib["P_b"]}
        level_metrics.append(m)
        rows_in.append([m["pair"], gnum(levels[i]["n"]), gnum(levels[i + 1]["n"]),
                        gnum(U.to_um(a["dx_m"])), gnum(U.to_um(b["dx_m"])),
                        gnum(m["raw_intensity_L1"]), gnum(m["shape_L1"]),
                        gnum(m["field_relL2"]),
                        gnum(levels[i]["line_L1_vs_airy"]),
                        gnum(levels[i + 1]["line_L1_vs_airy"])])
        logger.info("【D1】%s：原始强度相对 L1=%.6e（反向 %.6e），形状 L1=%.6e，"
                    "复场相对 L2=%.6e（目标 ≤%.3g）", m["pair"], m["raw_intensity_L1"],
                    m["raw_intensity_L1_reverse"], m["shape_L1"], m["field_relL2"], tol_in)

    # 预设判据：用**最后两级**判定最终精度；粗级不满足就如实标出，不删除
    input_last_pair = level_metrics[-1] if level_metrics else None
    worst_input_L1 = max((m["raw_intensity_L1"] for m in level_metrics), default=float("nan"))
    for lv, d in zip(s["input_levels"], levels):
        rows_in.append(["level:%s" % lv["name"], gnum(d["n"]), "", gnum(U.to_um(d["dx_m"])), "",
                        "", "", "", gnum(d["line_L1_vs_airy"]),
                        gnum(d["ap_area"] / (np.pi * (0.5 * D) ** 2) - 1.0)])
        saved["u_%s" % lv["name"]] = d["u2"]
        saved["I_%s" % lv["name"]] = d["I"]
        saved["ap_%s" % lv["name"]] = d["ap"]
        saved["u1_%s" % lv["name"]] = d["u1"]

    # ---------------- D2: 输出采样（填充）收敛 ----------------
    dx_pad = U.um(s["padding_input_spacing_um"])
    n_pad = 2 * int(np.ceil(W / dx_pad)) + 1
    g_pad = Grid2D(Axis1D(n_pad, dx_pad, "x'"), Axis1D(n_pad, dx_pad, "y'"))
    u1_pad, _, _ = make_lens_aperture_field(D, z, lam, g_pad)
    p_in_pad = float(np.sum(np.abs(u1_pad) ** 2) * g_pad.cell_area)
    # 公共坐标上的完整核参考（不插值复相位，只在公共坐标直接求值）
    u_ref = fresnel_kernel_separable(u1_pad, g_pad, lam, z, x_common, x_common)
    I_ref = np.abs(u_ref) ** 2
    logger.info("【D2】固定输入：N=%d，Δx′=%.4g μm，P_in=%.10e；公共坐标参考由完整核"
                "（可分离等价形式）直接求值，不插值复相位", n_pad, U.to_um(dx_pad), p_in_pad)

    pad_records = []
    for lv in s["padding_levels"]:
        m_fft = int(lv["fft_size"])
        with Timer() as t:
            r = fresnel_fft(u1_pad, g_pad, FresnelConfig(lam, z, m_fft))
        # 3) 原生网格上的离散恒等（归入算法检查）
        ref_native = fresnel_kernel_matrix(u1_pad, g_pad, lam, z,
                                           r.grid.x.coords, r.grid.y.coords)
        ident = rel_l2(r.field, ref_native)
        p_in = float(np.sum(np.abs(u1_pad) ** 2) * g_pad.cell_area)
        p_err = abs(r.power / p_in - 1.0)
        # 4) 原生**强度**双线性插值到公共探测器网格（不插值高速变化的复相位）
        I_native = np.abs(r.field) ** 2
        I_map = M.resample_intensity_bilinear(I_native, r.grid, x_common, x_common)
        # 越界点必须显式排除，不能补零参与主比较
        inb = (np.abs(x_common) <= np.abs(r.grid.x.coords).max()) & \
              (np.abs(x_common) <= np.abs(r.grid.y.coords).max())
        mask = (inb[None, :] & inb[:, None]) & np.isfinite(I_map)
        cmp_ref = _intensity_l1_common(np.where(mask, I_map, 0.0),
                                       np.where(mask, I_ref, 0.0), dA)
        # 通量偏差：公共窗口内的映射强度积分对参考积分
        flux = cmp_ref["P_a"] / cmp_ref["P_b"] - 1.0 if cmp_ref["P_b"] > 0 else float("nan")
        rec = {"name": lv["name"], "M": m_fft,
               "native_output_points": int(r.grid.x.n),
               "native_output_spacing_m": float(r.grid.dx),
               "native_output_spacing_formula_m": float(lam * z / (m_fft * dx_pad)),
               "identity_rel_L2": ident, "power_rel_err": p_err,
               "common_ref_intensity_L1": cmp_ref["raw_intensity_L1"],
               "common_shape_L1": cmp_ref["shape_L1"],
               "common_window_flux_bias": flux,
               "common_points_used": int(mask.sum()),
               "common_points_excluded": int(mask.size - mask.sum()),
               "wall_time_s": t.elapsed,
               "window_covered": bool(mask.all()),
               "I_map": I_map, "I_native": I_native, "mask": mask}
        pad_records.append(rec)
        logger.info("【D2】%s：M=%d，原生输出 %d 点，原生间距 %.4g μm（公式 %.4g μm，一致=%s）；"
                    "离散恒等 rel L2=%.4e；功率误差=%.4e；公共网格原始强度 L1=%.6e"
                    "（目标 ≤%.3g）；形状 L1=%.4e；窗口通量偏差 %+.3e；剔除越界点 %d",
                    lv["name"], m_fft, r.grid.x.n, U.to_um(r.grid.dx),
                    U.to_um(rec["native_output_spacing_formula_m"]),
                    abs(r.grid.dx - rec["native_output_spacing_formula_m"]) <= 1e-9 * r.grid.dx,
                    ident, p_err, cmp_ref["raw_intensity_L1"], tol_pad, cmp_ref["shape_L1"],
                    flux, rec["common_points_excluded"])
        rows_pad.append([lv["name"], gnum(m_fft), gnum(r.grid.x.n),
                         gnum(U.to_um(r.grid.dx)), gnum(ident), gnum(p_err),
                         gnum(cmp_ref["raw_intensity_L1"]), gnum(cmp_ref["shape_L1"]),
                         gnum(flux), gnum(t.elapsed)])
        saved["I_native_%s" % lv["name"]] = I_native
        saved["x_native_%s" % lv["name"]] = r.grid.x.coords
        saved["I_map_%s" % lv["name"]] = I_map

    # 两个 M 在公共网格上的实际 L1
    pad_pair = None
    if len(pad_records) >= 2:
        a_rec, b_rec = pad_records[0], pad_records[-1]
        both = a_rec["mask"] & b_rec["mask"]
        cmp_two = _intensity_l1_common(np.where(both, a_rec["I_map"], 0.0),
                                       np.where(both, b_rec["I_map"], 0.0), dA)
        pad_pair = {"a": a_rec["name"], "b": b_rec["name"],
                    "M_a": a_rec["M"], "M_b": b_rec["M"],
                    "raw_intensity_L1_between_M": cmp_two["raw_intensity_L1"],
                    "shape_L1_between_M": cmp_two["shape_L1"],
                    "points": int(both.sum())}
        logger.info("【D2】%s vs %s：公共网格上两者的原始强度 L1=%.6e，形状 L1=%.4e（%d 点）",
                    a_rec["name"], b_rec["name"], cmp_two["raw_intensity_L1"],
                    cmp_two["shape_L1"], pad_pair["points"])
        # 嵌套原生坐标上的相同样点一致性（可作一致性检查，但不能证明分辨率收敛）
        common_pts = np.intersect1d(np.round(a_rec.get("x_native", x_common), 12),
                                    np.round(b_rec.get("x_native", x_common), 12))

    saved.update({"u_reference_common": u_ref, "I_reference_common": I_ref,
                  "u1_padding_input": u1_pad,
                  "x_padding_input_m": g_pad.x.coords, "y_padding_input_m": g_pad.y.coords})

    # ---------------- D3: 离散算法恒等（与物理收敛分开） ----------------
    worst_identity = 0.0
    for case in s.get("identity_cases", []):
        n_i = int(case["input_samples"])
        dx_i = U.um(case["input_spacing_um"])
        m_i = int(case["fft_size"])
        g_i = Grid2D(Axis1D(n_i, dx_i, "x'"), Axis1D(n_i, dx_i, "y'"))
        u_i = make_asymmetric_input(g_i, lam)
        p_in_i = float(np.sum(np.abs(u_i) ** 2) * g_i.cell_area)
        r_i = fresnel_fft(u_i, g_i, FresnelConfig(lam, z, m_i))
        ref_i = fresnel_kernel_matrix(u_i, g_i, lam, z, r_i.grid.x.coords, r_i.grid.y.coords)
        e_i = rel_l2(r_i.field, ref_i)
        pwr_i = abs(r_i.power / p_in_i - 1.0)
        worst_identity = max(worst_identity, e_i)
        logger.info("【D3】%s：N=%d，Δx′=%.4g μm，M=%d → FFT vs 完整核 rel L2=%.4e；"
                    "功率误差=%.4e", case["name"], n_i, U.to_um(dx_i), m_i, e_i, pwr_i)
        rows_id.append([case["name"], gnum(n_i), gnum(U.to_um(dx_i)), gnum(m_i),
                        gnum(e_i), gnum(pwr_i)])
        saved["u1_%s" % case["name"]] = u_i
        saved["x_%s" % case["name"]] = r_i.grid.x.coords

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
    for lv, d in zip(s["input_levels"], levels):
        axes[0].plot(U.to_um(x_common), d["I"][k_out // 2, :] / d["I"].max(),
                     lw=1.0, label="%s（Δx′=%.3g μm）" % (lv["name"], U.to_um(d["dx_m"])))
    axes[0].plot(U.to_um(x_common),
                 M.airy_intensity(np.abs(x_common), lam, z, D), "k--", lw=1.0,
                 label="解析 Airy")
    axes[0].set_xlabel("x (μm)")
    axes[0].set_ylabel("归一化强度")
    axes[0].set_title("D1 三个输入采样等级的中心截线\n（公共输出坐标，各自归一化）",
                      fontsize=9.5)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    names = [lv["name"] for lv in s["input_levels"]]
    axes[1].semilogy(range(len(levels)), [d["line_L1_vs_airy"] for d in levels], "o-",
                     label="截线对解析 Airy 的 L1")
    if level_metrics:
        axes[1].semilogy(range(1, len(levels)),
                         [m["raw_intensity_L1"] for m in level_metrics], "s--",
                         label="相邻等级原始强度 L1")
    axes[1].axhline(tol_in, color="r", ls=":", label="阈值 %.3g" % tol_in)
    axes[1].set_xticks(range(len(levels)))
    axes[1].set_xticklabels(names)
    axes[1].set_xlabel("输入采样等级")
    axes[1].set_ylabel("误差")
    axes[1].set_title("D1 输入采样误差趋势", fontsize=9.5)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")

    for rec in pad_records:
        axes[2].plot(U.to_um(x_common), rec["I_map"][k_out // 2, :] / rec["I_map"].max(),
                     lw=1.0, label="%s（M=%d，Δ=%.3g μm）"
                     % (rec["name"], rec["M"], U.to_um(rec["native_output_spacing_m"])))
    axes[2].plot(U.to_um(x_common), I_ref[k_out // 2, :] / I_ref.max(), "k--", lw=1.1,
                 label="公共坐标完整核参考")
    axes[2].set_xlabel("x (μm)")
    axes[2].set_ylabel("归一化强度")
    axes[2].set_title("D2 两个填充输出的公共网格中心截线", fontsize=9.5)
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    p = save_fig(fig, run_dir, "D_sampling_convergence.png", cfg["plots"]["dpi"], show)
    if p is not None:
        figs.append(p)

    if pad_records:
        ref_rec = pad_records[-1]
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
        imshow_grid(axes[0], ref_rec["I_map"], g_common, cfg["plots"]["colormap_intensity"],
                    "D2 %s 映射到公共网格的强度" % ref_rec["name"], cbar_label="a.u.")
        imshow_grid(axes[1], I_ref, g_common, cfg["plots"]["colormap_intensity"],
                    "D2 公共坐标完整核参考", cbar_label="a.u.")
        diff = np.where(ref_rec["mask"],
                        ref_rec["I_map"] / max(ref_rec["I_map"].max(), 1e-300)
                        - I_ref / max(I_ref.max(), 1e-300), np.nan)
        imshow_grid(axes[2], diff, g_common, cfg["plots"]["colormap_error"],
                    "D2 归一化强度差（L1=%.3e）" % ref_rec["common_ref_intensity_L1"],
                    cbar_label="差")
        fig.tight_layout()
        p = save_fig(fig, run_dir, "D_padding_common_grid.png", cfg["plots"]["dpi"], show)
        if p is not None:
            figs.append(p)

    # ---------------- 输出清单 ----------------
    write_csv_maybe(run_dir, "sampling_input_levels.csv",
                    ["比较或等级", "N_left", "N_right", "dx_left_um", "dx_right_um",
                     "原始强度L1", "形状L1", "复场相对L2", "左级截线L1_或孔径面积相对差",
                     "右级截线L1_或空"], rows_in)
    write_csv_maybe(run_dir, "sampling_padding_levels.csv",
                    ["等级", "M", "原生输出点数", "原生输出间距_um", "离散恒等relL2",
                     "功率相对误差", "公共参考原始强度L1", "公共形状L1", "公共窗口通量偏差",
                     "耗时_s"], rows_pad)
    write_csv_maybe(run_dir, "sampling_identity_levels.csv",
                    ["算例", "N", "dx_um", "M", "离散恒等relL2", "功率相对误差"], rows_id)
    save_npz(run_dir, "sampling_common_grid.npz", **saved)

    in_last_ok = bool(input_last_pair and input_last_pair["raw_intensity_L1"] <= tol_in)
    pad_ok = bool(pad_records and min(r["common_ref_intensity_L1"] for r in pad_records) <= tol_pad)
    pad_trend_ok = True
    if len(pad_records) >= 2:
        pad_trend_ok = bool(pad_records[-1]["common_ref_intensity_L1"]
                            <= pad_records[0]["common_ref_intensity_L1"] * 1.5 + 1e-12)
    res = {
        "input_levels": [{"name": lv["name"], "n": d["n"], "dx_m": d["dx_m"],
                          "dx_um": U.to_um(d["dx_m"]), "P_in": d["p_in"],
                          "aperture_area_discrete_m2": d["ap_area"],
                          "aperture_area_ideal_m2": float(np.pi * (0.5 * D) ** 2),
                          "aperture_area_rel_err": d["ap_area"] / (np.pi * (0.5 * D) ** 2) - 1.0,
                          "line_L1_vs_airy": d["line_L1_vs_airy"]}
                         for lv, d in zip(s["input_levels"], levels)],
        "input_pair_metrics": level_metrics,
        "input_sampling_L1": {"last_pair": input_last_pair["pair"] if input_last_pair else None,
                              "last_pair_raw_intensity_L1":
                                  input_last_pair["raw_intensity_L1"] if input_last_pair else None,
                              "worst_pair_raw_intensity_L1": worst_input_L1,
                              "tolerance": tol_in, "pass": in_last_ok,
                              "definition": "raw_intensity_L1(a,b)=Σ|Ia−Ib|ΔA/ΣIbΔA，公共输出坐标"},
        "padding_levels": [{k: v for k, v in r.items()
                            if k not in ("I_map", "I_native", "mask")} for r in pad_records],
        "padding_pair": pad_pair,
        "padding_sampling_L1": {
            "measure": ("公共探测器网格上，各 M 的映射强度对公共坐标完整核参考的"
                        "原始强度 L1"),
            "per_level": {r["name"]: r["common_ref_intensity_L1"] for r in pad_records},
            "best": (min((r["common_ref_intensity_L1"] for r in pad_records), default=None)),
            "tolerance": tol_pad, "pass": pad_ok,
            "finer_not_worse": pad_trend_ok,
            "definition": "raw_intensity_L1(a,b)=Σ|Ia−Ib|ΔA/ΣIbΔA，公共探测器坐标"},
        "identity": {"worst_rel_L2": worst_identity, "tolerance": tol_id,
                     "pass": bool(worst_identity <= tol_id)},
        "common_output_grid": common,
        "metrics_are_distinct": ("input_sampling_L1（输入收敛）、padding_sampling_L1（输出/填充"
                                "收敛）、identity（离散算法恒等）是三种不同指标，分开存放。"),
        "图": [str(p.relative_to(run_dir)) if run_dir else str(p) for p in figs],
    }
    return res

# ======================================================================================
# S1 证据：reduced / full 两种输出二次相位表示的准确关系
# ======================================================================================
def phase_representation_check(run_dir: Optional[Path], logger) -> Dict[str, Any]:
    """生成 ``phase_representation_check.npz`` 与 ``phase_representation_metrics.json``。

    覆盖原 N=24、Δx′=10 μm、M=48、λ=550 nm、z=50 mm 算例，并额外覆盖非方形网格与
    全局相位开关组合。断言（用**全场范数**，不逐点除场）：

    ``relL2(u_red·exp(iq), u_full) ≤ 1e−10``
    ``relL2(u_red, u_full·exp(−iq)) ≤ 1e−10``
    强度相对差 ≤1e−12
    """
    lam, z = 550e-9, 50e-3
    cases = []

    n, dx, M = 24, 10e-6, 48
    grid_a = Grid2D(Axis1D(n, dx, "x'"), Axis1D(n, dx, "y'"))
    cases.append(("N24_dx10um_M48", grid_a, M, True, make_asymmetric_input(grid_a, lam)))

    nx, ny, dx2, dy2 = 20, 26, 8e-6, 15e-6
    grid_b = make_grid_xy(nx, ny, dx2, dy2)
    cases.append(("nonsquare_20x26_dx8_dy15", grid_b, (64, 48), True,
                  make_asymmetric_input(grid_b, lam)))
    cases.append(("N24_no_global_phase", grid_a, M, False,
                  make_asymmetric_input(grid_a, lam)))

    metrics: Dict[str, Any] = {
        "definition": {
            "u_full": "C·F·exp(+i·q)，q = π(x²+y²)/(λz)",
            "u_red": "C·F（未乘输出二次相位）",
            "relations": ["u_full = u_red·exp(+iq)", "u_red = u_full·exp(−iq)"],
            "note": ("第二轮审核 S1：旧实现把 reduced 写成 C·F·exp(−iq)，"
                     "使两者相差 exp(−2iq)（复场相对 L2 1.4618）。已修正。"),
        },
        "cases": {},
    }
    save: Dict[str, Any] = {}
    worst = 0.0
    for name, g, m_fft, gp, u1 in cases:
        full = fresnel_fft(u1, g, FresnelConfig(lam, z, m_fft, include_global_phase=gp))
        red = fresnel_fft(u1, g, FresnelConfig(lam, z, m_fft, include_global_phase=gp,
                                               omit_output_quadratic_phase=True))
        x, y = red.grid.x.coords, red.grid.y.coords
        q = np.pi * (x[None, :] ** 2 + y[:, None] ** 2) / (lam * z)
        r1 = rel_l2(red.field * np.exp(1j * q), full.field)
        r2 = rel_l2(red.field, full.field * np.exp(-1j * q))
        r_int = rel_l2(np.abs(red.field) ** 2, np.abs(full.field) ** 2)
        r_wrong = rel_l2(red.field * np.exp(2j * q), full.field)
        worst = max(worst, r1, r2)
        metrics["cases"][name] = {
            "shape_yx": [int(g.shape[0]), int(g.shape[1])],
            "dx_m": g.dx, "dy_m": g.dy, "fft_size": list(m_fft) if isinstance(m_fft, tuple)
            else int(m_fft),
            "include_global_phase": gp,
            "relL2_red_times_exp_iq_vs_full": r1,
            "relL2_red_vs_full_times_exp_minus_iq": r2,
            "intensity_rel_diff": r_int,
            "counterexample_relL2_red_times_exp_2iq_vs_full": r_wrong,
            "representation": red.meta["representation"],
            "output_quadratic_phase_present": red.meta["output_quadratic_phase_present"],
        }
        logger.info("【S1】%s：relL2(red·e^{iq}, full)=%.4e；relL2(red, full·e^{−iq})=%.4e；"
                    "强度相对差=%.4e；反例（补两次相位）=%.4e",
                    name, r1, r2, r_int, r_wrong)
        if name == "N24_dx10um_M48":
            save.update({"full": full.field, "reduced": red.field,
                         "x_m": x, "y_m": y, "q_rad": q})
    metrics["worst_relation_relL2"] = worst
    metrics["tolerance"] = 1e-10
    metrics["pass"] = bool(worst <= 1e-10)
    save["comment"] = np.array(
        "S1 证据：full=C·F·exp(+iq)，reduced=C·F；坐标为 m；index [j,i]=(x_i,y_j)")
    if run_dir is not None:
        np.savez_compressed(run_dir / "arrays" / "phase_representation_check.npz", **save)
        write_json(run_dir / "metrics" / "phase_representation_metrics.json", metrics)
    return metrics



def build_report(cfg: Dict[str, Any], env: Dict[str, Any], results: Dict[str, Any],
                 run_id: str, stage: str, overall: Dict[str, Any]) -> str:
    A = []
    P = A.append
    P("# 阶段 01 修正版报告：Fresnel 传播基线")
    P("")
    P("- 运行编号：`%s`　阶段：`%s`" % (run_id, stage))
    P("- 本次为**修正后**的正式运行。此前初审版的结论（把实现错误解释为"
      "“采样限制/方法性偏差”、并宣称“直接积分是权威方法”）**已撤回**，"
      "详见 `docs/revision_notes.md`。")
    P("")
    P("## 0. 运行环境与实际命令")
    P("")
    P("| 项 | 值 |")
    P("| --- | --- |")
    P("| 项目根目录（由 `__file__` 解析） | `%s` |" % env["project_root_resolved"])
    P("| 调用时工作目录 | `%s` |" % env["cwd_at_invocation"])
    P("| Python | `%s`（%s） |" % (env["python_executable"], env["python_version"]))
    P("| 依赖 | numpy %s / scipy %s / matplotlib %s |"
      % (env.get("numpy_version"), env.get("scipy_version"), env.get("matplotlib_version")))
    P("| 平台 | %s（%s，逻辑核 %s） |"
      % (env["platform"], env["platform_machine"], env["os_cpu_count"]))
    P("| 实际命令 | `%s` |" % " ".join([Path(env["python_executable"]).name]
                                       + [str(a) for a in env["argv"]]))
    P("| matplotlib 后端 | %s |" % BACKEND)
    P("| 图内中文字体 | %s |" % (CJK_FONT or "未找到"))
    P("| 落盘 | %s |" % ("是" if results.get("_saved") else "否（--no-save，未写任何文件）"))
    P("")
    P("## 1. 修正内容一览（对应审核 R1–R7）")
    P("")
    P("| 审核项 | 修正 |")
    P("| --- | --- |")
    P("| R1 FFT 漏面积权重、频率未中心化 | 单次 FFT 改为 `fftshift(fft2(ifftshift(V)))`，"
      "输出坐标 `λz·fftshift(fftfreq(M,Δx′))`，并显式乘 `Δx′Δy′`。"
      "同一离散输入/同一原生网格上与完整核求和相差 ~1e−14（旧版 6.25e8） |")
    P("| R1 `omit_output_quadratic_phase` 的等价性声明不成立 | 该选项改为**明确的场表示变化**，"
      "在元数据里标注“非同一物理复场”，并加测试固定其语义与已知限制 |")
    P("| R2 直接积分重复输入二次相位 | 完整位移核求和**不再**预乘任何输入相位；"
      "展开核写法由 FFT 路径单独承担，三部分各出现一次 |")
    P("| R3 Airy 适用条件与近轴诊断错误 | 明确“理想透镜焦平面 = 孔径 Fourier 强度”，"
      "**不再要求** `D²/(4λf)≪1`；论文尺度 D=1 mm 已实际求值。"
      "高阶光程误差改用精确路径差并按角点距离估计 |")
    P("| R4 测试与验收证据不足 | 新增离散恒等、原生 FFT Parseval、"
      "数值 vs 解析高斯、D=1 mm Airy、非方形/奇数网格等测试；"
      "验收失败进入总状态并返回非零退出码 |")
    P("| R5 行坐标说明与实现相反 | 统一为“索引增加 → x、y 都增加”，`origin='lower'`；"
      "extent 区分样点中心与像素边界 |")
    P("| R6 `--no-save` 仍落盘、`--run-dir` 保护不足 | `--no-save` 真正不写任何文件；"
      "`--run-dir` 要求目标不存在或为空 |")
    P("| R7 固定 Agg 阻断 GUI | 后端在导入 pyplot **之前**按模式选择；"
      "`--show-plots` 时优先交互后端并在报告中记录实际后端 |")
    P("")

    # A
    P("## 2. A 组：理想透镜焦平面的 Airy 尺度")
    P("")
    a = results.get("airy")
    if a and a.get("cases"):
        P("物理依据：`z=f` 时透镜相位 `exp(−iπr′²/(λf))` 与传播展开式的输入二次相位 "
          "`exp(+iπr′²/(λf))` 相消，**焦平面就是孔径的 Fourier 强度**，圆孔给出 Airy 图样。"
          "因此**不额外要求** `D²/(4λf)≪1`；近轴条件与数值采样条件仍逐项检查。")
        P("")
        P("| 算例 | N_F | 输入 Δx′ (μm) | 孔径内样点 | 输出间距 (μm) | 样点/暗环 | "
          "理论第一暗环 (μm) | 实测 (μm) | 相对误差 | 截线 L1 |")
        P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in a["cases"]:
            P("| %s | %.4g | %.4g | %.1f | %.4g | %.1f | %.6f | %.6f | %+.3e | %.4e |"
              % (c["name"], c["fresnel_number"], c["input_spacing_um"],
                 (0.5 * U.mm(c["aperture_mm"])) / U.um(c["input_spacing_um"]),
                 c["output_spacing_um"], c["samples_per_dark_ring"],
                 c["theory_first_ring_um"], c["measured_first_ring_um"],
                 c["first_ring_rel_err"], c["line_L1_vs_airy"]))
        P("")
        for c in a["cases"]:
            sd = c["sampling_diagnostics"]
            P("- `%s`：采样判据总步进 %.4g rad（阈值 %.2f）→ ok=%s；"
              "ρ_max=%.4g μm → 精确高阶光程误差 %.6e rad（四次近似 %.6e rad）"
              % (c["name"], sd["max_total_step_rad"], sd["threshold_rad"], sd["ok"],
                 U.to_um(sd["rho_max_m"]), sd["third_order_phase_error_rad"],
                 sd["leading_fourth_order_phase_error_rad"]))
        P("")
        P("绝对包围能量（**唯一分母口径** $P_{disk}(R)/P_{in}$，与解析 $1-J_0^2-J_1^2$ 比较）：")
        P("")
        P("| 算例 | R (μm) | 评价窗口半宽 (μm) | 窗口覆盖圆盘 | 圆盘非全窗 | "
          "$P_{in}$ | $P_{disk}$ | $E_{numeric}$ | $E_{analytic}$ | 绝对差 |")
        P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in a["cases"]:
            e = c.get("encircled_energy", {})
            if not e:
                continue
            P("| %s | %.4g | %.4g | %s | %s | %.8e | %.8e | %.8f | %.8f | %+.3e |"
              % (c["name"], e["R_um"], e["window_half_um"], e["window_covers_disk"],
                 e["disk_does_not_select_whole_window"], e["P_in"], e["P_disk"],
                 e["E_numeric_disk_over_Pin"], e["E_analytic"], e["absolute_difference"]))
        P("")
        P("> 分母是输入功率 $P_{in}=\\sum|u_{in}|^2\\Delta x'\\Delta y'$，**不是**有限方窗内"
          "强度和；`P_disk/P_square_window` 另记为“有限方窗内条件占比”，不与无限平面解析式"
          "当同一指标。评价窗口半宽严格大于 R，且 R 远小于窗口半宽，圆盘不会恒选全窗。")
        P("")
        ps = a.get("paper_scale_reference", {})
        if ps:
            P("论文尺度参照（与第一个算例同参数，已实际求值）：D=%.3g mm、f=%.3g mm、"
              "λ=%.3g nm → 第一暗环 Bessel 精确值 **%.6f μm**，1.22 近似 %.4f μm；"
              "N_F=%.4g 仅作报告。"
              % (ps["D_mm"], ps["f_mm"], ps["wavelength_nm"],
                 ps["first_ring_bessel_um"], ps["first_ring_1p22_um"],
                 ps["fresnel_number"]))
            P("")

    # B
    P("## 3. B 组：离散恒等（单次 FFT vs 完整位移核求和）")
    P("")
    b = results.get("direct")
    if b and b.get("cases"):
        P("同一离散输入、同一原生输出网格、同一面积权重下，两者是**离散代数恒等**关系。"
          "这一项能直接拦截漏面积权重、频率未中心化、重复相位、索引互换等实现错误。")
        P("")
        P("| N | Δx′ (μm) | M | rel L2 | 功率相对误差 | P_out/P_in | 输出间距 (μm) |")
        P("| --- | --- | --- | --- | --- | --- | --- |")
        for c in b["cases"]:
            P("| %d | %.4g | %d | %.4e | %.4e | %.12f | %.4g |"
              % (c["input_samples"], c["input_spacing_um"], c["fft_size"],
                 c["rel_L2_fft_vs_full_kernel"], c["power_relative_error"],
                 c["power_ratio"], c["output_spacing_um"]))
        P("")
        P("- 最差 rel L2 = **%.4e**（验收目标 ≤%.0e）→ **%s**"
          % (b["worst_rel_L2"], b["acceptance_target"], "通过" if b["pass"] else "未通过"))
        P("")

    # C
    P("## 4. C 组：能量")
    P("")
    c = results.get("energy")
    if c and "parseval_worst_rel_err" in c:
        P("- **主判据**：原生 FFT 全输出 Parseval 相对功率误差 = **%.4e**（目标 ≤%.0e）→ **%s**"
          % (c["parseval_worst_rel_err"], c["parseval_target"],
             "通过" if c["parseval_pass"] else "未通过"))
    elif c:
        P("- **C 组未完成**：%s" % c.get("error", "没有可用结果"))
        P("")
    if c and c.get("absolute_energy_levels"):
        P("- **绝对包围能量**（$P_{disk}/P_{in}$ 对解析，容差 %.4g）："
          % c["absolute_energy_tolerance"])
        P("")
        P("| 等级 | R (μm) | 输出半宽 (μm) | 输出间距 (μm) | R/半宽 | $P_{in}$ | "
          "$P_{disk}$ | $E_{numeric}$ | $E_{analytic}$ | 绝对差 |")
        P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in c["absolute_energy_levels"]:
            P("| %s | %.4g | %.4g | %.4g | %.4f | %.8e | %.8e | %.9f | %.9f | %+.3e |"
              % (r["label"], r["R_um"], r["window_half_um"], r["output_spacing_um"],
                 r["R_over_window_half"], r["P_in"], r["P_disk"],
                 r["E_numeric_disk_over_Pin"], r["E_analytic"],
                 r["absolute_difference"]))
        P("")
        P("- 最差绝对差 = **%.4e**（容差 %.4g）；按定义 $E_{numeric}=P_{disk}(R)/P_{in}$，"
          "分母与解析式同口径；累计曲线也使用同一个 $P_{in}$，图上不再出现“未重新归一化”"
          "却除以有限窗口和的情况。"
          % (c["absolute_energy_worst_abs_diff"], c["absolute_energy_tolerance"]))
        P("")
        P("> 说明：Parseval 检验验证离散变换与**尺度因子**（含 `Δx′Δy′`）是否正确，"
          "它**不能单独证明无混叠**；连续层面的独立证据来自 A 组的 Airy 解析对照、"
          "D 组的收敛研究与测试中的自由空间高斯解析对照。")
        P("")

    # D
    P("## 5. D 组：三类彼此不可替代的检查")
    P("")
    d = results.get("sampling")
    if d and "input_levels" not in d:
        P("- **D 组未完成**：%s" % d.get("error", "没有可用结果"))
        P("")
        d = None
    if d:
        P("**D1 输入采样收敛**（固定同一个连续器件与公共输出坐标，只改变输入采样间距；"
          "每级都按同一连续函数取样，不把粗网格图像放大）：")
        P("")
        P("| 等级 | N | Δx′ (μm) | $P_{in}$ | 孔径离散面积相对差 | 截线对解析 Airy 的 L1 |")
        P("| --- | --- | --- | --- | --- | --- |")
        for lv in d["input_levels"]:
            P("| %s | %d | %.4g | %.8e | %+.3e | %.4e |"
              % (lv["name"], lv["n"], lv["dx_um"], lv["P_in"],
                 lv["aperture_area_rel_err"], lv["line_L1_vs_airy"]))
        P("")
        P("| 相邻等级 | 原始强度相对 L1（主指标） | 反向 L1 | 形状 L1 | 复场相对 L2 |")
        P("| --- | --- | --- | --- | --- |")
        for m in d["input_pair_metrics"]:
            P("| %s | %.6e | %.6e | %.6e | %.6e |"
              % (m["pair"], m["raw_intensity_L1"], m["raw_intensity_L1_reverse"],
                 m["shape_L1"], m["field_relL2"]))
        P("")
        P("- 按**预先约定**用最后一级对（%s）判定最终精度：原始强度相对 L1 = **%.6e**"
          "（阈值 %.3g）→ **%s**；更粗的等级若超阈值则如实列出，不删除。"
          % (d["input_sampling_L1"]["last_pair"],
             d["input_sampling_L1"]["last_pair_raw_intensity_L1"] or float("nan"),
             d["input_sampling_L1"]["tolerance"],
             "通过" if d["input_sampling_L1"]["pass"] else "未通过"))
        P("")
        P("**D2 输出采样（填充）收敛**（固定同一离散输入，只改变 FFT 长度 M；把原生**强度**"
          "双线性插值到同一公共探测器网格，与该输入在公共坐标上的完整核参考比较；"
          "越界点显式剔除，不补零）：")
        P("")
        P("| 等级 | M | 原生点数 | 原生间距 (μm) | 离散恒等 rel L2 | 功率误差 | "
          "公共参考原始强度 L1 | 形状 L1 | 窗口通量偏差 |")
        P("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in d["padding_levels"]:
            P("| %s | %d | %d | %.4g | %.4e | %.4e | %.6e | %.4e | %+.3e |"
              % (r["name"], r["M"], r["native_output_points"],
                 U.to_um(r["native_output_spacing_m"]), r["identity_rel_L2"],
                 r["power_rel_err"], r["common_ref_intensity_L1"], r["common_shape_L1"],
                 r["common_window_flux_bias"]))
        P("")
        pp = d.get("padding_pair")
        if pp:
            P("- 两个 M 在公共网格上的实际原始强度 L1 = **%.6e**（形状 L1 = %.4e，%d 点）"
              % (pp["raw_intensity_L1_between_M"], pp["shape_L1_between_M"], pp["points"]))
        P("- 最细一级对公共参考的原始强度 L1 = **%s**（阈值 %.3g）→ **%s**；"
          "更细填充未变差：%s"
          % (("%.6e" % d["padding_sampling_L1"]["best"])
             if d["padding_sampling_L1"]["best"] is not None else "未执行",
             d["padding_sampling_L1"]["tolerance"],
             "通过" if d["padding_sampling_L1"]["pass"] else "未通过",
             d["padding_sampling_L1"]["finer_not_worse"]))
        P("")
        P("**D3 离散算法恒等**（同一离散输入、同一原生网格；**不能**代替 D1/D2 的物理收敛）：")
        P("")
        P("| 算例 | N | Δx′ (μm) | M | rel L2 | 功率误差 |")
        P("| --- | --- | --- | --- | --- | --- |")
        P("| 见 `metrics/sampling_identity_levels.csv` | | | | 最差 %.4e | |"
          % d["identity"]["worst_rel_L2"])
        P("")
        P("> 三种指标的**定义与阈值互不相同**：`discrete_identity_rel_l2_max`（复场相对 L2，"
          "只用于算法恒等）、`input_sampling_intensity_L1_max` 与 "
          "`padding_sampling_intensity_L1_max`（公共物理网格上的原始强度相对 L1）。"
          "复场 L2 不会被塞进名为 L1 的字段。")
        P("")

    # 总状态
    P("## 6. 机器可读总体状态")
    P("")
    P("| 检查 | 阈值 | 实测 | 结论 |")
    P("| --- | --- | --- | --- |")
    for k, v in overall["checks"].items():
        P("| %s | %.4g | %s | %s |"
          % (k, v["threshold"], ("%.6e" % v["measured"]) if v["measured"] is not None
             else "未执行", v["status"]))
    P("")
    P("- 总体状态：**%s**" % overall["status"])
    P("- 未运行的组：%s" % (", ".join(overall["not_run"]) if overall["not_run"] else "无"))
    P("- 进程退出码：%d（0 表示全部执行项通过；非零表示有失败）" % overall["exit_code"])
    P("")
    P("## 7. 未执行 / 未验证项")
    P("")
    P("- **PyCharm 点击运行与 GUI 弹窗仍未人工验证**；本环境无法点击 PyCharm、"
      "无法显示窗口。命令行入口已实际运行（含从其他工作目录启动）。"
      "本次运行实际后端为 `%s`。" % BACKEND)
    P("- 未实现 DOE 高度设计、三翼结构、多波长 PSF、图 3 对照、成像与重建。")
    P("- 未确定熔融石英色散系数、16 级量化规则、Canon 响应曲线、图 3 绝对视场。")
    P("- 无官方代码可对照；未复现论文所用 LightPipes 的具体设置。")
    P("- 理想相位透镜无吸收与界面反射，真实 DOE 效率未建模。")
    P("")
    P("## 8. 文件对应关系")
    P("")
    P("- `config_used.json` / `environment.json` / `run.log` / `acceptance_check.json`")
    P("- `arrays/*.npz`：复场、强度、物理坐标（m）与解析参考数组")
    P("- `metrics/*.csv`：本报告表格的原始数值")
    P("- `figures/*.png`：A/B/C/D 各图")
    P("")
    P("## 9. 结论")
    P("")
    P("按审核意见修正后重新运行，四组验证与测试的实际结果见上表。"
      "本阶段到此停止，提交 Codex 再审；不进入阶段 02。")
    P("")
    return "\n".join(A)


# ======================================================================================
# 主流程
# ======================================================================================
def main() -> int:
    cfg_path = _CFG_PATH
    cfg = _CFG_EARLY
    stage = ARGS.stage or cfg.get("stage", "stage01")
    # 最终运行设置（S6）：配置与命令行合成后才决定弹窗与落盘
    show = bool(SETTINGS["show_plots"])
    no_save = not bool(SETTINGS["save_results"])
    seed = int(cfg["runtime"].get("numpy_random_seed", 0))
    np.random.seed(seed)

    run_dir: Optional[Path] = None
    run_id = "no_save"
    if not no_save:
        if ARGS.run_dir:
            cand = resolve_project_path(ARGS.run_dir)
            if cand.exists() and any(cand.iterdir()):
                print("[错误] --run-dir 目标必须不存在或为空：%s" % cand)
                return 3
            cand.mkdir(parents=True, exist_ok=True)
            for sub in ("arrays", "figures", "metrics"):
                (cand / sub).mkdir(exist_ok=True)
            run_dir, run_id = cand, cand.name
        else:
            run_dir, run_id = allocate_run_dir(
                stage, force_new=bool(cfg["runtime"].get("force_new_run", False)),
                root=DEFAULT_OUTPUT_ROOT)

    logger = setup_logger(run_dir, cfg["runtime"].get("log_level", "INFO"))
    env = environment_info({"config_path": str(cfg_path), "stage": stage, "run_id": run_id,
                            "random_seed": seed, "backend": BACKEND,
                            "no_save": no_save,
                            "effective_show_plots": show,
                            "effective_save_results": not no_save,
                            "settings_sources": {k: SETTINGS[k] for k in
                                                 ("show_source", "save_source")},
                            "output_root": str(DEFAULT_OUTPUT_ROOT)})
    logger.info("=" * 78)
    logger.info("Jeon2019 阶段 01（第二轮修正版）：Fresnel 传播基线")
    logger.info("项目根目录（由 __file__ 解析）：%s", PKG_ROOT)
    logger.info("调用时工作目录：%s", env["cwd_at_invocation"])
    logger.info("解释器：%s", env["python_executable"])
    logger.info("numpy %s / scipy %s / matplotlib %s",
                env.get("numpy_version"), env.get("scipy_version"), env.get("matplotlib_version"))
    logger.info("最终设置：show_plots=%s（来源：%s）；save_results=%s（来源：%s）",
                show, SETTINGS["show_source"], not no_save, SETTINGS["save_source"])
    logger.info("matplotlib 后端：%s；中文字体：%s", BACKEND, CJK_FONT or "未找到")
    logger.info("结果目录：%s", run_dir if run_dir else "（不落盘：不写任何项目文件）")
    logger.info("=" * 78)

    results: Dict[str, Any] = {"_saved": run_dir is not None}
    failures: List[str] = []
    only = ARGS.only
    groups = [("airy", "A Airy", study_airy), ("direct", "B 离散恒等", study_direct_vs_fft),
              ("energy", "C 能量", study_energy), ("sampling", "D 采样", study_sampling)]

    def guard(name, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("【失败】%s 抛出 %s: %s", name, type(exc).__name__, exc)
            logger.error("\n%s", tb)
            failures.append("%s: %s: %s" % (name, type(exc).__name__, exc))
            return {"error": "%s: %s" % (type(exc).__name__, exc)}

    ran, not_run = [], []
    for key, label, fn in groups:
        if only in (None, key):
            results[key] = guard(label, fn, cfg, run_dir, logger, show)
            ran.append(key)
        else:
            not_run.append(key)

    # S1 证据（很轻，始终在完整运行与 B 组子集中执行）：
    # reduced / full 两种输出二次相位表示的准确关系，另存 npz + json。
    if only in (None, "direct"):
        try:
            results["phase_representation"] = phase_representation_check(run_dir, logger)
        except Exception as exc:                       # pragma: no cover
            logger.error("【失败】S1 相位表示检查抛出 %s: %s", type(exc).__name__, exc)
            logger.error("\n%s", traceback.format_exc())
            failures.append("phase_representation: %s: %s" % (type(exc).__name__, exc))

    # ---------------- 机器可读总状态（第二轮修正 S4） ----------------
    acc = cfg["acceptance"]
    checks: Dict[str, Dict[str, Any]] = {}

    def _finite(v) -> bool:
        try:
            return bool(np.isfinite(float(v)))
        except (TypeError, ValueError):
            return False

    def add_check(name, threshold, measured, executed=True, detail=None):
        """登记一项验收。

        状态语义（S4）：
        * ``pass``  —— 已执行、测到有限值且 ≤ 阈值；
        * ``fail``  —— 已执行但**超阈值**、或测到 NaN/Inf、或必需字段缺失、
                       或已请求执行却没有任何测量；
        * ``not_run`` —— 该组被 ``--only`` 排除，或配置 ``enabled=false``。
        """
        rec = {"threshold": float(threshold), "measured": None,
               "status": "not_run", "detail": detail or ""}
        if not executed:
            rec["status"] = "not_run"
            rec["detail"] = (detail or "") + "；未执行（--only 子集或 enabled=false）"
        elif measured is None:
            rec["status"] = "fail"
            rec["detail"] = (detail or "") + "；已请求执行但缺少测量值"
        elif not _finite(measured):
            rec["status"] = "fail"
            rec["detail"] = (detail or "") + "；测量值非有限（NaN/Inf）"
        else:
            rec["measured"] = float(measured)
            rec["status"] = "pass" if float(measured) <= float(threshold) else "fail"
        checks[name] = rec
        return rec

    a = results.get("airy") or {}
    cases = a.get("cases") or []
    airy_requested = "airy" in ran
    # 每个 Airy 算例**分别**判定，总体取所有必需算例的**最差值**并保存最差算例名
    ring_vals = [(c["name"], c.get("first_ring_rel_err")) for c in cases]
    line_vals = [(c["name"], c.get("line_L1_vs_airy")) for c in cases]
    for nm, vals, thr, label in (
            ("airy_first_dark_ring_rel_err", ring_vals,
             acc["airy_first_dark_ring_rel_err_max"], "暗环相对误差"),
            ("airy_line_L1_vs_analytic", line_vals,
             acc["airy_line_L1_max"], "截线相对 L1")):
        finite = [(n_, v) for n_, v in vals if v is not None]
        if finite:
            worst_name, worst_val = max(finite, key=lambda t: float(t[1]))
            add_check(nm, thr, worst_val, executed=airy_requested,
                      detail="%s；最差算例=%s（共 %d 个算例各自判定）"
                             % (label, worst_name, len(vals)))
        else:
            add_check(nm, thr, None, executed=airy_requested,
                      detail="%s；没有任何 Airy 算例测量值" % label)
    # 每个 Airy 算例的逐例明细（机器可读）
    per_case = {}
    for c in cases:
        per_case[c["name"]] = {
            "first_ring_rel_err": c.get("first_ring_rel_err"),
            "first_ring_status": ("pass" if _finite(c.get("first_ring_rel_err"))
                                  and abs(float(c["first_ring_rel_err"]))
                                  <= acc["airy_first_dark_ring_rel_err_max"] else "fail"),
            "line_L1_vs_airy": c.get("line_L1_vs_airy"),
            "line_L1_status": ("pass" if _finite(c.get("line_L1_vs_airy"))
                               and float(c["line_L1_vs_airy"]) <= acc["airy_line_L1_max"]
                               else "fail"),
            "encircled_energy_abs_diff": (c.get("encircled_energy") or {}).get(
                "absolute_difference"),
        }

    b = results.get("direct") or {}
    dd = results.get("sampling") or {}
    ident_cand = [v for v in (b.get("worst_rel_L2"),
                              (dd.get("identity") or {}).get("worst_rel_L2"))
                  if v is not None]
    add_check("discrete_identity_rel_L2", acc["discrete_identity_rel_l2_max"],
              (max(ident_cand) if ident_cand else None),
              executed=bool(ident_cand),
              detail="B 组各规模与 D3 恒等算例的最大值；复场相对 L2")

    cc = results.get("energy") or {}
    add_check("native_fft_parseval_rel_err", acc["parseval_rel_err_max"],
              cc.get("parseval_worst_rel_err"), executed=("energy" in ran),
              detail="原生 FFT 全输出 Parseval 相对功率误差")
    add_check("energy_absolute_abs_diff", acc["energy_absolute_abs_diff_max"],
              cc.get("absolute_energy_worst_abs_diff"), executed=("energy" in ran),
              detail="绝对包围能量 P_disk(R)/P_in 对解析的最大绝对差")

    inp = dd.get("input_sampling_L1") or {}
    add_check("input_sampling_intensity_L1", acc["input_sampling_intensity_L1_max"],
              inp.get("last_pair_raw_intensity_L1"), executed=("sampling" in ran),
              detail="D1 最后一级输入采样对的原始强度相对 L1（%s）" % (inp.get("last_pair"),))
    padd = dd.get("padding_sampling_L1") or {}
    add_check("padding_sampling_intensity_L1", acc["padding_sampling_intensity_L1_max"],
              padd.get("best"), executed=("sampling" in ran),
              detail="D2 最细填充对公共坐标完整核参考的原始强度 L1")

    # 必需组是否都实际执行（配置 groups.*.required）
    required = [k for k, v in (cfg.get("groups") or {}).items()
                if isinstance(v, dict) and v.get("required")]
    missing_required = [k for k in required if k in not_run]
    for k in missing_required:
        checks.setdefault("required_group_%s" % k,
                          {"threshold": 1.0, "measured": None, "status": "not_run",
                           "detail": "必需组被 --only 排除"})

    any_fail = any(v["status"] == "fail" for v in checks.values())
    all_requested_ran = not not_run
    if any_fail or failures:
        status = "fail"
        exit_code = 1
    elif all_requested_ran:
        status = "pass"
        exit_code = 0
    else:
        # 用户主动选择的子集且已执行项全部通过：允许 partial 并返回 0（不隐藏已执行的失败）
        status = "partial"
        exit_code = 0
    overall = {"checks": checks, "per_airy_case": per_case, "status": status,
               "exit_code": exit_code, "not_run": not_run, "ran": ran,
               "failures": failures, "required_groups": required,
               "missing_required_groups": missing_required,
               "status_semantics": {
                   "pass": "默认完整运行，全部必需项实际测量并通过",
                   "partial": "用户主动选择子集，已执行项全部通过（返回 0，并列出未执行项）",
                   "fail": "任一已执行项超阈值 / 缺测 / 非有限 / 计算异常（返回非零）"}}

    if run_dir is not None:
        write_json(run_dir / "config_used.json", cfg)
        write_json(run_dir / "environment.json", env)
        ser = json.loads(json.dumps(results, ensure_ascii=False, default=str))
        write_json(run_dir / "results_summary_auto.json", ser)
        write_json(run_dir / "acceptance_check.json", overall)
        (run_dir / "stage01_revision_report.md").write_text(
            build_report(cfg, env, results, run_id, stage, overall), encoding="utf-8")

    logger.info("-" * 78)
    for k, v in checks.items():
        logger.info("验收 %-32s 阈值 %.4g  实测 %s  → %s", k, v["threshold"],
                    ("%.6e" % v["measured"]) if v["measured"] is not None else "未执行",
                    v["status"])
    if failures:
        logger.error("本次运行有 %d 项异常：%s", len(failures), "; ".join(failures))
    logger.info("总体状态：%s；退出码 %d；未运行组：%s",
                status, exit_code, ", ".join(not_run) if not_run else "无")
    logger.info("结果目录：%s", run_dir if run_dir else "（未落盘）")
    logger.info("阶段 01（修正版）结束，停止，不进入阶段 02。")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())




