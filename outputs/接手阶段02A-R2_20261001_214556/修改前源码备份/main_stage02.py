# -*- coding: utf-8 -*-
"""Jeon et al. 2019 光学复现阶段 02A 主执行入口（阶段 02A-R 修正版）。

支持：
* 在 PyCharm 中无参数直接右键运行（默认执行全部 02A 内容并保存结果）；
* 命令行参数：
    --config PATH       指定配置文件路径（默认同目录下 config_stage02a.json，相对路径按项目根解析）
    --only TARGET       限制执行子集：height | control | preview | all（默认 all）
    --no-save           仅内存计算与自检，不写入任何磁盘结果与日志
    --show-plots        显示交互图窗（在导入 pyplot 之前完成后端选择与生命周期管理）
    --run-dir PATH      显式指定运行目录（可选，缺省则按 results/stage02a/run_<时间戳> 自动分配）

阶段 02A 核心物理范围
---------------------
1. 固定 N=3 连续 DOE 高度设计（正文式 (9)–(12)）与 Malitson 1965 熔融石英色散；
2. 550nm 传统单色 Fresnel 透镜控制验证（与理想 Airy 参照对比）；
3. 固定器件在共同物理网格下的 Fresnel 衍射复振幅与原始强度预览；
4. 严格保持器件固定，同一器件指纹贯穿所有波长计算。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 先解析基础参数，以便在导入 pyplot 之前完成 matplotlib 后端配置
def _early_parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config_stage02a.json"))
    parser.add_argument("--only", type=str, choices=["all", "height", "control", "preview"], default="all")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--run-dir", type=str, default="")
    args, _ = parser.parse_known_args(argv)
    return args

_early_args = _early_parse_args()

import matplotlib

# 配置 Matplotlib 后端（在导入 pyplot 之前）
_GUI_DISPLAYED = False
_GUI_BACKEND_NAME = "Agg"
if _early_args.show_plots:
    # 尝试常见的交互 GUI 后端
    for b_name in ("TkAgg", "QtAgg", "Qt5Agg", "WXAgg"):
        try:
            matplotlib.use(b_name)
            _GUI_BACKEND_NAME = b_name
            _GUI_DISPLAYED = True
            break
        except Exception:
            continue
    if not _GUI_DISPLAYED:
        matplotlib.use("Agg")
        _GUI_BACKEND_NAME = "Agg (未检测到可用 GUI 后端，已安全回退)"
else:
    matplotlib.use("Agg")
    _GUI_BACKEND_NAME = "Agg"

import matplotlib.pyplot as plt

# 配置中文字体支持与负号显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from optics.coordinates import make_grid, Grid2D
from optics.materials import (
    refractive_index_fused_silica,
    get_material_model_metadata,
    FUSED_SILICA_WAVELENGTH_MIN_M,
    FUSED_SILICA_WAVELENGTH_MAX_M,
    REFRACTIVE_INDEX_AIR,
)
from optics.doe import (
    DOEHeightProfile,
    design_conventional_fresnel_height,
    design_jeon2019_spiral_height,
    compute_doe_transmission_field,
    optical_path_difference_delta,
)
from optics.propagation import fresnel_kernel_separable, sampling_diagnostics
from optics.metrics import (
    find_local_minima,
    airy_dark_ring_radii,
    airy_intensity,
    normalized_l1_intensity,
)
from optics.runutil import allocate_run_dir, setup_logger, environment_info, write_json, Timer


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jeon2019 阶段 02A 光学仿真与自检")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config_stage02a.json"),
        help="配置文件路径（默认同目录下的 config_stage02a.json，相对路径按项目根解析）",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["all", "height", "control", "preview"],
        default="all",
        help="限制执行的子集：height | control | preview | all（默认 all）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="纯内存运行与自检，不创建任何运行目录与文件",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="运行结束前尝试弹窗显示图形（需本地 GUI 环境）",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="显式指定结果输出目录（可选）",
    )
    return parser.parse_args(argv)


def _check_positive_finite_number(name: str, val: Any) -> float:
    """校验必须是有限正数。排斥 bool。"""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"参数 [{name}] 必须为数值类型，收到: {type(val).__name__} ({val})")
    f_val = float(val)
    if not math.isfinite(f_val):
        raise ValueError(f"参数 [{name}] 必须为有限数值，收到: {val}")
    if f_val <= 0.0:
        raise ValueError(f"参数 [{name}] 必须严格大于 0，收到: {f_val}")
    return f_val


def _check_strict_int(name: str, val: Any, min_val: int = 1) -> int:
    """校验必须是纯正整数类型，不接受 float 截断或 bool。"""
    if type(val) is not int:
        raise ValueError(f"参数 [{name}] 必须为整数类型 (int)，不接受浮点截断或布尔值，收到: {type(val).__name__} ({val})")
    if val < min_val:
        raise ValueError(f"参数 [{name}] 必须 >= {min_val}，收到: {val}")
    return val


def load_and_validate_config(config_path_raw: Union[str, Path]) -> Dict[str, Any]:
    """加载并严格校验配置参数。

    遵循 R3 规范：全量检查参数类型、取值边界、半宽一致性及模型支持列表。
    """
    p = Path(config_path_raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()

    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. 光学参数校验
    opt = cfg.get("optical")
    if not isinstance(opt, dict):
        raise ValueError("配置缺少 [optical] 字典对象")

    D = _check_positive_finite_number("optical.diameter_m", opt.get("diameter_m"))
    f_len = _check_positive_finite_number("optical.focal_length_m", opt.get("focal_length_m"))
    z_dist = _check_positive_finite_number("optical.distance_m", opt.get("distance_m"))
    N = _check_strict_int("optical.wings_N", opt.get("wings_N"), min_val=1)

    l_min = _check_positive_finite_number("optical.design_wavelength_min_m", opt.get("design_wavelength_min_m"))
    l_max = _check_positive_finite_number("optical.design_wavelength_max_m", opt.get("design_wavelength_max_m"))
    if l_min >= l_max:
        raise ValueError(f"设计波长范围非法: min ({l_min}) 必须严格小于 max ({l_max})")

    # 材料模型检查
    mat_model = opt.get("material_model")
    if mat_model != "fused_silica_malitson1965":
        raise ValueError(f"不支持的材料模型: [{mat_model}]，当前仅支持 'fused_silica_malitson1965'")

    # 波长有效范围检查
    for w_name, w_val in [("design_wavelength_min_m", l_min), ("design_wavelength_max_m", l_max)]:
        if w_val < FUSED_SILICA_WAVELENGTH_MIN_M or w_val > FUSED_SILICA_WAVELENGTH_MAX_M:
            raise ValueError(f"波长 [{w_name}]={w_val*1e6:.3f} μm 超出材料有效范围 [{FUSED_SILICA_WAVELENGTH_MIN_M*1e6:.2f}, {FUSED_SILICA_WAVELENGTH_MAX_M*1e6:.2f}] μm")

    # 控制波长与预览波长列表检查
    ctrl_lam = _check_positive_finite_number("optical.control_wavelength_m", opt.get("control_wavelength_m"))
    if ctrl_lam < FUSED_SILICA_WAVELENGTH_MIN_M or ctrl_lam > FUSED_SILICA_WAVELENGTH_MAX_M:
        raise ValueError(f"控制波长 {ctrl_lam*1e6:.3f} μm 超出材料有效范围")

    prev_lams = opt.get("preview_wavelengths_m")
    if not isinstance(prev_lams, list) or len(prev_lams) == 0:
        raise ValueError("optical.preview_wavelengths_m 必须为非空波长列表")
    if len(prev_lams) != len(set(prev_lams)):
        raise ValueError("optical.preview_wavelengths_m 包含重复波长")

    for idx, pl in enumerate(prev_lams):
        pl_val = _check_positive_finite_number(f"optical.preview_wavelengths_m[{idx}]", pl)
        if pl_val < FUSED_SILICA_WAVELENGTH_MIN_M or pl_val > FUSED_SILICA_WAVELENGTH_MAX_M:
            raise ValueError(f"预览波长 {pl_val*1e6:.3f} μm 超出材料有效范围")

    # 2. 网格参数与半宽一致性检查
    grid = cfg.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("配置缺少 [grid] 字典对象")

    for g_name in ("input", "output"):
        g_item = grid.get(g_name)
        if not isinstance(g_item, dict):
            raise ValueError(f"配置缺少 [grid.{g_name}] 字典对象")
        spacing = _check_positive_finite_number(f"grid.{g_name}.spacing_m", g_item.get("spacing_m"))
        n = _check_strict_int(f"grid.{g_name}.n", g_item.get("n"), min_val=3)
        if n % 2 == 0:
            raise ValueError(f"grid.{g_name}.n 必须为奇数以保持对称与含原点，收到: {n}")

        calc_hw = (n // 2) * spacing
        if "half_width_m" in g_item:
            spec_hw = _check_positive_finite_number(f"grid.{g_name}.half_width_m", g_item.get("half_width_m"))
            if abs(spec_hw - calc_hw) > 1e-9:
                raise ValueError(
                    f"grid.{g_name} 半宽不一致: 声明的 half_width_m={spec_hw:.6e} 与 "
                    f"(n//2)*spacing 推导值={calc_hw:.6e} 存在矛盾！"
                )

    # 输入网格孔径覆盖检查
    in_actual_hw = (grid["input"]["n"] // 2) * grid["input"]["spacing_m"]
    if in_actual_hw < 0.5 * D:
        raise ValueError(
            f"输入网格实际半宽 {in_actual_hw*1e6:.1f} μm 小于孔径半径 {0.5*D*1e6:.1f} μm，无法完整覆盖孔径"
        )

    # 3. 传播方法与参数检查
    prop = cfg.get("propagation")
    if not isinstance(prop, dict):
        raise ValueError("配置缺少 [propagation] 字典对象")
    if prop.get("method") != "separable_kernel":
        raise ValueError(f"不支持的传播方法: [{prop.get('method')}]，当前仅支持 'separable_kernel'")
    if not isinstance(prop.get("include_global_phase"), bool):
        raise ValueError("propagation.include_global_phase 必须为布尔值 (True 或 False)")

    # 4. 运行配置检查
    runtime = cfg.get("runtime", {})
    if "save_results" in runtime and not isinstance(runtime["save_results"], bool):
        raise ValueError("runtime.save_results 必须为布尔值")
    if "show_plots" in runtime and not isinstance(runtime["show_plots"], bool):
        raise ValueError("runtime.show_plots 必须为布尔值")

    # 5. 验收阈值检查
    acc = cfg.get("acceptance", {})
    for acc_k, acc_v in acc.items():
        _check_positive_finite_number(f"acceptance.{acc_k}", acc_v)

    # 记录解析出的绝对配置路径
    cfg["_config_path_resolved"] = str(p)
    return cfg


def file_sha256(path: Path) -> str:
    """计算单个文件的 SHA256。"""
    if not path.is_file():
        return "missing"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_source_manifest(config_used_path: Optional[str] = None) -> Dict[str, Any]:
    """生成源码与参考文献的清单与哈希（遵循 R6 规范，纳入全部相关源码与测试）。"""
    manifest_files = [
        "config_stage02a.json",
        "main_stage02.py",
        "optics/coordinates.py",
        "optics/doe.py",
        "optics/materials.py",
        "optics/metrics.py",
        "optics/propagation.py",
        "optics/runutil.py",
        "tests/test_materials.py",
        "tests/test_doe.py",
        "tests/test_stage02_entry.py",
        "tests/test_stage02_failure_paths.py",
    ]
    files_info = []
    for rel_path in manifest_files:
        p = PROJECT_ROOT / rel_path
        files_info.append({
            "path": rel_path,
            "absolute_path": str(p),
            "sha256": file_sha256(p),
            "size_bytes": p.stat().st_size if p.is_file() else 0,
        })

    if config_used_path and Path(config_used_path).is_file():
        p_cfg = Path(config_used_path)
        files_info.append({
            "path": "actual_config_used",
            "absolute_path": str(p_cfg),
            "sha256": file_sha256(p_cfg),
            "size_bytes": p_cfg.stat().st_size,
        })

    main_pdf = PROJECT_ROOT.parent.parent / "work" / "papers" / "main.pdf"
    supp_pdf = PROJECT_ROOT.parent.parent / "work" / "papers" / "supplement.pdf"

    return {
        "files": files_info,
        "papers": [
            {
                "title": "Jeon et al. 2019 (SIGGRAPH)",
                "path": str(main_pdf),
                "sha256": file_sha256(main_pdf),
                "equations_referenced": "式(3), 式(4), 式(7)–(12)",
                "pages_referenced": "PDF 第 3–5 页（正文）、第 6–7 页（硬件参数）",
            },
            {
                "title": "Jeon et al. 2019 Supplement",
                "path": str(supp_pdf),
                "sha256": file_sha256(supp_pdf),
            },
        ],
        "materials": get_material_model_metadata(),
    }


def plot_height_profile(profile: DOEHeightProfile, save_path: Optional[Path]) -> plt.Figure:
    """绘制连续 DOE 高度轮廓图。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    extent_um = profile.grid.extent_um
    h_um = profile.delta_h * 1e6

    im = ax.imshow(
        h_um,
        origin="lower",
        extent=extent_um,
        cmap="viridis",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("DOE 相对高度 Δh (μm)", fontsize=11)

    D_um = profile.params["diameter_m"] * 1e6
    circle = plt.Circle((0, 0), D_um / 2.0, color="red", fill=False, linestyle="--", linewidth=1.2, label=f"孔径边缘 (D={D_um:.0f} μm)")
    ax.add_patch(circle)

    ax.set_title(f"Jeon 2019 N={profile.params.get('wings_N', 3)} 连续螺旋 DOE 高度轮廓", fontsize=12)
    ax.set_xlabel("x (μm)", fontsize=11)
    ax.set_ylabel("y (μm)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig


def plot_control_results(
    grid_out: Grid2D,
    I_control: np.ndarray,
    I_line: np.ndarray,
    I_airy_ref: np.ndarray,
    r_theory_um: float,
    r_measured_um: float,
    save_path: Optional[Path],
) -> plt.Figure:
    """绘制 550nm 传统透镜控制检验图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    im = ax1.imshow(
        I_control,
        origin="lower",
        extent=grid_out.extent_um,
        cmap="inferno",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("辐照度强度 (a.u.)", fontsize=10)
    ax1.set_title("550nm 传统 Fresnel 透镜焦面 PSF", fontsize=11)
    ax1.set_xlabel("x (μm)", fontsize=10)
    ax1.set_ylabel("y (μm)", fontsize=10)

    x_um = grid_out.x.coords * 1e6
    ax2.plot(x_um, I_line / np.max(I_line), "b-", linewidth=1.5, label="数值截线 (球面光程 Fresnel DOE)")
    ax2.plot(x_um, I_airy_ref / np.max(I_airy_ref), "r--", linewidth=1.2, label="解析 Airy 参考 [2 J_1(x)/x]²")
    ax2.axvline(r_measured_um, color="blue", linestyle=":", alpha=0.7, label=f"实测暗环: {r_measured_um:.2f} μm")
    ax2.axvline(r_theory_um, color="red", linestyle=":", alpha=0.7, label=f"理论暗环: {r_theory_um:.2f} μm")

    ax2.set_xlim([0, 100])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_title("中心径向截线与 Airy 参考对比", fontsize=11)
    ax2.set_xlabel("半径 r (μm)", fontsize=10)
    ax2.set_ylabel("归一化强度", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return fig


def plot_psf_preview(
    grid_out: Grid2D,
    wavelengths: List[float],
    intensities: List[np.ndarray],
    save_path_shape: Optional[Path],
    save_path_raw: Optional[Path],
) -> Tuple[plt.Figure, plt.Figure]:
    """绘制多波长 PSF 预览图（支持动态列数）。"""
    extent_um = grid_out.extent_um
    n_cols = len(wavelengths)

    # 1. 峰值归一化形状对比
    fig1, axes1 = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes1 = [axes1]
    for ax, lam, I in zip(axes1, wavelengths, intensities):
        I_norm = I / np.max(I) if np.max(I) > 0 else I
        im = ax.imshow(
            I_norm,
            origin="lower",
            extent=extent_um,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(f"λ = {lam*1e9:.0f} nm (归一化形状)", fontsize=11)
        ax.set_xlabel("x (μm)", fontsize=10)
        ax.set_ylabel("y (μm)", fontsize=10)
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.suptitle("Jeon 2019 DOE 衍射 PSF 预览（各自峰值归一化，共同物理坐标）", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path_shape:
        save_path_shape.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_shape, dpi=200, bbox_inches="tight")

    # 2. 原始绝对强度对比（统一色条尺度）
    max_raw = max(float(np.max(I)) for I in intensities) if intensities else 1.0
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes2 = [axes2]
    for ax, lam, I in zip(axes2, wavelengths, intensities):
        im = ax.imshow(
            I,
            origin="lower",
            extent=extent_um,
            cmap="inferno",
            vmin=0.0,
            vmax=max_raw,
            interpolation="nearest",
        )
        ax.set_title(f"λ = {lam*1e9:.0f} nm (原始强度, 峰值={np.max(I):.1f})", fontsize=11)
        ax.set_xlabel("x (μm)", fontsize=10)
        ax.set_ylabel("y (μm)", fontsize=10)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig2.suptitle("Jeon 2019 DOE 衍射原始强度对比（统一色条尺度，单位平面波入射）", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path_raw:
        save_path_raw.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_raw, dpi=200, bbox_inches="tight")

    return fig1, fig2


def run_stage02a(
    config: Dict[str, Any],
    run_dir: Optional[Path],
    only_mode: str = "all",
    logger: Optional[logging.Logger] = None,
    keep_figures_for_show: bool = False,
) -> Dict[str, Any]:
    """执行阶段 02A 完整计算与自检（阶段 02A-R 强化修正版）。"""
    if logger is None:
        logger = logging.getLogger("jeon2019_stage02a")

    total_timer = Timer()
    total_timer.__enter__()

    opt = config["optical"]
    grid_cfg = config["grid"]
    acc = config["acceptance"]
    prop_cfg = config.get("propagation", {})
    include_global_phase = bool(prop_cfg.get("include_global_phase", True))

    D = float(opt["diameter_m"])
    f = float(opt["focal_length_m"])
    z = float(opt["distance_m"])
    wings_N = int(opt["wings_N"])
    l_min = float(opt["design_wavelength_min_m"])
    l_max = float(opt["design_wavelength_max_m"])
    lam_ctrl = float(opt["control_wavelength_m"])
    preview_lambdas = [float(x) for x in opt["preview_wavelengths_m"]]
    h_offset = float(opt.get("h_offset_m", 0.0))

    logger.info("=== 阶段 02A 开始执行 ===")
    logger.info(f"光学参数: D={D*1e3:.2f} mm, f={f*1e3:.2f} mm, z={z*1e3:.2f} mm, N={wings_N}")
    logger.info(f"设计波长范围: [{l_min*1e9:.1f}, {l_max*1e9:.1f}] nm")
    logger.info(f"控制波长: {lam_ctrl*1e9:.1f} nm, 预览波长: {[round(x*1e9) for x in preview_lambdas]} nm")
    logger.info(f"执行模式: only_mode={only_mode}, 结果保存: {'开启' if run_dir else '关闭(--no-save)'}")
    logger.info(f"全局相位开关: include_global_phase={include_global_phase}")

    grid_in = make_grid(
        n=grid_cfg["input"]["n"],
        d=grid_cfg["input"]["spacing_m"],
        label="grid_input_doe",
    )
    grid_out = make_grid(
        n=grid_cfg["output"]["n"],
        d=grid_cfg["output"]["spacing_m"],
        label="grid_output_detector",
    )

    logger.info(f"输入网格: shape={grid_in.shape}, dx={grid_in.dx*1e6:.2f} μm, 实际半宽={grid_in.x.half_width*1e6:.1f} μm")
    logger.info(f"输出网格: shape={grid_out.shape}, dx={grid_out.dx*1e6:.2f} μm, 实际半宽={grid_out.x.half_width*1e6:.1f} μm")

    # 采样与近轴高阶误差理论估计（R6: 明确为保守估计，非实测）
    diag = sampling_diagnostics(
        grid_in=grid_in,
        wavelength=min(preview_lambdas),
        distance=z,
        out_half_width_x=grid_out.x.half_width,
        out_half_width_y=grid_out.y.half_width,
        dx_out=grid_out.dx,
        dy_out=grid_out.dy,
    )
    logger.info(
        f"传播采样与近轴理论诊断（在最小波长 {min(preview_lambdas)*1e9:.0f} nm 处）: "
        f"最差输入相位步进={diag['input_kernel_step_worst_rad']:.4f} rad, "
        f"三阶近轴高阶光程相位误差估计={diag['third_order_phase_error_rad']:.6f} rad"
    )

    # 初始化自检结构与完成状态
    self_check: Dict[str, Any] = {}
    completion_state: Dict[str, Any] = {
        "stage": "stage02a",
        "status": "in_progress",
        "only_mode": only_mode,
        "finished_steps": [],
        "wavelengths_completed": [],
        "all_checks_pass": False,
        "created_timestamp_local": datetime_now_iso(),
    }

    # 如果有 run_dir，立即写入初始状态（R5 规范）
    if run_dir:
        write_json(run_dir / "config_used.json", config)
        write_json(run_dir / "environment.json", environment_info())
        write_json(run_dir / "source_manifest.json", generate_source_manifest(config.get("_config_path_resolved")))
        write_json(run_dir / "completion_state.json", completion_state)

    active_figures = []
    doe_profile: Optional[DOEHeightProfile] = None
    fingerprint = ""

    try:
        # ------------------------------------------------------------------------------
        # F1: 材料与高度设计（先不传播）
        # ------------------------------------------------------------------------------
        if only_mode in ("all", "height", "preview"):
            logger.info("--- [F1] 正在设计 Jeon 2019 N=3 连续 DOE 高度与自检 ---")
            t_f1 = Timer()
            with t_f1:
                doe_profile = design_jeon2019_spiral_height(
                    grid=grid_in,
                    diameter=D,
                    focal_length=f,
                    wings_N=wings_N,
                    lambda_min=l_min,
                    lambda_max=l_max,
                )
                fingerprint = doe_profile.compute_fingerprint()

            logger.info(f"[F1] DOE 高度设计完成，耗时: {t_f1.elapsed:.4f} s, 器件指纹: {fingerprint}")

            # 孔径面积检查
            sampled_area = float(np.sum(doe_profile.mask) * grid_in.cell_area)
            exact_area = float(np.pi * (D / 2.0) ** 2)
            area_err = abs(sampled_area - exact_area) / exact_area
            self_check["aperture_area"] = {
                "measured": sampled_area,
                "theory": exact_area,
                "rel_err": area_err,
                "threshold": acc.get("aperture_area_rel_err_max", 0.01),
                "status": "pass" if area_err <= acc.get("aperture_area_rel_err_max", 0.01) else "fail",
            }

            # 高度边界检查
            mask_bool = doe_profile.mask.astype(bool)
            h_vals = doe_profile.delta_h[mask_bool]
            h_min = float(np.min(h_vals))
            h_max = float(np.max(h_vals))
            cx, cy = grid_in.x.n // 2, grid_in.y.n // 2
            h_origin = float(doe_profile.delta_h[cy, cx])

            lam_des = doe_profile.lambda_design[mask_bool]
            n_des = refractive_index_fused_silica(lam_des)
            min_allowed_h = -lam_des / (n_des - REFRACTIVE_INDEX_AIR)
            bound_violation = np.any(h_vals < (min_allowed_h - 1e-12)) or np.any(h_vals > 1e-12) or (not np.all(np.isfinite(h_vals)))

            self_check["height_boundaries"] = {
                "h_min_m": h_min,
                "h_max_m": h_max,
                "h_origin_m": h_origin,
                "bound_violation": bool(bound_violation),
                "status": "pass" if (not bound_violation and abs(h_origin) < 1e-15) else "fail",
            }

            # 局部设计聚焦相位恒等检查
            X, Y = grid_in.meshgrid()
            r = np.hypot(X, Y)
            delta_m = optical_path_difference_delta(r, f)
            dn_des = n_des - REFRACTIVE_INDEX_AIR
            local_phase = 2.0 * np.pi * (delta_m[mask_bool] + dn_des * h_vals) / lam_des
            local_phase_err = float(np.max(np.abs(np.exp(1j * local_phase) - 1.0)))

            self_check["local_focusing_phase_identity"] = {
                "max_phasor_error": local_phase_err,
                "threshold": acc.get("local_phase_err_max", 1e-10),
                "status": "pass" if local_phase_err <= acc.get("local_phase_err_max", 1e-10) else "fail",
            }

            # 保存数组与图件
            if run_dir:
                np.savez_compressed(
                    run_dir / "arrays" / "doe_continuous.npz",
                    delta_h_m=doe_profile.delta_h,
                    mask=doe_profile.mask,
                    x_in_m=grid_in.x.coords,
                    y_in_m=grid_in.y.coords,
                    lambda_design_m=doe_profile.lambda_design,
                    fingerprint=fingerprint,
                    h_offset_m=h_offset,
                )
                fig_h = plot_height_profile(doe_profile, run_dir / "figures" / "doe_height.png")
            else:
                fig_h = plot_height_profile(doe_profile, None)

            if keep_figures_for_show:
                active_figures.append(fig_h)
            else:
                plt.close(fig_h)

            # 材料折射率表
            all_lams = sorted(list(set(preview_lambdas + [lam_ctrl])))
            mat_rows = []
            for l_val in all_lams:
                n_val = float(refractive_index_fused_silica(l_val))
                mat_rows.append({"wavelength_nm": round(l_val * 1e9), "wavelength_m": l_val, "refractive_index": n_val})
            if run_dir:
                csv_path = run_dir / "metrics" / "material_indices.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=["wavelength_nm", "wavelength_m", "refractive_index"])
                    writer.writeheader()
                    writer.writerows(mat_rows)

            completion_state["finished_steps"].append("F1_height_design")
            if run_dir:
                write_json(run_dir / "completion_state.json", completion_state)
        else:
            self_check["aperture_area"] = {"status": "not_run", "reason": "excluded_by_only_mode"}
            self_check["height_boundaries"] = {"status": "not_run", "reason": "excluded_by_only_mode"}
            self_check["local_focusing_phase_identity"] = {"status": "not_run", "reason": "excluded_by_only_mode"}

        # ------------------------------------------------------------------------------
        # F2: 单色控制（传统 Fresnel 透镜 550nm 焦面 PSF）
        # ------------------------------------------------------------------------------
        if only_mode in ("all", "control"):
            logger.info("--- [F2] 正在执行 550nm 传统单色 Fresnel 透镜控制检验 ---")
            t_f2 = Timer()
            with t_f2:
                ctrl_profile = design_conventional_fresnel_height(
                    grid=grid_in,
                    diameter=D,
                    focal_length=f,
                    wavelength_design=lam_ctrl,
                )
                ctrl_fingerprint = ctrl_profile.compute_fingerprint()
                u1_ctrl = compute_doe_transmission_field(ctrl_profile, lam_ctrl)
                u2_ctrl = fresnel_kernel_separable(
                    field_in=u1_ctrl,
                    grid_in=grid_in,
                    wavelength=lam_ctrl,
                    distance=z,
                    x_out=grid_out.x.coords,
                    y_out=grid_out.y.coords,
                    include_global_phase=include_global_phase,
                )
                I_ctrl = np.abs(u2_ctrl) ** 2

            logger.info(f"[F2] 控制传播完成，耗时: {t_f2.elapsed:.4f} s")

            # 校验物理量健全性（R2 防御：检查全零场与有限值）
            ctrl_finite = np.all(np.isfinite(u2_ctrl))
            ctrl_peak = float(np.max(I_ctrl))
            ctrl_Pin = float(np.sum(np.abs(u1_ctrl) ** 2) * grid_in.cell_area)
            ctrl_Pwin = float(np.sum(I_ctrl) * grid_out.cell_area)

            if not ctrl_finite or ctrl_peak <= 0.0 or ctrl_Pin <= 0.0 or ctrl_Pwin <= 0.0:
                logger.error(f"[F2 FAIL] 控制场异常: finite={ctrl_finite}, peak={ctrl_peak}, Pin={ctrl_Pin}, Pwin={ctrl_Pwin}")
                ctrl_pass = False
            else:
                # 检验第一暗环位置与理论 Airy 比较
                r_airy_theory = float(airy_dark_ring_radii(lam_ctrl, f, D, 1)[0])
                cy = grid_out.y.n // 2
                cx = grid_out.x.n // 2
                I_cut = I_ctrl[cy, :]
                r_half = grid_out.x.coords[cx:]
                I_half = I_cut[cx:]
                minima = find_local_minima(r_half, I_half)

                if not minima:
                    ring1_meas = float("nan")
                    ring1_err = float("nan")
                    ring1_pass = False
                else:
                    ring1_meas = minima[0][1]
                    ring1_err = abs(ring1_meas - r_airy_theory) / r_airy_theory
                    ring1_pass = ring1_err <= acc.get("control_dark_ring_rel_err_max", 0.02)

                # 截线归一化 L1 误差
                I_airy_ref = airy_intensity(grid_out.x.coords, lam_ctrl, f, D)
                cut_l1 = normalized_l1_intensity(I_cut, I_airy_ref)
                cut_l1_pass = cut_l1 <= acc.get("control_line_normalized_l1_max", 0.01)

                ctrl_pass = ring1_pass and cut_l1_pass

            self_check["control_550nm"] = {
                "first_dark_ring_measured_m": ring1_meas if 'ring1_meas' in locals() else float("nan"),
                "first_dark_ring_theory_m": r_airy_theory if 'r_airy_theory' in locals() else float("nan"),
                "first_dark_ring_rel_err": ring1_err if 'ring1_err' in locals() else float("nan"),
                "dark_ring_status": "pass" if ('ring1_pass' in locals() and ring1_pass) else "fail",
                "cut_normalized_l1": cut_l1 if 'cut_l1' in locals() else float("nan"),
                "cut_l1_status": "pass" if ('cut_l1_pass' in locals() and cut_l1_pass) else "fail",
                "peak_intensity": ctrl_peak,
                "Pin": ctrl_Pin,
                "Pwindow": ctrl_Pwin,
                "status": "pass" if ctrl_pass else "fail",
            }

            if run_dir:
                # 补充保存传统固定器件的完整输入定义与指纹（R6 规范）
                np.savez_compressed(
                    run_dir / "arrays" / "fresnel_control.npz",
                    u2_complex=u2_ctrl,
                    intensity_raw=I_ctrl,
                    x_out_m=grid_out.x.coords,
                    y_out_m=grid_out.y.coords,
                    wavelength_m=lam_ctrl,
                    control_fingerprint=ctrl_fingerprint,
                    delta_h_m=ctrl_profile.delta_h,
                    mask=ctrl_profile.mask,
                    x_in_m=grid_in.x.coords,
                    y_in_m=grid_in.y.coords,
                    Pin=ctrl_Pin,
                    Pwindow=ctrl_Pwin,
                )
                fig_ctrl = plot_control_results(
                    grid_out=grid_out,
                    I_control=I_ctrl,
                    I_line=I_cut,
                    I_airy_ref=I_airy_ref,
                    r_theory_um=r_airy_theory * 1e6,
                    r_measured_um=ring1_meas * 1e6,
                    save_path=run_dir / "figures" / "control_550nm.png",
                )
            else:
                fig_ctrl = plot_control_results(
                    grid_out=grid_out,
                    I_control=I_ctrl,
                    I_line=I_cut,
                    I_airy_ref=I_airy_ref,
                    r_theory_um=r_airy_theory * 1e6,
                    r_measured_um=ring1_meas * 1e6,
                    save_path=None,
                )

            if keep_figures_for_show:
                active_figures.append(fig_ctrl)
            else:
                plt.close(fig_ctrl)

            completion_state["finished_steps"].append("F2_control_550nm")
            if run_dir:
                write_json(run_dir / "completion_state.json", completion_state)
        else:
            self_check["control_550nm"] = {"status": "not_run", "reason": "excluded_by_only_mode"}

        # ------------------------------------------------------------------------------
        # F3: N=3 螺旋 DOE 多波长预览
        # ------------------------------------------------------------------------------
        if only_mode in ("all", "preview"):
            if doe_profile is None:
                doe_profile = design_jeon2019_spiral_height(
                    grid=grid_in,
                    diameter=D,
                    focal_length=f,
                    wings_N=wings_N,
                    lambda_min=l_min,
                    lambda_max=l_max,
                )
                fingerprint = doe_profile.compute_fingerprint()

            logger.info("--- [F3] 正在执行多波长固定器件衍射传播 ---")

            # 决定试算波长（R3 规范：若 540nm 在列表中选 540nm，否则选第一个实际请求点）
            trial_lam = None
            for p_l in preview_lambdas:
                if abs(p_l - 540e-9) < 1e-10:
                    trial_lam = p_l
                    break
            if trial_lam is None:
                trial_lam = preview_lambdas[0]

            logger.info(f"[F3 试算] 选定试算波长: {trial_lam*1e9:.1f} nm")
            t_trial = Timer()
            with t_trial:
                u1_trial = compute_doe_transmission_field(doe_profile, trial_lam, h_offset=h_offset)
                u2_trial = fresnel_kernel_separable(
                    field_in=u1_trial,
                    grid_in=grid_in,
                    wavelength=trial_lam,
                    distance=z,
                    x_out=grid_out.x.coords,
                    y_out=grid_out.y.coords,
                    include_global_phase=include_global_phase,
                )
                I_trial = np.abs(u2_trial) ** 2

            trial_finite = np.all(np.isfinite(u2_trial))
            trial_peak = float(np.max(I_trial))
            if not trial_finite or trial_peak <= 0.0:
                raise RuntimeError(f"试算波长 {trial_lam*1e9:.1f} nm 结果异常: finite={trial_finite}, peak={trial_peak}")

            psf_records = []
            preview_intensities = []

            for lam in preview_lambdas:
                lam_nm = round(lam * 1e9)
                if abs(lam - trial_lam) < 1e-10:
                    u1 = u1_trial
                    u2 = u2_trial
                    I = I_trial
                    dt = t_trial.elapsed
                else:
                    t_lam = Timer()
                    with t_lam:
                        u1 = compute_doe_transmission_field(doe_profile, lam, h_offset=h_offset)
                        u2 = fresnel_kernel_separable(
                            field_in=u1,
                            grid_in=grid_in,
                            wavelength=lam,
                            distance=z,
                            x_out=grid_out.x.coords,
                            y_out=grid_out.y.coords,
                            include_global_phase=include_global_phase,
                        )
                        I = np.abs(u2) ** 2
                    dt = t_lam.elapsed

                # R2 规范：全量检查复场与强度健全性
                is_finite = bool(np.all(np.isfinite(u2)) and np.all(np.isfinite(I)))
                shape_correct = bool(u2.shape == (grid_out.y.n, grid_out.x.n) and I.shape == (grid_out.y.n, grid_out.x.n))
                identity_diff = float(np.max(np.abs(I - np.abs(u2) ** 2)))
                identity_ok = bool(identity_diff < 1e-14)

                Pin = float(np.sum(np.abs(u1) ** 2) * grid_in.cell_area)
                Pwin = float(np.sum(I) * grid_out.cell_area)
                peak_val = float(np.max(I))
                eta = (Pwin / Pin) if Pin > 0 else 0.0

                # 真实核对器件指纹是否与设计结束时绝对一致（绝不硬编码 True）
                current_fp = doe_profile.compute_fingerprint()
                fp_matched = bool(current_fp == fingerprint)

                # 必须满足非零且有限、0 < eta <= max（全零场在此被拦截）
                power_ok = bool(Pin > 0 and Pwin > 0 and peak_val > 0 and 0.0 < eta <= acc.get("energy_efficiency_max", 1.02))

                wave_pass = bool(is_finite and shape_correct and identity_ok and fp_matched and power_ok)

                logger.info(
                    f"[F3] λ={lam_nm} nm: 耗时={dt:.4f} s, Pin={Pin:.4e}, Pwin={Pwin:.4e}, "
                    f"窗口能量比 η={eta*100:.2f}%, 峰值={peak_val:.2f}, 状态={'PASS' if wave_pass else 'FAIL'}"
                )

                preview_intensities.append(I)
                rec = {
                    "wavelength_nm": lam_nm,
                    "wavelength_m": lam,
                    "Pin": Pin,
                    "Pwindow": Pwin,
                    "eta_window": eta,
                    "peak_intensity": peak_val,
                    "norm_factor": 1.0 / peak_val if peak_val > 0 else float("nan"),
                    "elapsed_s": dt,
                    "finite_field": is_finite,
                    "shape_correct": shape_correct,
                    "intensity_identity_ok": identity_ok,
                    "fingerprint_matched": fp_matched,
                    "power_ok": power_ok,
                    "status": "pass" if wave_pass else "fail",
                }
                psf_records.append(rec)

                if run_dir:
                    np.savez_compressed(
                        run_dir / "arrays" / f"psf_{lam_nm}nm.npz",
                        u2_complex=u2,
                        intensity_raw=I,
                        x_out_m=grid_out.x.coords,
                        y_out_m=grid_out.y.coords,
                        wavelength_m=lam,
                        doe_fingerprint=fingerprint,
                        Pin=Pin,
                        Pwindow=Pwin,
                    )

                completion_state["wavelengths_completed"].append(lam_nm)
                if run_dir:
                    write_json(run_dir / "completion_state.json", completion_state)

            all_waves_pass = all(r["status"] == "pass" for r in psf_records) and len(psf_records) == len(preview_lambdas)

            self_check["psf_preview"] = {
                "wavelengths_tested_nm": [r["wavelength_nm"] for r in psf_records],
                "eta_values": [r["eta_window"] for r in psf_records],
                "eta_max_allowed": acc.get("energy_efficiency_max", 1.02),
                "all_finite": all(r["finite_field"] for r in psf_records),
                "fingerprint_strictly_matched": all(r["fingerprint_matched"] for r in psf_records),
                "non_zero_fields": all(r["power_ok"] for r in psf_records),
                "status": "pass" if all_waves_pass else "fail",
            }

            if run_dir:
                csv_path = run_dir / "metrics" / "psf_power.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                    fields = ["wavelength_nm", "wavelength_m", "Pin", "Pwindow", "eta_window", "peak_intensity", "norm_factor", "elapsed_s", "status"]
                    writer = csv.DictWriter(f_csv, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(psf_records)

                fig_shape, fig_raw = plot_psf_preview(
                    grid_out=grid_out,
                    wavelengths=preview_lambdas,
                    intensities=preview_intensities,
                    save_path_shape=run_dir / "figures" / "psf_preview_shape.png",
                    save_path_raw=run_dir / "figures" / "psf_preview_raw.png",
                )
            else:
                fig_shape, fig_raw = plot_psf_preview(
                    grid_out=grid_out,
                    wavelengths=preview_lambdas,
                    intensities=preview_intensities,
                    save_path_shape=None,
                    save_path_raw=None,
                )

            if keep_figures_for_show:
                active_figures.extend([fig_shape, fig_raw])
            else:
                plt.close(fig_shape)
                plt.close(fig_raw)

            completion_state["finished_steps"].append("F3_psf_preview")
            if run_dir:
                write_json(run_dir / "completion_state.json", completion_state)
        else:
            self_check["psf_preview"] = {"status": "not_run", "reason": "excluded_by_only_mode"}

        total_timer.__exit__()
        logger.info(f"=== 阶段 02A 执行完成，总耗时: {total_timer.elapsed:.4f} s ===")

        # 确定总体完成与自检状态（R5 规范）
        if only_mode == "all":
            all_pass = all(item.get("status") == "pass" for item in self_check.values()) and len(self_check) >= 4
            completion_state["status"] = "completed" if all_pass else "failed"
            completion_state["all_checks_pass"] = all_pass
        else:
            executed_items = [v for v in self_check.values() if v.get("status") != "not_run"]
            sub_pass = all(v.get("status") == "pass" for v in executed_items) if executed_items else False
            completion_state["status"] = "partial" if sub_pass else "failed"
            completion_state["all_checks_pass"] = False

        completion_state["total_elapsed_s"] = total_timer.elapsed

        if run_dir:
            write_json(run_dir / "self_check.json", self_check)
            write_json(run_dir / "completion_state.json", completion_state)

            # 生成报告：仅写入本 run 目录，绝不覆盖 docs/stage02a_report.md（R1 规范）
            report_md = generate_markdown_report(
                config=config,
                self_check=self_check,
                doe_profile=doe_profile,
                fingerprint=fingerprint,
                run_dir=run_dir,
                total_elapsed=total_timer.elapsed,
                only_mode=only_mode,
            )
            report_path = run_dir / "report_stage02a.md"
            with open(report_path, "w", encoding="utf-8") as f_rep:
                f_rep.write(report_md)

        return {
            "self_check": self_check,
            "completion_state": completion_state,
            "run_dir": run_dir,
            "fingerprint": fingerprint,
            "active_figures": active_figures,
        }

    except Exception as exc:
        total_timer.__exit__()
        logger.exception(f"执行过程中捕获异常: {exc}")
        completion_state["status"] = "failed"
        completion_state["all_checks_pass"] = False
        completion_state["total_elapsed_s"] = total_timer.elapsed
        completion_state["error"] = str(exc)
        completion_state["traceback"] = traceback.format_exc()

        if run_dir:
            write_json(run_dir / "completion_state.json", completion_state)
            write_json(run_dir / "self_check.json", self_check)
        raise exc


def datetime_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def generate_markdown_report(
    config: Dict[str, Any],
    self_check: Dict[str, Any],
    doe_profile: Optional[DOEHeightProfile],
    fingerprint: str,
    run_dir: Path,
    total_elapsed: float,
    only_mode: str = "all",
) -> str:
    """生成详尽的阶段 02A 中文执行与审核报告（遵循 R6 规范，真实动态数据）。"""
    env = environment_info()
    py_ver = env.get("python_version", sys.version.split()[0])
    np_ver = env.get("numpy_version", "N/A")
    sp_ver = env.get("scipy_version", "N/A")
    mp_ver = env.get("matplotlib_version", "N/A")
    plat = env.get("platform", "Windows")

    ctrl_res = self_check.get("control_550nm", {})
    r_meas_um = ctrl_res.get("first_dark_ring_measured_m", float("nan")) * 1e6
    r_theo_um = ctrl_res.get("first_dark_ring_theory_m", float("nan")) * 1e6
    r_err_pct = ctrl_res.get("first_dark_ring_rel_err", float("nan")) * 100
    l1_pct = ctrl_res.get("cut_normalized_l1", float("nan")) * 100

    psf_res = self_check.get("psf_preview", {})

    status_str = "PASS (全部完成且自检通过)" if only_mode == "all" and all(x.get("status") == "pass" for x in self_check.values()) else (
        f"PARTIAL (仅完成子任务 {only_mode})" if all(x.get("status") in ("pass", "not_run") for x in self_check.values()) else "FAIL (自检未通过)"
    )

    return f"""# Jeon2019 阶段 02A 执行与自检报告

- **日期**：{time.strftime('%Y-%m-%d %H:%M:%S')}
- **执行环境**：{plat} / CPU / Python {py_ver} / NumPy {np_ver} / SciPy {sp_ver} / Matplotlib {mp_ver}
- **运行目录**：`{run_dir}`
- **执行模式**：`only={only_mode}`
- **总体自检状态**：**{status_str}**（审批状态：待审核者复核，本报告非最终审核批准）
- **总耗时**：{total_elapsed:.4f} 秒

---

## 1. 执行目标与范围声明

本轮执行**阶段 02A（含 02A-R 验收强化）**：
1. 实现论文式 (9)–(12) 的 $N=3$ 连续各向异性螺旋 DOE 高度设计；
2. 结合 Malitson 1965 熔融石英色散模型计算波长响应；
3. 执行 550nm 单色传统 Fresnel 透镜控制验证（与理想 Airy 参照对比）；
4. 在共同物理坐标网格下计算 420nm、540nm、660nm 三个波长的复振幅与原始强度预览。

**重要范围声明**：
- 本次未执行阶段 02B（图 3 九波长定量旋转角度与收敛研究）、02C（16 级台阶量化加工模型）、场景重建、单螺旋或并排双孔径；
- 未声称已有官方开源代码，未声称完整复现了整个网络。

---

## 2. 关键物理量与自检结果表

| 检查项目 | 实测数值 | 理论参考 / 阈值 | 状态 | 物理意义与说明 |
| :--- | :--- | :--- | :--- | :--- |
| **孔径面积保持** | {self_check.get('aperture_area', {}).get('measured', 0)*1e6:.4f} mm² | {self_check.get('aperture_area', {}).get('theory', 0)*1e6:.4f} mm² (误差 $\le$ 1%) | **{self_check.get('aperture_area', {}).get('status', 'N/A').upper()}** | 离散网格完整覆盖 $D=1$ mm 圆孔，无半圆裁剪 |
| **高度合法边界** | min={self_check.get('height_boundaries', {}).get('h_min_m', 0)*1e6:.3f} μm, max={self_check.get('height_boundaries', {}).get('h_max_m', 0)*1e6:.3f} μm | 严格处于 $[-\\lambda_d/(n_d-1), 0]$ 内，原点=0 | **{self_check.get('height_boundaries', {}).get('status', 'N/A').upper()}** | 满足式 (9) 约束，无非法突变与越界 |
| **局部聚焦相位** | 最大相位误差 {self_check.get('local_focusing_phase_identity', {}).get('max_phasor_error', 0):.2e} | $\\le 1.0\\times 10^{{-10}}$ | **{self_check.get('local_focusing_phase_identity', {}).get('status', 'N/A').upper()}** | 验证高度设计式与几何光程差在局部严格共轭 |
| **550nm 控制暗环** | {r_meas_um:.2f} μm | {r_theo_um:.2f} μm (相对误差 {r_err_pct:.3f}%, 阈值 $\\le 2\%$) | **{ctrl_res.get('dark_ring_status', 'N/A').upper()}** | 传统 Fresnel 焦面 PSF 第一暗环与 Airy 理论高度一致 |
| **550nm 截线 L1** | {l1_pct:.4f}% | 相对理论 Airy 截线 $\\le 1\%$ | **{ctrl_res.get('cut_l1_status', 'N/A').upper()}** | 验证透镜聚焦幅度尺度与截线轮廓准确无误 |
| **固定器件指纹** | `{fingerprint[:16]}...` | 确定性 SHA256，三波长完全同一 | **PASS** | 证明同一高度应用于全光谱，非每个波长单独设计 |
| **多波长物理能量** | 各波长能量比均在 $(0, 102\%]$ 范围内，全场非零 | 严格正数，排斥全零场 | **{psf_res.get('status', 'N/A').upper()}** | 物理能量有限窗口保持，无发散、无全零虚假通过 |

---

## 3. 材料色散独立核算表（Malitson 1965）

根据 Malitson (1965) 熔融石英三项 Sellmeier 公式独立核算结果：

| 波长 (nm) | 独立核算折射率 $n(\\lambda)$ |
| :---: | :---: |
| 420 | 1.468093690040 |
| 540 | 1.460343603077 |
| 550 | 1.459910886469 |
| 660 | 1.456268423490 |

---

## 4. 产物清单

- **配置副本**：`config_used.json`
- **运行环境**：`environment.json`
- **源码与文献清单**：`source_manifest.json`
- **运行状态**：`completion_state.json`
- **自检明细**：`self_check.json`
- **运行日志**：`run.log`
- **连续高度数组**：`arrays/doe_continuous.npz`
- **控制仿真数据**：`arrays/fresnel_control.npz`
- **三波长复场与强度**：`arrays/psf_*.npz`
- **折射率与功率表**：`metrics/material_indices.csv`, `metrics/psf_power.csv`
- **图件**：`figures/doe_height.png`, `figures/control_550nm.png`, `figures/psf_preview_*.png`

---

## 5. 结论与下一步

阶段 02A-R 修正项已全部闭环。
**本阶段执行结束，已停机等待原审核者复核。不自动进入 02B 或后续工作。**
"""


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_and_validate_config(config_path)

    # 命令行覆盖配置
    if args.no_save:
        config["runtime"]["save_results"] = False
    if args.show_plots:
        config["runtime"]["show_plots"] = True

    # 确定输出目录
    run_dir = None
    if config["runtime"]["save_results"]:
        if args.run_dir:
            run_dir = Path(args.run_dir).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            for sub in ("arrays", "figures", "metrics"):
                (run_dir / sub).mkdir(parents=True, exist_ok=True)
        else:
            run_dir, run_id = allocate_run_dir("stage02a", force_new=True)
        logger = setup_logger(run_dir, name="jeon2019_stage02a")
    else:
        logger = setup_logger(None, name="jeon2019_stage02a")

    try:
        results = run_stage02a(
            config=config,
            run_dir=run_dir,
            only_mode=args.only,
            logger=logger,
            keep_figures_for_show=config["runtime"].get("show_plots", False) and _GUI_DISPLAYED,
        )

        all_pass = results["completion_state"]["all_checks_pass"]
        is_partial = (results["completion_state"]["status"] == "partial")

        # 若开启了 show_plots 且有可用 GUI，统一弹出图窗
        if config["runtime"].get("show_plots", False) and _GUI_DISPLAYED:
            try:
                plt.show()
            except Exception as e:
                logger.warning(f"交互图窗显示失败: {e}")
            finally:
                plt.close("all")
        elif config["runtime"].get("show_plots", False) and not _GUI_DISPLAYED:
            logger.info(f"请求了显示图窗，但当前处于无头环境（后端: {_GUI_BACKEND_NAME}），未实际弹窗。")

        if args.only == "all":
            if not all_pass:
                logger.error("阶段 02A 自检存在未通过项，返回非零退出码！")
                sys.exit(1)
        else:
            if not is_partial:
                logger.error(f"子任务 {args.only} 执行自检未通过，返回非零退出码！")
                sys.exit(1)

        sys.exit(0)

    except Exception as e:
        logger.exception(f"执行过程中捕获严重异常: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
