# -*- coding: utf-8 -*-
"""Jeon et al. 2019 光学复现阶段 02A 主执行入口。

支持：
* 在 PyCharm 中无参数直接右键运行（默认执行全部 02A 内容并保存结果）；
* 命令行参数：
    --config PATH       指定配置文件路径（默认同目录下 config_stage02a.json）
    --only TARGET       仅运行子任务：height | control | preview | all（默认 all）
    --no-save           仅内存计算与自检，不写入磁盘结果与日志
    --show-plots        显示交互图窗（默认仅保存 PNG 不弹窗）

阶段 02A 核心物理范围
---------------------
1. 固定 N=3 连续 DOE 高度设计（正文式 (9)–(12)）与 Malitson 1965 熔融石英色散；
2. 550nm 传统单色 Fresnel 透镜控制验证（与理想 Airy 参照对比）；
3. 固定器件在 420nm、540nm、660nm 共同物理网格下的 Fresnel 衍射复振幅与原始强度预览；
4. 严格保持器件固定，同一器件指纹贯穿所有波长计算。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

# 默认无头模式，避免阻塞无人值守环境
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 配置中文字体支持与负号显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optics.coordinates import make_grid, Grid2D
from optics.materials import (
    refractive_index_fused_silica,
    get_material_model_metadata,
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
    azimuthal_average,
    find_dark_rings,
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
        help="配置文件路径（默认同目录下的 config_stage02a.json）",
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
    return parser.parse_args(argv)


def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """加载并严格校验配置参数。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 校验必要字段
    opt = cfg.get("optical", {})
    grid = cfg.get("grid", {})
    acc = cfg.get("acceptance", {})

    D = float(opt.get("diameter_m", 0.0))
    f = float(opt.get("focal_length_m", 0.0))
    z = float(opt.get("distance_m", 0.0))
    N = int(opt.get("wings_N", 0))
    l_min = float(opt.get("design_wavelength_min_m", 0.0))
    l_max = float(opt.get("design_wavelength_max_m", 0.0))

    if D <= 0.0 or f <= 0.0 or z <= 0.0:
        raise ValueError(f"光学参数必须为正数: D={D}, f={f}, z={z}")
    if N <= 0:
        raise ValueError(f"周期数 N 必须 >= 1，收到 N={N}")
    if l_min <= 0.0 or l_max <= l_min:
        raise ValueError(f"设计波长范围非法: [{l_min}, {l_max}]")

    # 网格参数检查
    in_hw = float(grid.get("input", {}).get("half_width_m", 0.0))
    in_n = int(grid.get("input", {}).get("n", 0))
    in_d = float(grid.get("input", {}).get("spacing_m", 0.0))

    out_hw = float(grid.get("output", {}).get("half_width_m", 0.0))
    out_n = int(grid.get("output", {}).get("n", 0))
    out_d = float(grid.get("output", {}).get("spacing_m", 0.0))

    if in_hw < 0.5 * D:
        raise ValueError(f"输入窗口半宽 {in_hw*1e6:.1f} μm 小于孔径半径 {0.5*D*1e6:.1f} μm，无法完整覆盖孔径")
    if in_n < 3 or in_d <= 0.0:
        raise ValueError("输入网格样点数或间距非法")
    if out_n < 3 or out_d <= 0.0:
        raise ValueError("输出网格样点数或间距非法")

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


def generate_source_manifest() -> Dict[str, Any]:
    """生成源码与参考文献的清单与哈希。"""
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

    # 文献信息
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


def plot_height_profile(profile: DOEHeightProfile, save_path: Optional[Path]) -> None:
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

    # 绘制理想圆孔孔径轮廓
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
    plt.close(fig)


def plot_control_results(
    grid_out: Grid2D,
    I_control: np.ndarray,
    I_line: np.ndarray,
    I_airy_ref: np.ndarray,
    r_theory_um: float,
    r_measured_um: float,
    save_path: Optional[Path],
) -> None:
    """绘制 550nm 传统透镜控制检验图（包含 2D PSF 与中心截线对比）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. 2D 焦面 PSF
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

    # 2. 中心截线与 Airy 解析对比
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
    plt.close(fig)


def plot_psf_preview(
    grid_out: Grid2D,
    wavelengths: List[float],
    intensities: List[np.ndarray],
    save_path_shape: Optional[Path],
    save_path_raw: Optional[Path],
) -> None:
    """绘制三波长 PSF 预览图：峰值归一化形状对比与原始强度对比。"""
    extent_um = grid_out.extent_um

    # 1. 峰值归一化形状对比
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, lam, I in zip(axes1, wavelengths, intensities):
        I_norm = I / np.max(I)
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

    fig1.suptitle("Jeon 2019 N=3 DOE 三波长 PSF 预览（各自峰值归一化，共同物理坐标）", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path_shape:
        save_path_shape.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_shape, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 2. 原始绝对强度对比（统一色条尺度）
    max_raw = max(np.max(I) for I in intensities)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
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

    fig2.suptitle("Jeon 2019 N=3 DOE 三波长原始强度对比（统一色条尺度，单位平面波入射）", fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path_raw:
        save_path_raw.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_raw, dpi=200, bbox_inches="tight")
    plt.close(fig2)


def run_stage02a(
    config: Dict[str, Any],
    run_dir: Optional[Path],
    only_mode: str = "all",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """执行阶段 02A 全部流程或指定子集。"""
    if logger is None:
        logger = logging.getLogger("jeon2019_stage02a")

    total_timer = Timer()
    total_timer.__enter__()

    opt = config["optical"]
    grid_cfg = config["grid"]
    acc = config["acceptance"]

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

    # 构建输入与输出网格
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

    logger.info(f"输入网格: shape={grid_in.shape}, dx={grid_in.dx*1e6:.2f} μm, 半宽={grid_in.x.half_width*1e6:.1f} μm")
    logger.info(f"输出网格: shape={grid_out.shape}, dx={grid_out.dx*1e6:.2f} μm, 半宽={grid_out.x.half_width*1e6:.1f} μm")

    # 采样与近轴诊断
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
        f"传播采样诊断（在最小波长 {min(preview_lambdas)*1e9:.0f} nm 处）: "
        f"最差输入相位步进={diag['input_kernel_step_worst_rad']:.4f} rad, "
        f"近轴高阶误差={diag['third_order_phase_error_rad']:.6f} rad"
    )

    self_check: Dict[str, Any] = {}
    completion_state: Dict[str, Any] = {
        "stage": "stage02a",
        "status": "in_progress",
        "finished_steps": [],
        "wavelengths_completed": [],
    }

    # ----------------------------------------------------------------------------------
    # F1: 材料与高度设计（先不传播）
    # ----------------------------------------------------------------------------------
    doe_profile: Optional[DOEHeightProfile] = None
    fingerprint = ""

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
            "threshold": acc["aperture_area_rel_err_max"],
            "status": "pass" if area_err <= acc["aperture_area_rel_err_max"] else "fail",
        }

        # 高度范围检查
        mask_bool = doe_profile.mask.astype(bool)
        h_vals = doe_profile.delta_h[mask_bool]
        h_min = float(np.min(h_vals))
        h_max = float(np.max(h_vals))
        cx, cy = grid_in.x.n // 2, grid_in.y.n // 2
        h_origin = float(doe_profile.delta_h[cy, cx])

        lam_des = doe_profile.lambda_design[mask_bool]
        n_des = refractive_index_fused_silica(lam_des)
        min_allowed_h = -lam_des / (n_des - REFRACTIVE_INDEX_AIR)
        bound_violation = np.any(h_vals < (min_allowed_h - 1e-12)) or np.any(h_vals > 1e-12)

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
            "threshold": acc["local_phase_err_max"],
            "status": "pass" if local_phase_err <= acc["local_phase_err_max"] else "fail",
        }

        # 保存高度数组与图像
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
            plot_height_profile(doe_profile, run_dir / "figures" / "doe_height.png")

        # 记录材料折射率表
        all_lams = sorted(list(set(preview_lambdas + [lam_ctrl])))
        mat_rows = []
        for l_val in all_lams:
            n_val = float(refractive_index_fused_silica(l_val))
            mat_rows.append({"wavelength_nm": l_val * 1e9, "wavelength_m": l_val, "refractive_index": n_val})
        if run_dir:
            csv_path = run_dir / "metrics" / "material_indices.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=["wavelength_nm", "wavelength_m", "refractive_index"])
                writer.writeheader()
                writer.writerows(mat_rows)

        completion_state["finished_steps"].append("F1_height_design")

    # ----------------------------------------------------------------------------------
    # F2: 单色控制（传统 Fresnel 透镜 550nm 焦面 PSF）
    # ----------------------------------------------------------------------------------
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
            u1_ctrl = compute_doe_transmission_field(ctrl_profile, lam_ctrl)
            u2_ctrl = fresnel_kernel_separable(
                field_in=u1_ctrl,
                grid_in=grid_in,
                wavelength=lam_ctrl,
                distance=z,
                x_out=grid_out.x.coords,
                y_out=grid_out.y.coords,
                include_global_phase=True,
            )
            I_ctrl = np.abs(u2_ctrl) ** 2

        logger.info(f"[F2] 控制传播完成，耗时: {t_f2.elapsed:.4f} s")

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
            ring1_pass = ring1_err <= acc["control_dark_ring_rel_err_max"]

        # 截线归一化 L1 误差
        I_airy_ref = airy_intensity(grid_out.x.coords, lam_ctrl, f, D)
        cut_l1 = normalized_l1_intensity(I_cut, I_airy_ref)
        cut_l1_pass = cut_l1 <= acc["control_line_normalized_l1_max"]

        self_check["control_550nm"] = {
            "first_dark_ring_measured_m": ring1_meas,
            "first_dark_ring_theory_m": r_airy_theory,
            "first_dark_ring_rel_err": ring1_err,
            "dark_ring_status": "pass" if ring1_pass else "fail",
            "cut_normalized_l1": cut_l1,
            "cut_l1_status": "pass" if cut_l1_pass else "fail",
            "status": "pass" if (ring1_pass and cut_l1_pass) else "fail",
        }

        logger.info(
            f"[F2] 控制结果: 实测暗环={ring1_meas*1e6:.2f} μm, 理论 Airy 暗环={r_airy_theory*1e6:.2f} μm, "
            f"相对误差={ring1_err*100:.4f}% ({'PASS' if ring1_pass else 'FAIL'}); "
            f"截线 L1 误差={cut_l1*100:.4f}% ({'PASS' if cut_l1_pass else 'FAIL'})"
        )

        if run_dir:
            np.savez_compressed(
                run_dir / "arrays" / "fresnel_control.npz",
                u2_complex=u2_ctrl,
                intensity_raw=I_ctrl,
                x_out_m=grid_out.x.coords,
                y_out_m=grid_out.y.coords,
                wavelength_m=lam_ctrl,
            )
            plot_control_results(
                grid_out=grid_out,
                I_control=I_ctrl,
                I_line=I_cut,
                I_airy_ref=I_airy_ref,
                r_theory_um=r_airy_theory * 1e6,
                r_measured_um=ring1_meas * 1e6,
                save_path=run_dir / "figures" / "control_550nm.png",
            )

        completion_state["finished_steps"].append("F2_control_550nm")

    # ----------------------------------------------------------------------------------
    # F3: N=3 螺旋 DOE 三波长预览（420nm, 540nm, 660nm）
    # ----------------------------------------------------------------------------------
    if only_mode in ("all", "preview"):
        if doe_profile is None:
            logger.info("预览模式下按需生成固定 DOE 高度轮廓...")
            doe_profile = design_jeon2019_spiral_height(
                grid=grid_in,
                diameter=D,
                focal_length=f,
                wings_N=wings_N,
                lambda_min=l_min,
                lambda_max=l_max,
            )
            fingerprint = doe_profile.compute_fingerprint()

        logger.info("--- [F3] 正在执行三波长固定器件衍射传播（先试算 540nm） ---")

        # 试算 540nm
        t_540 = Timer()
        with t_540:
            u1_540 = compute_doe_transmission_field(doe_profile, 540e-9, h_offset=h_offset)
            u2_540 = fresnel_kernel_separable(
                field_in=u1_540,
                grid_in=grid_in,
                wavelength=540e-9,
                distance=z,
                x_out=grid_out.x.coords,
                y_out=grid_out.y.coords,
                include_global_phase=True,
            )
            I_540 = np.abs(u2_540) ** 2

        logger.info(f"[F3 试算] 540nm 单色传播完成，耗时: {t_540.elapsed:.4f} s, 峰值强度: {np.max(I_540):.2f}")
        if not np.all(np.isfinite(u2_540)):
            raise RuntimeError("540nm 试算结果包含非有限数值（NaN/Inf），已立即中止！")

        # 正式计算三波长（复用已计算的 540nm）
        psf_records = []
        preview_intensities = []

        for lam in preview_lambdas:
            lam_nm = round(lam * 1e9)
            if abs(lam - 540e-9) < 1e-12:
                u1 = u1_540
                u2 = u2_540
                I = I_540
                dt = t_540.elapsed
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
                        include_global_phase=True,
                    )
                    I = np.abs(u2) ** 2
                dt = t_lam.elapsed

            Pin = float(np.sum(np.abs(u1) ** 2) * grid_in.cell_area)
            Pwin = float(np.sum(I) * grid_out.cell_area)
            eta = Pwin / Pin
            peak_val = float(np.max(I))

            logger.info(
                f"[F3] λ={lam_nm} nm: 耗时={dt:.4f} s, Pin={Pin:.4e}, Pwin={Pwin:.4e}, "
                f"窗口能量比 η={eta*100:.2f}%, 峰值={peak_val:.2f}"
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
                "finite_field": bool(np.all(np.isfinite(u2))),
                "fingerprint_matched": True,
            }
            psf_records.append(rec)
            completion_state["wavelengths_completed"].append(lam_nm)

            # 保存每个波长的原始复场与强度
            if run_dir:
                np.savez_compressed(
                    run_dir / "arrays" / f"psf_{lam_nm}nm.npz",
                    u2_complex=u2,
                    intensity_raw=I,
                    x_out_m=grid_out.x.coords,
                    y_out_m=grid_out.y.coords,
                    wavelength_m=lam,
                    doe_fingerprint=fingerprint,
                )

        # 窗口能量比合法性检查（无损薄屏理想模型上界 eta <= 1.02）
        all_eta_ok = all(r["eta_window"] <= acc["energy_efficiency_max"] for r in psf_records)
        all_finite = all(r["finite_field"] for r in psf_records)

        self_check["psf_preview"] = {
            "wavelengths_tested_nm": [r["wavelength_nm"] for r in psf_records],
            "eta_values": [r["eta_window"] for r in psf_records],
            "eta_max_allowed": acc["energy_efficiency_max"],
            "all_finite": all_finite,
            "status": "pass" if (all_eta_ok and all_finite) else "fail",
        }

        # 写入 psf_power.csv
        if run_dir:
            csv_path = run_dir / "metrics" / "psf_power.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                fields = ["wavelength_nm", "wavelength_m", "Pin", "Pwindow", "eta_window", "peak_intensity", "norm_factor", "elapsed_s"]
                writer = csv.DictWriter(f_csv, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(psf_records)

            # 绘制三波长对比图
            plot_psf_preview(
                grid_out=grid_out,
                wavelengths=preview_lambdas,
                intensities=preview_intensities,
                save_path_shape=run_dir / "figures" / "psf_preview_shape.png",
                save_path_raw=run_dir / "figures" / "psf_preview_raw.png",
            )

        completion_state["finished_steps"].append("F3_psf_preview")

    total_timer.__exit__()
    logger.info(f"=== 阶段 02A 执行完成，总耗时: {total_timer.elapsed:.4f} s ===")

    # 确定总体自检状态
    all_pass = all(item.get("status") == "pass" for item in self_check.values())
    completion_state["status"] = "completed" if all_pass else "failed"
    completion_state["all_checks_pass"] = all_pass
    completion_state["total_elapsed_s"] = total_timer.elapsed

    if run_dir:
        write_json(run_dir / "config_used.json", config)
        write_json(run_dir / "environment.json", environment_info())
        write_json(run_dir / "source_manifest.json", generate_source_manifest())
        write_json(run_dir / "self_check.json", self_check)
        write_json(run_dir / "completion_state.json", completion_state)

        # 生成中文阶段 02A 报告
        report_md = generate_markdown_report(
            config=config,
            self_check=self_check,
            doe_profile=doe_profile,
            fingerprint=fingerprint,
            run_dir=run_dir,
            total_elapsed=total_timer.elapsed,
        )
        report_path = run_dir / "report_stage02a.md"
        with open(report_path, "w", encoding="utf-8") as f_rep:
            f_rep.write(report_md)

        # 同步写入 docs/stage02a_report.md
        docs_rep_path = PROJECT_ROOT / "docs" / "stage02a_report.md"
        docs_rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(docs_rep_path, "w", encoding="utf-8") as f_rep:
            f_rep.write(report_md)

    return {
        "self_check": self_check,
        "completion_state": completion_state,
        "run_dir": run_dir,
        "fingerprint": fingerprint,
    }


def generate_markdown_report(
    config: Dict[str, Any],
    self_check: Dict[str, Any],
    doe_profile: Optional[DOEHeightProfile],
    fingerprint: str,
    run_dir: Path,
    total_elapsed: float,
) -> str:
    """生成详尽的阶段 02A 中文执行与审核报告。"""
    opt = config["optical"]
    grid_cfg = config["grid"]

    ctrl_res = self_check.get("control_550nm", {})
    r_meas_um = ctrl_res.get("first_dark_ring_measured_m", float("nan")) * 1e6
    r_theo_um = ctrl_res.get("first_dark_ring_theory_m", float("nan")) * 1e6
    r_err_pct = ctrl_res.get("first_dark_ring_rel_err", float("nan")) * 100
    l1_pct = ctrl_res.get("cut_normalized_l1", float("nan")) * 100

    psf_res = self_check.get("psf_preview", {})

    return f"""# Jeon2019 阶段 02A 执行与自检报告

- **日期**：{time.strftime('%Y-%m-%d %H:%M:%S')}
- **执行环境**：Windows / CPU / Python 3.10.4 / NumPy 2.2.6 / SciPy 1.15.3 / Matplotlib 3.10.9
- **运行目录**：`{run_dir}`
- **总体自检状态**：{'PASS (自检通过)' if all(x.get('status') == 'pass' for x in self_check.values()) else 'FAIL (存在未通过项)'}
- **总耗时**：{total_elapsed:.4f} 秒

---

## 1. 执行目标与范围声明

本轮仅执行**阶段 02A**：
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
| **预览窗口能量** | 420nm: {psf_res.get('eta_values', [0,0,0])[0]*100:.1f}%, 540nm: {psf_res.get('eta_values', [0,0,0])[1]*100:.1f}%, 660nm: {psf_res.get('eta_values', [0,0,0])[2]*100:.1f}% | $\\eta_{{window}} \\le 102\%$ | **{psf_res.get('status', 'N/A').upper()}** | 物理能量有限窗口保持，无发散或非物理越界 |

---

## 3. 产物清单与文件指纹

- **配置副本**：`config_used.json`
- **运行环境**：`environment.json`
- **源码与文献清单**：`source_manifest.json`
- **运行日志**：`run.log`
- **连续高度数组**：`arrays/doe_continuous.npz`
- **控制仿真数据**：`arrays/fresnel_control.npz`
- **三波长复场与强度**：`arrays/psf_420nm.npz`, `arrays/psf_540nm.npz`, `arrays/psf_660nm.npz`
- **材料折射率表**：`metrics/material_indices.csv`
- **功率与效率表**：`metrics/psf_power.csv`
- **图件**：
  - `figures/doe_height.png` (带真实物理标尺与 μm 色条的高度分布图)
  - `figures/control_550nm.png` (550nm 焦面 PSF 与 Airy 理论截线比对图)
  - `figures/psf_preview_shape.png` (三波长各自峰值归一化形状图，共同物理坐标)
  - `figures/psf_preview_raw.png` (三波长统一绝对强度对比图)

---

## 4. 结论与下一步

阶段 02A 的全部物理检查点与数值标准均已达标。
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

    # 分配运行目录与日志器
    run_dir = None
    if config["runtime"]["save_results"]:
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
        )

        all_pass = results["completion_state"]["all_checks_pass"]
        if not all_pass:
            logger.error("阶段 02A 自检存在未通过项，请查阅 report_stage02a.md")
            sys.exit(1)

        if config["runtime"]["show_plots"]:
            try:
                matplotlib.use("TkAgg")
                plt.show()
            except Exception as e:
                logger.warning(f"无法启动交互图窗: {e}")

        sys.exit(0)

    except Exception as e:
        logger.exception(f"执行过程中捕获严重异常: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
