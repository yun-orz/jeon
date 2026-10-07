# -*- coding: utf-8 -*-
"""Jeon 光学编码复现：02B-1 九波长对照 —— 主执行入口。

本批范围（严格限定）
--------------------
在**同一个探测器物理网格**上，为**两张固定 DOE 高度**各算九个入射波长，
共 18 组真实衍射复场与原始强度：

1. 固定 550nm 设计的传统 Fresnel DOE —— 作为色散对照；
2. Jeon 连续高度 N=3 DOE —— 作为正文图 3 的三翼旋转编码对象。

回答：旋转方向与角度如何随波长变化？形状尺寸是否变化？方窗与圆盘捕获能量是多少？
角度是否可靠？哪些结论还受窗口和采样限制？

**不做**：DOE 采样加密、加工量化、场景成像、重建、1×2 双孔径、沿轴级联、半圆裁切、
神经网络。N=3 是本论文基线，不改成 N=1。本批结果**不能**替代下一批的采样收敛证明。

命令行
------
::

    python -B main_stage02b.py                    # 默认完整执行（保存结果、不弹窗）
    python -B main_stage02b.py --no-save          # 不写任何项目文件
    python -B main_stage02b.py --only height      # 只做高度设计（partial）
    python -B main_stage02b.py --only control     # 只做单器件单波长（partial）
    python -B main_stage02b.py --only preview     # 只做 Jeon 九波长（partial）
    python -B main_stage02b.py --only analyze --from-run <run_dir>   # 只分析已有 run
    python -B main_stage02b.py --show-plots       # 尝试弹窗（不可用则回退并记录）
    python -B main_stage02b.py --run-dir <空目录>  # 显式指定运行目录

``--config`` 与 ``--run-dir`` 的相对路径一律按**项目根**（``__file__``）解析。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 必须先在导入 pyplot 之前完成命令行解析与配置读取（沿用 02A-R2 的 C2 约定） ----
_EARLY = None


def _early_parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default="config_stage02b.json")
    p.add_argument("--only", type=str, default="all",
                   choices=["all", "height", "control", "preview", "analyze"])
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--show-plots", action="store_true")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--from-run", type=str, default="")
    args, _ = p.parse_known_args(argv)
    return args


_EARLY = _early_parse() if __name__ == "__main__" else argparse.Namespace(
    config="config_stage02b.json", only="all", no_save=False, show_plots=False,
    run_dir="", from_run="")

from optics.coordinates import Axis1D, Grid2D                    # noqa: E402
from optics.doe import (                                          # noqa: E402
    DOEHeightProfile, compute_doe_transmission_field,
    design_conventional_fresnel_height, design_jeon2019_spiral_height,
    optical_path_difference_delta,
)
from optics.materials import (                                    # noqa: E402
    REFRACTIVE_INDEX_AIR, get_material_model_metadata,
    refractive_index_fused_silica,
)
from optics.propagation import fresnel_kernel_separable, sampling_diagnostics  # noqa: E402
from optics.psf_analysis import (                                 # noqa: E402
    c3_rotation_metric, height_fingerprint_from_arrays, psf_metrics,
    summarize_size_stability, unwrap_angles_over_reliable_segments,
)
from optics.runutil import environment_info, write_json           # noqa: E402
from optics.stage02_runtime import (                              # noqa: E402
    assert_fresh_run_dir, configure_plotting, effective_runtime,
    prepare_run_dir, resolve_project_path,
)

# 02B-1 的来源文件集合与验收项（显式身份列表，不复用 02A 的 required_checks）
SOURCE_FILES = [
    "main_stage02b.py",
    "config_stage02b.json",
    "optics/psf_analysis.py",
    "optics/doe.py",
    "optics/materials.py",
    "optics/coordinates.py",
    "optics/propagation.py",
    "optics/stage02_runtime.py",
    "optics/runutil.py",
    "optics/units.py",
    "optics/metrics.py",
    # 修正批 R8 追加：把本批的测试与说明也纳入源码清单，使 run 记录的指纹同时覆盖
    # “生产代码 + 测试代码”，避免出现“测试跑的是被改过的版本”这类歧义。
    # 注意：**不**把 docs/stage02b1_revision_report.md 这类“叙述本次 run 的报告”放进清单——
    # 报告里要写本次 run 的名字，而 run 又记录报告哈希，会形成无法收敛的自指循环。
    "tests/test_stage02b1.py",
    "tests/_stage02b1_support.py",
    "tests/test_stage02_entry.py",
    "tests/test_stage02_r2.py",
    "tests/test_negative_paths.py",
    "README.md",
]

DEVICE_KEYS = ("fresnel", "jeon")
# 完整 all 必须交付的图（R4：报告/图/表也是“完成”的一部分）
REQUIRED_FIGURE_NAMES = (
    "figure3_comparison_peak_normalized.png",
    "figure3_comparison_raw_shared_scale.png",
    "figure3_comparison_raw_per_device_scale.png",
    "rotation_angles.png",
    "rotation_reliability.png",
    "size_metrics.png",
    "energy_metrics.png",
)
PSF_ARRAY_KEYS_REQUIRED = (
    "u2_complex", "intensity_raw", "x_out_m", "y_out_m", "wavelength_m",
    "device_key", "device_fingerprint", "refractive_index",
    "Pin", "Pwindow", "include_global_phase", "amplitude",
    "eta_window", "peak_intensity",
    "propagation_distance_m", "design_focal_length_m",
)


# ======================================================================================
# 配置
# ======================================================================================
def _pos_finite(name: str, value: Any) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0):
        raise ValueError("配置项 %s 必须为有限正数，收到 %r" % (name, value))
    return v


def _strict_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("配置项 %s 必须为整数，收到 %r" % (name, value))
    if value < minimum:
        raise ValueError("配置项 %s 必须 >= %d" % (name, minimum))
    return int(value)


def load_and_validate_config(path: Path) -> Dict[str, Any]:
    """读取并**严格校验**配置：网格 n/间距/半宽必须自洽，输入网格必须覆盖完整孔径。"""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path_resolved"] = str(path)

    opt = cfg["optical"]
    D = _pos_finite("optical.diameter_m", opt["diameter_m"])
    f = _pos_finite("optical.focal_length_m", opt["focal_length_m"])
    z = _pos_finite("optical.distance_m", opt["distance_m"])
    _pos_finite("optical.amplitude", opt.get("amplitude", 1.0))
    N = _strict_int("optical.wings_N", opt["wings_N"])
    if N != 3:
        raise ValueError("本批要求 N=3（论文基线），收到 N=%r" % (N,))
    lmin = _pos_finite("optical.design_wavelength_min_m", opt["design_wavelength_min_m"])
    lmax = _pos_finite("optical.design_wavelength_max_m", opt["design_wavelength_max_m"])
    if not (lmin < lmax):
        raise ValueError("设计波长范围必须满足 min < max")
    lam0 = _pos_finite("optical.conventional_design_wavelength_m",
                       opt["conventional_design_wavelength_m"])
    lambdas = [float(x) for x in opt["incident_wavelengths_m"]]
    if len(lambdas) != 9:
        raise ValueError("本批要求恰好 9 个入射波长，收到 %d 个" % len(lambdas))
    if sorted(lambdas) != lambdas:
        raise ValueError("入射波长列表必须按升序给出")
    if len(set(round(x * 1e9) for x in lambdas)) != 9:
        raise ValueError("入射波长按整数 nm 必须互不相同")
    for lam in lambdas:
        refractive_index_fused_silica(lam)      # 越界会抛 ValueError

    for key, label in (("input", "输入"), ("output", "输出")):
        g = cfg["grid"][key]
        n = _strict_int("grid.%s.n" % key, g["n"], 3)
        d = _pos_finite("grid.%s.spacing_m" % key, g["spacing_m"])
        hw = _pos_finite("grid.%s.half_width_m" % key, g["half_width_m"])
        # 半宽定义 = (n // 2) * d（与 optics.coordinates.Axis1D.half_width 一致）
        expected_hw = (n // 2) * d
        if abs(expected_hw - hw) > 1e-12 * max(hw, 1e-12):
            raise ValueError(
                "grid.%s 的半宽与 n/间距不自洽：(n//2)*d=%.9g m，配置写 %.9g m"
                % (key, expected_hw, hw))

    g_in = cfg["grid"]["input"]
    if (g_in["n"] // 2) * g_in["spacing_m"] < D / 2.0:
        raise ValueError("输入网格半宽 %.6g m 未覆盖完整孔径半径 %.6g m"
                         % ((g_in["n"] // 2) * g_in["spacing_m"], D / 2.0))

    bands = cfg["rotation_bands"]
    for key in ("primary", "sensitivity"):
        b = bands[key]
        r0 = _pos_finite("rotation_bands.%s.r_min_m" % key, b["r_min_m"])
        r1 = _pos_finite("rotation_bands.%s.r_max_m" % key, b["r_max_m"])
        if not (r0 < r1):
            raise ValueError("环带 %s 必须满足 r_min < r_max" % key)
    roi = cfg["size_roi"]
    _pos_finite("size_roi.r_max_m", roi["r_max_m"])
    qs = [float(x) for x in roi["quantiles"]]
    if qs != [0.5, 0.1]:
        raise ValueError("本批形状阈值固定为峰值的 50%% 与 10%%，收到 %r" % (qs,))

    acc = cfg["acceptance"]
    _pos_finite("acceptance.energy_efficiency_max", acc["energy_efficiency_max"])
    a3 = float(acc["angle_reliability_threshold_A3"])
    if not (a3 >= 0):
        raise ValueError("A3 阈值不能为负")
    if _strict_int("acceptance.identity_set_expected_size",
                   acc["identity_set_expected_size"]) != 18:
        raise ValueError("身份集合期望大小必须为 18（2 器件 × 9 波长）")

    cfg["_derived"] = {
        "wavelengths_nm": [round(x * 1e9) for x in lambdas],
        "input_half_width_m": (g_in["n"] // 2) * g_in["spacing_m"],
        "output_half_width_m": (cfg["grid"]["output"]["n"] // 2)
                               * cfg["grid"]["output"]["spacing_m"],
    }
    return cfg


# ======================================================================================
# 指纹与清单
# ======================================================================================
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_source_manifest() -> Dict[str, Any]:
    entries = {}
    for rel in SOURCE_FILES:
        p = PROJECT_ROOT / rel
        if p.exists():
            entries[rel] = {"sha256": file_sha256(p), "size": p.stat().st_size}
        else:
            entries[rel] = {"sha256": None, "status": "missing"}
    hasher = hashlib.sha256()
    for rel in sorted(entries):
        hasher.update(("%s=%s\n" % (rel, entries[rel].get("sha256"))).encode("utf-8"))
    return {"stage": "stage02b1", "files": entries,
            "manifest_fingerprint": hasher.hexdigest(),
            "note": "按显式 02B-1 来源文件集合计算；不含 02A 专用集合"}


# ======================================================================================
# 传播
# ======================================================================================
def propagate_device(profile: DOEHeightProfile, wavelength_m: float, grid_in: Grid2D,
                     grid_out: Grid2D, include_global_phase: bool, amplitude: float,
                     h_offset: float, distance_m: float, logger: logging.Logger, tag: str
                     ) -> Dict[str, Any]:
    """固定高度器件在给定入射波长下的真实衍射复场（论文式(4) 的可分离等价实现）。

    R3 修正（2026-10-02）：传播距离 ``distance_m`` 由**调用方显式传入**并真正用于传播。
    旧实现取 ``profile.params['focal_length_m']``，于是 ``config.optical.distance_m``
    只被读取、仅用于诊断；用户改 z 后会得到“参数与实际传播不一致”的结果。
    """
    z = float(distance_m)
    if not (math.isfinite(z) and z > 0):
        raise ValueError("传播距离必须为有限正数，收到 %r" % (distance_m,))
    t0 = time.perf_counter()
    u1 = compute_doe_transmission_field(profile, wavelength_m,
                                       amplitude=amplitude, h_offset=h_offset)
    t_u1 = time.perf_counter() - t0
    t1 = time.perf_counter()
    u2 = fresnel_kernel_separable(field_in=u1, grid_in=grid_in,
                                 wavelength=wavelength_m, distance=z,
                                 x_out=grid_out.x.coords, y_out=grid_out.y.coords,
                                 include_global_phase=include_global_phase)
    t_prop = time.perf_counter() - t1
    I = np.abs(u2) ** 2
    logger.info("[%s] λ=%d nm：u1 %.3f s + 传播 %.3f s = %.3f s（实际 z=%.6g mm；"
                "器件设计焦距 f=%.6g mm）；|u2|max=%.6g，有限=%s",
                tag, round(wavelength_m * 1e9), t_u1, t_prop, t_u1 + t_prop,
                z * 1e3, float(profile.params["focal_length_m"]) * 1e3,
                float(np.max(np.abs(u2))),
                bool(np.all(np.isfinite(u2)) and np.all(np.isfinite(I))))
    return {"u1": u1, "u2": u2, "intensity": I, "distance_m": z,
            "design_focal_length_m": float(profile.params["focal_length_m"]),
            "t_u1_s": t_u1, "t_propagation_s": t_prop, "t_total_s": t_u1 + t_prop}


def field_health_check(u1: np.ndarray, u2: np.ndarray, intensity: np.ndarray,
                       grid_in: Grid2D, grid_out: Grid2D,
                       energy_max: float) -> Dict[str, Any]:
    """绝对幅度/功率/形状健康检查；**不使用任何归一化补救**。

    关键纪律：**只有全部检查通过的载荷才允许被复用**。失败时必须立刻抛出并停止，
    绝不能把失败的场留在缓存里让后续波长“精确复用控制点”——那会把一次失败
    悄悄扩散成多组看似正常的结果。
    """
    if u1.shape != grid_in.shape or u2.shape != grid_out.shape \
            or intensity.shape != grid_out.shape:
        return {"status": "fail", "reason": "形状与物理网格不符"}
    if not (np.all(np.isfinite(u1)) and np.all(np.isfinite(u2))
            and np.all(np.isfinite(intensity))):
        return {"status": "fail", "reason": "复场或强度包含 NaN/Inf"}
    pin = float(np.sum(np.abs(u1) ** 2) * grid_in.cell_area)
    pwin = float(np.sum(intensity) * grid_out.cell_area)
    peak = float(np.max(intensity))
    eta = pwin / pin if pin > 0 else float("nan")
    identity = bool(np.array_equal(intensity, np.abs(u2) ** 2))
    ok = (all(math.isfinite(v) for v in (pin, pwin, peak, eta))
          and pin > 0 and pwin > 0 and peak > 0 and 0.0 < eta <= float(energy_max)
          and identity)
    return {"status": "pass" if ok else "fail", "Pin": pin, "Pwindow": pwin,
            "peak_intensity": peak,
            "eta_window": (eta if ok else None),
            "eta_window_raw": eta,
            "intensity_identity_exact": identity,
            "reusable": ok,
            "reason": "" if ok else "非零/有限/模平方/绝对功率检查失败"}


# ======================================================================================
# 保存与重读
# ======================================================================================
def psf_array_name(device_key: str, wavelength_m: float) -> str:
    return "psf_%s_%dnm.npz" % (device_key, round(wavelength_m * 1e9))


def save_height_arrays(run_dir: Path, key: str, profile: DOEHeightProfile,
                       fingerprint: str, grid_in: Grid2D, cfg: Dict[str, Any]) -> Path:
    opt = cfg["optical"]
    p = run_dir / "arrays" / ("height_%s.npz" % key)
    np.savez_compressed(
        p,
        delta_h_m=profile.delta_h,
        mask=profile.mask,
        x_in_m=grid_in.x.coords,
        y_in_m=grid_in.y.coords,
        # R5：必须保存**精确**间距本身。用坐标差 (x[1]-x[0]) 反推会丢精度
        # （实测 1e-6 会变成 9.99999999999916e-7），导致独立重算的高度指纹不符。
        dx_m=np.array(float(grid_in.dx)),
        dy_m=np.array(float(grid_in.dy)),
        n_x=np.array(int(grid_in.x.n)),
        n_y=np.array(int(grid_in.y.n)),
        lambda_design_m=(profile.lambda_design if profile.lambda_design is not None
                         else np.zeros(grid_in.shape)),
        fingerprint=np.array(fingerprint),
        design_type=np.array(profile.design_type),
        diameter_m=np.array(float(opt["diameter_m"])),
        focal_length_m=np.array(float(opt["focal_length_m"])),
        wings_N=np.array(int(opt["wings_N"])),
        design_wavelength_min_m=np.array(float(opt["design_wavelength_min_m"])),
        design_wavelength_max_m=np.array(float(opt["design_wavelength_max_m"])),
        conventional_design_wavelength_m=np.array(
            float(opt["conventional_design_wavelength_m"])),
        h_offset_m=np.array(float(opt.get("h_offset_m", 0.0))),
        params_json=np.array(json.dumps(profile.params, sort_keys=True)),
        note=np.array("固定高度只生成一次；孔径外 mask=0 且高度为 0；坐标为 m"),
    )
    return p


def save_psf(run_dir: Path, device_key: str, wavelength_m: float,
             payload: Dict[str, Any]) -> Path:
    p = run_dir / "arrays" / psf_array_name(device_key, wavelength_m)
    np.savez_compressed(
        p,
        u2_complex=payload["u2"],
        intensity_raw=payload["intensity"],
        x_out_m=payload["x_out_m"],
        y_out_m=payload["y_out_m"],
        wavelength_m=np.array(float(wavelength_m)),
        device_key=np.array(device_key),
        device_fingerprint=np.array(payload["device_fingerprint"]),
        refractive_index=np.array(float(payload["refractive_index"])),
        Pin=np.array(float(payload["Pin"])),
        Pwindow=np.array(float(payload["Pwindow"])),
        include_global_phase=np.array(bool(payload["include_global_phase"])),
        amplitude=np.array(float(payload["amplitude"])),
        eta_window=np.array(float(payload["eta_window"])),
        peak_intensity=np.array(float(payload["peak_intensity"])),
        t_propagation_s=np.array(float(payload["t_propagation_s"])),
        propagation_distance_m=np.array(float(payload.get("propagation_distance_m",
                                                           payload.get("distance_m")))),
        design_focal_length_m=np.array(float(payload["design_focal_length_m"])),
        note=np.array("原始复场 complex128 与原始强度 float64，均未做峰值归一化；坐标为 m"),
    )
    return p


def verify_saved_identity_set(run_dir: Path, cfg: Dict[str, Any], grid_in: Grid2D,
                              grid_out: Grid2D, fingerprints: Dict[str, str],
                              profiles: Dict[str, DOEHeightProfile],
                              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """重读全部产物：身份集合严格等于 {两器件}×{九波长}，且**独立重算**物理量。

    R2 修正（2026-10-02）
    --------------------
    旧实现只检查保存的 ``Pwindow/Pin`` 是否落在 (0, 1.02]：把 ``Pin`` 与 ``Pwindow``
    **同时乘 2** 后复场与强度不变，重读仍然 pass（审核反例 ``forged_power_*``）。
    现在改为：

    1. 用**实际配置 + 固定高度 + 该入射波长**重新构造 ``u1``，独立算出 ``Pin``；
    2. 从**保存的强度与输出网格**重新积分 ``Pwindow``；
    3. 两者分别与保存值按明确容差比对（复现性容差，不是物理阈值）；
    4. 另行核对 η、峰值、振幅、n(λ)、全局相位、dtype、坐标、精确 λ、器件指纹；
    5. 高度文件按保存数组**重算内容指纹**，并核对设计 λ 图与参数；
    6. 缺字段/损坏数据给出**明确 fail 原因**，不依赖偶然的 ``KeyError``。

    容差说明：``Pin`` 由同一高度与同一 λ 重建，属**确定性复现**，故用极紧的相对容差
    （1e−12）；``Pwindow`` 由已保存强度重积分，同一 ``float64`` 累加顺序可能不同，
    故用相对容差 1e−9。
    """
    expected_lambdas = [float(x) for x in cfg["optical"]["incident_wavelengths_m"]]
    expected = set()
    for key in DEVICE_KEYS:
        for lam in expected_lambdas:
            expected.add((key, round(lam * 1e9)))
    if len(expected) != int(cfg["acceptance"]["identity_set_expected_size"]):
        raise ValueError("内部期望身份集合大小异常：%d" % len(expected))

    opt = cfg["optical"]
    amplitude_cfg = float(opt.get("amplitude", 1.0))
    h_offset_cfg = float(opt.get("h_offset_m", 0.0))
    gp_cfg = bool(cfg["propagation"]["include_global_phase"])
    distance_cfg = float(opt["distance_m"])
    tol_pin = 1e-12
    tol_pwin = 1e-9

    problems: List[str] = []
    seen: Dict[Tuple[str, int], Dict[str, Any]] = {}
    worst_pin_diff = 0.0
    worst_pwin_diff = 0.0
    arrays_dir = run_dir / "arrays"

    # 先重读高度文件：后续 PSF 核对要用它重算 u1
    loaded_heights: Dict[str, Dict[str, Any]] = {}
    for key in DEVICE_KEYS:
        hp = arrays_dir / ("height_%s.npz" % key)
        if not hp.exists():
            problems.append("缺少高度文件 %s" % hp.name)
            continue
        try:
            with np.load(hp, allow_pickle=False) as data:
                required = ("delta_h_m", "mask", "x_in_m", "y_in_m", "lambda_design_m",
                            "fingerprint", "design_type", "diameter_m", "focal_length_m",
                            "wings_N", "design_wavelength_min_m",
                            "design_wavelength_max_m",
                            "conventional_design_wavelength_m", "h_offset_m", "params_json")
                missing = [f for f in required if f not in data]
                if missing:
                    problems.append("%s 缺少字段：%s" % (hp.name, ", ".join(missing)))
                    continue
                payload = {f: data[f] for f in data.files}
        except Exception as exc:                                     # noqa: BLE001
            problems.append("%s 无法读取：%s: %s" % (hp.name, type(exc).__name__, exc))
            continue
        prof = profiles[key]
        dh = payload["delta_h_m"]
        mask = payload["mask"]
        if dh.shape != grid_in.shape or mask.shape != grid_in.shape:
            problems.append("%s 的高度/掩膜形状不符" % hp.name)
            continue
        if not (np.all(np.isfinite(dh)) and np.all(np.isfinite(mask))):
            problems.append("%s 含非有限高度或掩膜" % hp.name)
        if not np.array_equal(dh, prof.delta_h):
            problems.append("%s 的 delta_h 与实际使用高度不符" % hp.name)
        if not np.array_equal(mask, prof.mask):
            problems.append("%s 的 mask 与实际不符" % hp.name)
        if not np.array_equal(payload["x_in_m"], grid_in.x.coords) or \
                not np.array_equal(payload["y_in_m"], grid_in.y.coords):
            problems.append("%s 的输入坐标不符" % hp.name)
        if payload["lambda_design_m"].shape != grid_in.shape:
            problems.append("%s 的 lambda_design 形状不符" % hp.name)
        elif prof.lambda_design is not None and \
                not np.array_equal(payload["lambda_design_m"], prof.lambda_design):
            problems.append("%s 的 lambda_design 与实际设计不符" % hp.name)
        # 独立重算内容指纹（不信任文件里的 fingerprint 字段）
        try:
            params = json.loads(str(payload["params_json"]))
        except Exception as exc:                                     # noqa: BLE001
            problems.append("%s 的 params_json 无法解析：%s" % (hp.name, exc))
            params = None
        if params is not None:
            # 用**保存的精确间距**重建网格做独立指纹重算；旧 run 若缺字段会退化为
            # 坐标差并可能因此不符，此时如实报 fail 而不是放宽核对。
            saved_axis = {k: payload[k] for k in
                          ("x_in_m", "y_in_m", "dx_m", "dy_m", "n_x", "n_y")
                          if k in payload}
            g_saved = grid_in_from_saved_height(
                saved_axis, fallback_spacing=float(cfg["grid"]["input"]["spacing_m"]))
            if not np.array_equal(g_saved.x.coords, grid_in.x.coords) or \
                    not np.array_equal(g_saved.y.coords, grid_in.y.coords):
                problems.append("%s 重建的输入坐标与实际网格不符" % hp.name)
            recomputed = height_fingerprint_from_arrays(
                str(payload["design_type"]), g_saved, dh, mask, params)
            if recomputed != fingerprints[key]:
                problems.append("%s 独立重算的内容指纹与本次器件不符" % hp.name)
            if str(payload["fingerprint"]) != recomputed:
                problems.append("%s 保存的 fingerprint 字段与独立重算不一致" % hp.name)
        if abs(float(payload["h_offset_m"]) - h_offset_cfg) > 0:
            problems.append("%s 的 h_offset 与本次配置不符" % hp.name)
        loaded_heights[key] = {"delta_h_m": dh, "mask": mask}

    for key in DEVICE_KEYS:
        for lam in expected_lambdas:
            nm = round(lam * 1e9)
            p = arrays_dir / psf_array_name(key, lam)
            if not p.exists():
                problems.append("缺少数组文件 %s" % p.name)
                continue
            try:
                with np.load(p, allow_pickle=False) as data:
                    missing = [f for f in PSF_ARRAY_KEYS_REQUIRED if f not in data]
                    if missing:
                        problems.append("%s 缺少字段：%s" % (p.name, ", ".join(missing)))
                        continue
                    d = {f: data[f] for f in data.files}
            except Exception as exc:                                 # noqa: BLE001
                problems.append("%s 无法读取：%s: %s" % (p.name, type(exc).__name__, exc))
                continue

            ident = (str(d["device_key"]), int(round(float(d["wavelength_m"]) * 1e9)))
            if ident != (key, nm):
                problems.append("%s 身份不符：文件内 %r，期望 %r" % (p.name, ident, (key, nm)))
                continue
            if ident in seen:
                problems.append("身份重复：%r 出现在 %s 与 %s"
                                % (ident, seen[ident]["file"], p.name))
                continue
            if float(d["wavelength_m"]) != float(lam):
                problems.append("%s 的 wavelength_m 与请求值不精确相等" % p.name)
            if not np.array_equal(d["x_out_m"], grid_out.x.coords) or \
                    not np.array_equal(d["y_out_m"], grid_out.y.coords):
                problems.append("%s 的探测器坐标与本次输出网格不符" % p.name)
            if str(d["device_fingerprint"]) != fingerprints[key]:
                problems.append("%s 的器件指纹与本次设计不符" % p.name)
            u2 = d["u2_complex"]
            I = d["intensity_raw"]
            if u2.shape != grid_out.shape or I.shape != grid_out.shape:
                problems.append("%s 的复场/强度形状不符" % p.name)
                continue
            if u2.dtype != np.complex128:
                problems.append("%s 的复场 dtype=%s，应为 complex128" % (p.name, u2.dtype))
            if I.dtype != np.float64:
                problems.append("%s 的强度 dtype=%s，应为 float64" % (p.name, I.dtype))
            if not (np.all(np.isfinite(u2)) and np.all(np.isfinite(I))):
                problems.append("%s 含非有限值" % p.name)
            if not np.array_equal(I, np.abs(u2) ** 2):
                problems.append("%s 的 Iraw != |u2|^2" % p.name)

            # ---- 独立重算 Pin：用实际配置 + 固定高度 + 该 λ 重建 u1 ----
            if key in loaded_heights:
                u1_re = compute_doe_transmission_field(
                    profiles[key], lam, amplitude=amplitude_cfg, h_offset=h_offset_cfg)
                pin_re = float(np.sum(np.abs(u1_re) ** 2) * grid_in.cell_area)
            else:
                pin_re = None
            # ---- 独立重算 Pwindow：从保存强度与输出网格重积分 ----
            pwin_re = float(np.sum(I) * grid_out.cell_area)

            pin_saved = float(d["Pin"])
            pwin_saved = float(d["Pwindow"])
            eta_saved = float(d["eta_window"])
            # 记录独立重算的偏差（无论是否传 logger，都要进 JSON 供审核核对）
            if pin_re is not None and pin_re != 0:
                worst_pin_diff = max(worst_pin_diff,
                                     abs(pin_saved - pin_re) / abs(pin_re))
            if pwin_re != 0:
                worst_pwin_diff = max(worst_pwin_diff,
                                      abs(pwin_saved - pwin_re) / abs(pwin_re))
            if not (math.isfinite(pin_saved) and pin_saved > 0):
                problems.append("%s 的 Pin 非有限正数" % p.name)
            elif pin_re is not None and \
                    abs(pin_saved - pin_re) > tol_pin * abs(pin_re):
                problems.append(
                    "%s 的 Pin 与独立重算不符：保存 %.12e，重算 %.12e（相对差 %.3e）"
                    % (p.name, pin_saved, pin_re, abs(pin_saved - pin_re) / abs(pin_re)))
            if not math.isfinite(pwin_saved):
                problems.append("%s 的 Pwindow 非有限" % p.name)
            elif abs(pwin_saved - pwin_re) > tol_pwin * abs(pwin_re):
                problems.append(
                    "%s 的 Pwindow 与从保存强度重积分不符：保存 %.12e，重积分 %.12e"
                    "（相对差 %.3e）" % (p.name, pwin_saved, pwin_re,
                                       abs(pwin_saved - pwin_re) / abs(pwin_re)))
            if pin_saved > 0 and math.isfinite(pwin_saved):
                eta_re = pwin_saved / pin_saved
                if not math.isfinite(eta_saved) or abs(eta_saved - eta_re) > 1e-12 * abs(eta_re):
                    problems.append("%s 的 ηwindow 字段与 Pwindow/Pin 不一致" % p.name)
                if not (0.0 < eta_re <= float(cfg["acceptance"]["energy_efficiency_max"])):
                    problems.append("%s 的 ηwindow=%.6g 越界" % (p.name, eta_re))
            # 峰值
            if abs(float(d["peak_intensity"]) - float(np.max(I))) > \
                    1e-12 * float(np.max(I)):
                problems.append("%s 的 peak_intensity 字段与强度不符" % p.name)
            # n(λ)
            n_expect = float(refractive_index_fused_silica(lam))
            if abs(float(d["refractive_index"]) - n_expect) > 1e-14 * n_expect:
                problems.append("%s 的 refractive_index 与 n(λ) 不符" % p.name)
            # 振幅与全局相位
            if abs(float(d["amplitude"]) - amplitude_cfg) > 0:
                problems.append("%s 的 amplitude 与本次配置不符" % p.name)
            if bool(d["include_global_phase"]) != gp_cfg:
                problems.append("%s 的全局相位配置与本次不符" % p.name)
            # 传播距离必须等于本次配置 z（R3）；设计焦距必须等于器件设计 f
            if float(d["propagation_distance_m"]) != distance_cfg:
                problems.append(
                    "%s 记录的传播距离 %.12g m 与本次配置 z=%.12g m 不符"
                    % (p.name, float(d["propagation_distance_m"]), distance_cfg))
            f_design = float(profiles[key].params["focal_length_m"])
            if float(d["design_focal_length_m"]) != f_design:
                problems.append(
                    "%s 记录的设计焦距 %.12g m 与器件设计 f=%.12g m 不符"
                    % (p.name, float(d["design_focal_length_m"]), f_design))

            seen[ident] = {
                "file": p.name, "Pin": pin_saved, "Pwindow": pwin_saved,
                "Pin_recomputed": pin_re, "Pwindow_recomputed": pwin_re,
                "peak": float(np.max(I)),
                "eta_window": (pwin_saved / pin_saved) if pin_saved > 0 else None,
            }

    got = set(seen.keys())
    if got != expected:
        problems.append("身份集合不严格等于期望：多出 %r，缺少 %r"
                        % (sorted(got - expected), sorted(expected - got)))

    result = {"status": "pass" if not problems else "fail",
              "expected_identity_count": len(expected),
              "verified_identity_count": len(seen),
              "problems": problems,
              "identities": ["%s@%dnm" % k for k in sorted(seen)],
              "independent_power_check": True,
              "tolerances": {"Pin_relative": tol_pin, "Pwindow_relative": tol_pwin},
              "distance_m_used_in_recompute": distance_cfg,
              "worst_pin_relative_diff": worst_pin_diff,
              "worst_pwindow_relative_diff": worst_pwin_diff}
    if logger is not None:
        logger.info("[V] 独立重算功率：Pin 最差相对差 %.3e（容差 %.1e）；"
                    "Pwindow 最差相对差 %.3e（容差 %.1e）；重算使用 z=%.6g m",
                    worst_pin_diff, tol_pin, worst_pwin_diff, tol_pwin, distance_cfg)
    return result


# ======================================================================================
# 完成记录
# ======================================================================================
def _is_nan_like(x: Any) -> bool:
    try:
        return bool(isinstance(x, float) and not math.isfinite(x))
    except Exception:
        return False


def to_json_safe(obj: Any) -> Any:
    """把 NaN/Inf 换成字符串标记，避免 JSON 里伪装成有效数。"""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
        return obj
    if isinstance(obj, (np.floating,)):
        return to_json_safe(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    return obj


def required_checks_02b(only_mode: str, save_mode: bool = True) -> set:
    """02B-1 自己的必需检查集合（不复用 02A 的集合）。

    ``save_mode=False``（``--no-save``）时，只有**落盘才能验证**的项目
    （身份集合重读、保存后复核、功率范围复核）置为 ``not_applicable``，
    其余物理与设计检查照旧必须通过，因此 ``--no-save`` 也能得出 completed。
    """
    height = {"height_generated", "aperture_area", "local_phase_err",
              "device_design_lambda", "height_fingerprint_stable"}
    identity = {"identity_set_exact", "saved_arrays_reverified",
                "intensity_identity_exact", "power_in_range"}
    save_only = {"identity_set_exact", "saved_arrays_reverified",
                 "intensity_identity_exact", "power_in_range"}
    mapping = {
        "height": height,
        "control": height | {"control_single_field_health"},
        "preview": height | identity | {"jeon_nine_wavelengths"},
        "all": height | identity | {"control_single_field_health",
                                    "jeon_nine_wavelengths",
                                    "fresnel_nine_wavelengths", "analysis_completed"},
        "analyze": identity | {"analysis_completed"},
    }
    if only_mode not in mapping:
        raise ValueError("未知执行模式：%r" % (only_mode,))
    req = set(mapping[only_mode])
    # 最终验收必须包含报告与图表；预检由调用方明确排除交付项。
    if save_mode and only_mode == "all":
        req |= {"report_delivered", "required_artifacts_delivered"}
    if not save_mode:
        req -= save_only
    return req


def checks_pass_02b(checks: Dict[str, Any], only_mode: str, save_mode: bool = True,
                    include_delivery: bool = True) -> bool:
    required = required_checks_02b(only_mode, save_mode)
    if not include_delivery:
        required -= {"report_delivered", "required_artifacts_delivered"}
    return all(checks.get(name, {}).get("status") == "pass"
               for name in required)


# ======================================================================================
# 绘图
# ======================================================================================
def figure3_comparison(intensities: Dict[str, List[np.ndarray]],
                       wavelengths_nm: Sequence[int], grid_out: Grid2D,
                       path: Optional[Path], dpi: int, cmap: str,
                       mode: str = "peak_normalized"):
    """2×9 对照图：上行传统 Fresnel、下行 Jeon，列按 λ 升序；同 FOV/坐标轴/采样。

    三种口径，色标含义**不可混用**，图题中写明：

    ``peak_normalized``
        每幅除自身峰值（幅内标注“各幅除自身峰值”），只观察形状。
    ``raw_absolute_shared``
        18 幅共用**同一个绝对强度色标**（全部 18 幅的最大峰值）。
        这是最严格的 raw 共用色标，但本批两种器件的峰值量级差约 7 倍
        （传统 ~826 @540nm vs Jeon ~134 @570nm），因此 Jeon 行在此图里很暗；
        它用于横向比较绝对亮度，不用于观察 Jeon 的翼形。
    ``raw_per_device_shared``
        raw 原始强度，但**同一行内**共用该器件的最大峰值色标。
        它保留器件内的真实相对亮度变化（色散趋势），使 Jeon 行的三翼形状与旋转可见。
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(wavelengths_nm),
                             figsize=(2.05 * len(wavelengths_nm), 5.0))
    if mode == "peak_normalized":
        vmax_shared = None
    elif mode == "raw_absolute_shared":
        vmax_shared = max(float(np.max(I)) for key in DEVICE_KEYS
                          for I in intensities[key])
    elif mode == "raw_per_device_shared":
        vmax_shared = {key: max(float(np.max(I)) for I in intensities[key])
                       for key in DEVICE_KEYS}
    else:
        raise ValueError("未知对照图口径：%r" % (mode,))
    ext = grid_out.extent_pixel_um
    for row, key in enumerate(DEVICE_KEYS):
        for col, nm in enumerate(wavelengths_nm):
            ax = axes[row, col]
            I = intensities[key][col]
            if mode == "peak_normalized":
                img, vmax, extra = I / float(np.max(I)), 1.0, "\n(各幅除自身峰值)"
            elif mode == "raw_absolute_shared":
                img, vmax, extra = I, vmax_shared, "\n(raw, 18幅共用绝对色标)"
            else:
                img, vmax, extra = I, vmax_shared[key], "\n(raw, 同一行共用色标)"
            im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap,
                           vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_title("%s\n%d nm%s" % ("传统Fresnel" if key == "fresnel" else "Jeon N=3",
                                          nm, extra), fontsize=8)
            if col == 0:
                ax.set_ylabel("y (μm)")
            ax.set_xlabel("x (μm)", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    titles = {
        "peak_normalized": "峰值归一化（只观察形状，不改变 raw I 与 Pin）",
        "raw_absolute_shared": "raw 原始强度，18 幅共用同一绝对色标（比较绝对亮度）",
        "raw_per_device_shared": "raw 原始强度，同一行共用色标（观察色散趋势与三翼形状）",
    }
    fig.suptitle("02B-1 图3 对照：%s" % titles[mode], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return fig


def plot_rotation(metrics: Dict[str, Any], wavelengths_nm: Sequence[int],
                  path: Optional[Path], dpi: int):
    """角度与可靠性曲线：null 处**断开**，不画跨缺失的连线。"""
    import matplotlib.pyplot as plt
    # 角度曲线只针对 Jeon 器件：metrics["order"] 含 18 条身份，不能直接与 9 个波长配对
    jeon_keys = ["jeon@%dnm" % nm for nm in wavelengths_nm]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for band in ("primary", "sensitivity"):
        alphas = [metrics["records"][k]["rotation_bands"][band]["alpha_wrapped_deg"]
                  for k in jeon_keys]
        rel = [metrics["records"][k]["rotation_bands"][band]["reliable"]
               for k in jeon_keys]
        xs = [nm if r else None for nm, r in zip(wavelengths_nm, rel)]
        ys = [a if r else None for a, r in zip(alphas, rel)]
        axes[0].plot(xs, ys, "o-", label="%s（包裹角）" % band, lw=1.0, ms=4)
        unwrapped = metrics["unwrap"][band]["alpha_unwrapped_rad"]
        axes[1].plot(wavelengths_nm,
                     [None if v is None else math.degrees(v) for v in unwrapped],
                     "s-", label="%s（可靠段展开）" % band, lw=1.0, ms=4)
        a3 = [metrics["records"][k]["rotation_bands"][band]["A3"] for k in jeon_keys]
        axes[2].semilogy(wavelengths_nm, a3, "^-", label=band, lw=1.0, ms=4)
    axes[2].axhline(metrics["a3_min"], color="r", ls=":", lw=1.0,
                    label="A3 阈值 %.3g" % metrics["a3_min"])
    axes[0].set_title("包裹角 arg(C3)/3（120° 周期）\n不可靠点不连线", fontsize=9.5)
    axes[0].set_ylabel("角度 (度)")
    axes[1].set_title("可靠连续段内展开角", fontsize=9.5)
    axes[1].set_ylabel("展开角 (度)")
    axes[2].set_title("相对幅度 A3（角度可靠性的分子）", fontsize=9.5)
    axes[2].set_ylabel("A3")
    for ax in axes:
        ax.set_xlabel("入射波长 (nm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return fig


def plot_size_metrics(metrics: Dict[str, Any], wavelengths_nm: Sequence[int],
                      path: Optional[Path], dpi: int):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for key, style in (("fresnel", "o--"), ("jeon", "s-")):
        r50 = [metrics["records"]["%s@%dnm" % (key, nm)]["R50"]["radius_um"]
               for nm in wavelengths_nm]
        r80 = [metrics["records"]["%s@%dnm" % (key, nm)]["R80"]["radius_um"]
               for nm in wavelengths_nm]
        rq5 = [metrics["records"]["%s@%dnm" % (key, nm)]["shape"]["q0.5"]["r_equiv_um"]
               for nm in wavelengths_nm]
        rq1 = [metrics["records"]["%s@%dnm" % (key, nm)]["shape"]["q0.1"]["r_equiv_um"]
               for nm in wavelengths_nm]
        axes[0].plot(wavelengths_nm, r50, style, label="%s R50" % key, lw=1.0, ms=4)
        axes[0].plot(wavelengths_nm, r80, style, alpha=0.6,
                     label="%s R80" % key, lw=1.0, ms=4)
        axes[1].plot(wavelengths_nm, rq5, style, label="%s r_eq(50%%)" % key, lw=1.0, ms=4)
        axes[1].plot(wavelengths_nm, rq1, style, alpha=0.6,
                     label="%s r_eq(10%%)" % key, lw=1.0, ms=4)
    axes[0].set_title("绝对包围能量尺寸 R50/R80（未达到处断线）", fontsize=9.5)
    axes[0].set_ylabel("半径 (μm)")
    axes[1].set_title("峰值阈值面积等效半径", fontsize=9.5)
    axes[1].set_ylabel("r_eq (μm)")
    for ax in axes:
        ax.set_xlabel("入射波长 (nm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return fig


def plot_energy_metrics(metrics: Dict[str, Any], wavelengths_nm: Sequence[int],
                        path: Optional[Path], dpi: int):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for key, style in (("fresnel", "o--"), ("jeon", "s-")):
        eta = [metrics["records"]["%s@%dnm" % (key, nm)]["eta_window"]
               for nm in wavelengths_nm]
        e150 = [next(v["Eabs"] for v in
                     metrics["records"]["%s@%dnm" % (key, nm)]["abs_encircled"]
                     if abs(v["R_m"] - 1.5e-4) < 1e-15) for nm in wavelengths_nm]
        axes[0].plot(wavelengths_nm, eta, style, label="%s ηwindow" % key, lw=1.0, ms=4)
        axes[1].plot(wavelengths_nm, e150, style, label="%s Eabs(150μm)" % key,
                     lw=1.0, ms=4)
    axes[0].axhline(1.02, color="r", ls=":", lw=1.0, label="健康上界 1.02")
    axes[0].set_title("方窗捕获能量比 ηwindow（粗健康检查）", fontsize=9.5)
    axes[0].set_ylabel("ηwindow")
    axes[1].set_title("圆盘绝对包围能量 Eabs(150μm)=Pdisk/Pin", fontsize=9.5)
    axes[1].set_ylabel("Eabs")
    for ax in axes:
        ax.set_xlabel("入射波长 (nm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return fig


# ======================================================================================
# 主流程
# ======================================================================================
def run_stage02b1(cfg: Dict[str, Any], run_dir: Optional[Path], only_mode: str,
                  logger: logging.Logger, keep_figures: bool = False) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    save_mode = run_dir is not None
    required_checks_02b(only_mode, save_mode)
    # 非空 run 目录的拒绝检查已在 main() 中、**日志器启动之前**完成；
    # 此处不再重复，否则日志器刚写下的 run.log 会被自己判成“既有产物”。

    opt = cfg["optical"]
    gc = cfg["grid"]
    acc = cfg["acceptance"]
    prop = cfg["propagation"]
    include_global_phase = bool(prop["include_global_phase"])
    amplitude = float(opt.get("amplitude", 1.0))
    h_offset = float(opt.get("h_offset_m", 0.0))
    D = float(opt["diameter_m"])
    f = float(opt["focal_length_m"])
    z = float(opt["distance_m"])
    lambdas = [float(x) for x in opt["incident_wavelengths_m"]]
    wavelengths_nm = [round(x * 1e9) for x in lambdas]
    energy_max = float(acc["energy_efficiency_max"])
    a3_min = float(acc["angle_reliability_threshold_A3"])
    ambiguity_rad = float(acc["angle_ambiguity_wrapped_diff_rad"])

    grid_in = Grid2D(Axis1D(gc["input"]["n"], gc["input"]["spacing_m"], "x'"),
                     Axis1D(gc["input"]["n"], gc["input"]["spacing_m"], "y'"),
                     label="doe_input")
    grid_out = Grid2D(Axis1D(gc["output"]["n"], gc["output"]["spacing_m"], "x"),
                      Axis1D(gc["output"]["n"], gc["output"]["spacing_m"], "y"),
                      label="detector")

    checks: Dict[str, Any] = {}
    state: Dict[str, Any] = {
        "stage": "stage02b1",
        "batch_title": cfg.get("batch_title", ""),
        "status": "in_progress",
        "only_mode": only_mode,
        "calculation_status": "in_progress",
        "saving_status": "not_started",
        "analysis_status": "not_started",
        "finished_steps": [],
        "wavelengths_completed": [],
        "identity_records": [],
        "limitations": [],
        "created_timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    t_all = time.perf_counter()
    logger.info("=" * 80)
    logger.info("阶段 02B-1 开始：%s" % cfg.get("batch_title", ""))
    logger.info("D=%.4g mm，f=z=%.4g mm，N=%d，设计范围 [%.0f, %.0f] nm，传统设计 λ0=%.0f nm",
                D * 1e3, f * 1e3, int(opt["wings_N"]),
                float(opt["design_wavelength_min_m"]) * 1e9,
                float(opt["design_wavelength_max_m"]) * 1e9,
                float(opt["conventional_design_wavelength_m"]) * 1e9)
    logger.info("入射波长：%s nm", wavelengths_nm)
    logger.info("输入网格 %s dx=%.4g μm 半宽=%.4g μm；输出网格 %s dx=%.4g μm 半宽=%.4g μm",
                grid_in.shape, grid_in.dx * 1e6, grid_in.x.half_width * 1e6,
                grid_out.shape, grid_out.dx * 1e6, grid_out.x.half_width * 1e6)
    logger.info("全局相位 include_global_phase=%s；执行模式 only=%s；保存=%s",
                include_global_phase, only_mode, run_dir is not None)

    diag_min = sampling_diagnostics(grid_in, min(lambdas), z,
                                    grid_out.x.half_width, grid_out.y.half_width,
                                    dx_out=grid_out.dx, dy_out=grid_out.dy)
    diag_max = sampling_diagnostics(grid_in, max(lambdas), z,
                                    grid_out.x.half_width, grid_out.y.half_width,
                                    dx_out=grid_out.dx, dy_out=grid_out.dy)
    # 审核口径修正：如实打印**实际数值**，不再笼统写成"< 0.2 rad"。
    for _lam, _dg in ((min(lambdas), diag_min), (max(lambdas), diag_max)):
        logger.info("采样诊断（%.0f nm）：输入核步进 worst=%.6f rad（阈值 %.2f，ok=%s）；"
                    "输出表示步进 x=%.6f rad；三阶近轴高阶光程相位误差=%.6f rad。"
                    "这些只是保守提示，不是收敛证明",
                    _lam * 1e9, _dg["input_kernel_step_worst_rad"],
                    _dg["threshold_rad"], _dg["ok"],
                    _dg["output_quadratic_step_x_rad"],
                    _dg["third_order_phase_error_rad"])

    state["sampling_diagnostics"] = [
        {"wavelength_nm": round(min(lambdas) * 1e9),
         "input_kernel_step_worst_rad": diag_min["input_kernel_step_worst_rad"],
         "output_quadratic_step_x_rad": diag_min["output_quadratic_step_x_rad"],
         "third_order_phase_error_rad": diag_min["third_order_phase_error_rad"],
         "threshold_rad": diag_min["threshold_rad"], "ok": diag_min["ok"]},
        {"wavelength_nm": round(max(lambdas) * 1e9),
         "input_kernel_step_worst_rad": diag_max["input_kernel_step_worst_rad"],
         "output_quadratic_step_x_rad": diag_max["output_quadratic_step_x_rad"],
         "third_order_phase_error_rad": diag_max["third_order_phase_error_rad"],
         "threshold_rad": diag_max["threshold_rad"], "ok": diag_max["ok"]},
    ]
    if run_dir is not None:
        write_json(run_dir / "config_effective.json", to_json_safe(cfg))
        write_json(run_dir / "environment.json",
                   environment_info({"stage": "stage02b1", "only_mode": only_mode}))
        write_json(run_dir / "source_manifest.json", generate_source_manifest())
        write_json(run_dir / "checkpoint.json", to_json_safe(state))

    profiles: Dict[str, DOEHeightProfile] = {}
    fingerprints: Dict[str, str] = {}
    intensities: Dict[str, List[np.ndarray]] = {k: [] for k in DEVICE_KEYS}
    records: Dict[str, Dict[str, Any]] = {}
    active_figures: List[Any] = []

    try:
        # ------------------------------------------------------------------ H: 高度
        if only_mode in ("all", "height", "control", "preview"):
            logger.info("--- [H] 固定高度设计（只生成一次，孔径外 mask=0） ---")
            t = time.perf_counter()
            profiles["jeon"] = design_jeon2019_spiral_height(
                grid_in, D, f, int(opt["wings_N"]),
                float(opt["design_wavelength_min_m"]),
                float(opt["design_wavelength_max_m"]))
            fingerprints["jeon"] = profiles["jeon"].compute_fingerprint()
            profiles["fresnel"] = design_conventional_fresnel_height(
                grid_in, D, f, float(opt["conventional_design_wavelength_m"]))
            fingerprints["fresnel"] = profiles["fresnel"].compute_fingerprint()
            logger.info("[H] 完成（%.3f s）：Jeon 指纹 %s，传统 指纹 %s",
                        time.perf_counter() - t, fingerprints["jeon"][:16],
                        fingerprints["fresnel"][:16])

            area_ok = True
            exact_area = math.pi * (D / 2.0) ** 2
            for key, prof in profiles.items():
                sampled = float(np.sum(prof.mask) * grid_in.cell_area)
                rel = abs(sampled - exact_area) / exact_area
                ok = rel <= float(acc["aperture_area_rel_err_max"])
                area_ok = area_ok and ok
                logger.info("[H] %-8s 孔径离散面积=%.6e m²（理想 %.6e，相对差 %.3e，阈值 %.3g）",
                            key, sampled, exact_area, rel, acc["aperture_area_rel_err_max"])
            checks["aperture_area"] = {"status": "pass" if area_ok else "fail",
                                       "rel_err_max": float(acc["aperture_area_rel_err_max"])}

            phase_ok = True
            for key, prof in profiles.items():
                if prof.lambda_design is None:
                    phase_ok = False
                    continue
                mb = prof.mask.astype(bool)
                X, Y = grid_in.meshgrid()
                delta = optical_path_difference_delta(np.hypot(X, Y), f)
                n_des = refractive_index_fused_silica(prof.lambda_design)
                dn = n_des - REFRACTIVE_INDEX_AIR
                lp = 2.0 * np.pi * (delta[mb] + dn[mb] * prof.delta_h[mb]) / prof.lambda_design[mb]
                err = float(np.max(np.abs(np.exp(1j * lp) - 1.0)))
                ok = err <= float(acc["local_phase_err_max"])
                phase_ok = phase_ok and ok
                logger.info("[H] %-8s 局部聚焦相位恒等最大相量误差=%.3e（阈值 %.3g）",
                            key, err, acc["local_phase_err_max"])
            checks["local_phase_err"] = {"status": "pass" if phase_ok else "fail"}

            # 设计 λ 图与高度范围
            for key, prof in profiles.items():
                mb = prof.mask.astype(bool)
                hv = prof.delta_h[mb]
                ld = prof.lambda_design[mb]
                logger.info("[H] %-8s Δh∈[%.6g, %.6g] m（应位于 [-λd/(n-1), 0]）；"
                            "设计 λ∈[%.1f, %.1f] nm；原点 Δh=%.3g m",
                            key, float(hv.min()), float(hv.max()),
                            float(ld.min()) * 1e9, float(ld.max()) * 1e9,
                            float(prof.delta_h[grid_in.y.n // 2, grid_in.x.n // 2]))
                checks["height_generated_%s" % key] = {"status": "pass"}
            checks["height_generated"] = {"status": "pass"}
            checks["device_design_lambda"] = {
                "status": "pass",
                "jeon_lambda_design_nm": [float(profiles["jeon"].lambda_design.min()) * 1e9,
                                          float(profiles["jeon"].lambda_design.max()) * 1e9],
                "conventional_lambda0_nm":
                    float(opt["conventional_design_wavelength_m"]) * 1e9,
            }
            checks["height_fingerprint_stable"] = {
                "status": "pass", "jeon": fingerprints["jeon"],
                "fresnel": fingerprints["fresnel"],
                "note": "高度只生成一次；指纹在全部 18 组传播与重读中必须一致"}

            if run_dir is not None:
                save_height_arrays(run_dir, "jeon", profiles["jeon"],
                                   fingerprints["jeon"], grid_in, cfg)
                save_height_arrays(run_dir, "fresnel", profiles["fresnel"],
                                   fingerprints["fresnel"], grid_in, cfg)
            state["finished_steps"].append("H_height_design")
        else:
            for name in required_checks_02b("height"):
                checks[name] = {"status": "not_run", "reason": "excluded_by_only_mode"}

        # ------------------------------------------------------------------ C: 单场控制
        # 先用一个器件一个波长验证成本与健康，失败立即停止，不做归一化补救
        need_control = only_mode in ("all", "control")
        if need_control:
            lam_ctrl = 540e-9
            if lam_ctrl not in lambdas:
                raise RuntimeError("540nm 必须在本批入射波长列表内作为单场试算点")
            logger.info("--- [C] 单场试算（Jeon，%.0f nm）：记录耗时/尺寸/能量健康 ---",
                        lam_ctrl * 1e9)
            payload = propagate_device(profiles["jeon"], lam_ctrl, grid_in, grid_out,
                                      include_global_phase, amplitude, h_offset, z,
                                      logger, "C")
            health = field_health_check(payload["u1"], payload["u2"], payload["intensity"],
                                        grid_in, grid_out, energy_max)
            logger.info("[C] 健康检查：%s（ηwindow=%s，形状 %s，%.1f MB）",
                        health["status"],
                        "null" if health["eta_window"] is None
                        else "%.6f" % health["eta_window"],
                        payload["u2"].shape, payload["u2"].nbytes / 2**20)
            if health["status"] != "pass" or not health.get("reusable"):
                # 失败立即停止；**不**把失败载荷放进可复用缓存，也不做归一化补救
                checks["control_single_field_health"] = dict(health, status="fail")
                raise RuntimeError("单场试算未通过物理健康检查：%s" % health["reason"])
            checks["control_single_field_health"] = dict(health, status="pass")
            if payload["t_total_s"] > 120.0:
                logger.warning("[C] 单场耗时 %.1f s 超过 120 s 诊断触发参考值，"
                               "请检查矩阵形状/BLAS 线程/重复计算", payload["t_total_s"])
            state["finished_steps"].append("C_single_field_control")
        else:
            checks["control_single_field_health"] = {
                "status": "not_run", "reason": "excluded_by_only_mode"}

        # ------------------------------------------------------------------ P: 18 组
        do_preview = only_mode in ("all", "preview")
        if do_preview:
            logger.info("--- [P] 正式计算 %d 组衍射场（2 器件 × 9 波长）---",
                        int(acc["identity_set_expected_size"]))
            # 精确复用控制点：540nm 的 Jeon 场不再重算
            reuse: Dict[str, Dict[str, Any]] = {}
            if need_control:
                reuse["jeon@%d" % round(lam_ctrl * 1e9)] = payload

            for key in DEVICE_KEYS:
                prof = profiles[key]
                intensities[key] = []
                for lam in lambdas:
                    nm = round(lam * 1e9)
                    ident = "%s@%dnm" % (key, nm)
                    cache_key = "%s@%d" % (key, nm)
                    t0 = time.perf_counter()
                    if cache_key in reuse:
                        pay = reuse[cache_key]
                        logger.info("[P] %s 精确复用控制点结果（不重算）", ident)
                    else:
                        pay = propagate_device(prof, lam, grid_in, grid_out,
                                               include_global_phase, amplitude,
                                               h_offset, z, logger, "P")
                    health = field_health_check(pay["u1"], pay["u2"], pay["intensity"],
                                                grid_in, grid_out, energy_max)
                    if health["status"] != "pass" or not health.get("reusable"):
                        # 失败立即停止：不缓存、不归一化、不让后续波长复用失败载荷
                        state["failure_identity"] = ident
                        raise RuntimeError("%s 物理健康检查失败：%s" % (ident, health["reason"]))
                    fp_now = prof.compute_fingerprint()
                    if fp_now != fingerprints[key]:
                        raise RuntimeError("%s 计算过程中器件指纹发生变化" % ident)
                    n_lam = float(refractive_index_fused_silica(lam))
                    rec = {
                        "identity": ident, "device_key": key,
                        "wavelength_nm": nm, "wavelength_m": lam,
                        "device_fingerprint": fingerprints[key],
                        "refractive_index": n_lam,
                        "Pin": health["Pin"], "Pwindow": health["Pwindow"],
                        "eta_window": health["eta_window"],
                        "peak_intensity": health["peak_intensity"],
                        "intensity_identity_exact": health["intensity_identity_exact"],
                        "t_propagation_s": pay["t_propagation_s"],
                        "t_total_s": pay["t_total_s"],
                        "reused_control_point": cache_key in reuse,
                    }
                    if run_dir is not None:
                        save_psf(run_dir, key, lam, {
                            "u2": pay["u2"], "intensity": pay["intensity"],
                            "x_out_m": grid_out.x.coords, "y_out_m": grid_out.y.coords,
                            "device_fingerprint": fingerprints[key],
                            "refractive_index": n_lam,
                            "Pin": health["Pin"], "Pwindow": health["Pwindow"],
                            "include_global_phase": include_global_phase,
                            "amplitude": amplitude, "eta_window": health["eta_window"],
                            "peak_intensity": health["peak_intensity"],
                            "propagation_distance_m": pay["distance_m"],
                            "design_focal_length_m": pay["design_focal_length_m"],
                            "t_propagation_s": pay["t_propagation_s"]})
                        rec["saved"] = True
                    records[ident] = rec
                    intensities[key].append(pay["intensity"])
                    state["identity_records"].append(
                        {k: rec[k] for k in ("identity", "device_key", "wavelength_nm",
                                             "eta_window", "peak_intensity", "saved")
                         if k in rec})
                    if nm not in state["wavelengths_completed"]:
                        state["wavelengths_completed"].append(nm)
                    if run_dir is not None:
                        write_json(run_dir / "checkpoint.json", to_json_safe(state))
                    logger.info("[P] %s 完成：ηwindow=%.6f 峰值=%.6g（累计 %.1f s）",
                                ident, rec["eta_window"], rec["peak_intensity"],
                                time.perf_counter() - t0)
            checks["jeon_nine_wavelengths"] = {
                "status": "pass" if len(intensities["jeon"]) == 9 else "fail",
                "count": len(intensities["jeon"])}
            checks["fresnel_nine_wavelengths"] = {
                "status": "pass" if len(intensities["fresnel"]) == 9 else "fail",
                "count": len(intensities["fresnel"])}
            state["finished_steps"].append("P_eighteen_fields")
            state["calculation_status"] = "completed"
        else:
            for name in ("jeon_nine_wavelengths", "fresnel_nine_wavelengths"):
                checks[name] = {"status": "not_run", "reason": "excluded_by_only_mode"}
            state["calculation_status"] = "not_run"

        # ------------------------------------------------------------------ V: 重读
        if run_dir is not None and do_preview:
            logger.info("--- [V] 保存后重读全部产物，核对身份集合与内容 ---")
            verification = verify_saved_identity_set(run_dir, cfg, grid_in, grid_out,
                                                     fingerprints, profiles,
                                                     logger=logger)
            logger.info("[V] 身份集合：期望 %d，实读 %d；问题 %d 条",
                        verification["expected_identity_count"],
                        verification["verified_identity_count"],
                        len(verification["problems"]))
            for msg in verification["problems"]:
                logger.error("[V] %s", msg)
            checks["identity_set_exact"] = {
                "status": "pass" if verification["verified_identity_count"]
                == verification["expected_identity_count"] else "fail",
                "expected": verification["expected_identity_count"],
                "verified": verification["verified_identity_count"]}
            checks["saved_arrays_reverified"] = {
                "status": verification["status"], "problems": verification["problems"],
                # R2 证据：独立重算功率与保存值的最差相对偏差（进 JSON 供审核核对）
                "worst_pin_relative_diff": verification.get("worst_pin_relative_diff"),
                "worst_pwindow_relative_diff":
                    verification.get("worst_pwindow_relative_diff"),
                "tolerances": verification.get("tolerances"),
                "independent_power_check": verification.get("independent_power_check"),
                "distance_m_used_in_recompute":
                    verification.get("distance_m_used_in_recompute")}
            checks["intensity_identity_exact"] = {
                "status": "pass" if not any("Iraw" in m for m in verification["problems"])
                else "fail"}
            eta_ok = not any("ηwindow" in m for m in verification["problems"])
            checks["power_in_range"] = {"status": "pass" if eta_ok else "fail"}
            state["saving_status"] = "completed" if verification["status"] == "pass" \
                else "failed"
            if verification["status"] != "pass":
                raise RuntimeError("保存后重读未通过：%s" % "; ".join(verification["problems"]))
            state["finished_steps"].append("V_verify_saved")
        else:
            for name in ("identity_set_exact", "saved_arrays_reverified",
                         "intensity_identity_exact", "power_in_range"):
                checks[name] = {"status": "not_applicable",
                                "reason": "未落盘（--no-save）：该检查只在保存后可验证"}
            state["saving_status"] = "not_applicable"

        # ------------------------------------------------------------------ A: 分析
        metrics: Dict[str, Any] = {"order": [], "records": {}}
        if do_preview:
            logger.info("--- [A] 定量分析（角度/尺寸/能量）---")
            for key in DEVICE_KEYS:
                for idx, lam in enumerate(lambdas):
                    nm = round(lam * 1e9)
                    ident = "%s@%dnm" % (key, nm)
                    I = intensities[key][idx]
                    m = psf_metrics(I, grid_out, records[ident]["Pin"],
                                    records[ident]["Pwindow"],
                                    cfg["rotation_bands"], cfg["size_roi"],
                                    cfg["energy"], a3_min, ambiguity_rad)
                    m["identity"] = ident
                    m["device_key"] = key
                    m["wavelength_nm"] = nm
                    metrics["records"][ident] = m
                    if ident not in metrics["order"]:
                        metrics["order"].append(ident)

            metrics["unwrap"] = {}
            for band in ("primary", "sensitivity"):
                alphas = [metrics["records"]["jeon@%dnm" % nm]["rotation_bands"][band]
                          ["alpha_wrapped_rad"] for nm in wavelengths_nm]
                rel = [metrics["records"]["jeon@%dnm" % nm]["rotation_bands"][band]
                       ["reliable"] for nm in wavelengths_nm]
                u = unwrap_angles_over_reliable_segments(
                    lambdas, alphas, rel, max_abs_wrapped_diff_rad=ambiguity_rad)
                metrics["unwrap"][band] = u
                logger.info("[A] %s：可靠 %d/9；连续段 %d 个；歧义对 %d",
                            band, u["reliable_count"], len(u["segments"]),
                            len(u["ambiguous_pairs"]))
                for seg in u["segments"]:
                    logger.info("[A]   %s 段 %s nm：展开角 %.4f° → %.4f°（%s，跨度 %.4f°）",
                                band, seg["wavelengths_nm"],
                                math.degrees(seg["alpha_unwrapped_rad"][0]),
                                math.degrees(seg["alpha_unwrapped_rad"][-1]),
                                seg["direction"], seg["span_deg"])
                for ap in u["ambiguous_pairs"]:
                    logger.warning("[A]   角度采样歧义：%d→%d nm，包裹差 %.4fπ",
                                   ap["from_wavelength_nm"], ap["to_wavelength_nm"],
                                   ap["wrapped_diff_over_pi"]
                                   if "wrapped_diff_over_pi" in ap
                                   else ap["wrapped_triple_diff_over_pi"])
                if u["has_ambiguity"]:
                    state["limitations"].append(
                        "%s：相邻波长三倍相位包裹差接近 π，九点角度采样存在歧义，"
                        "不能声称旋转方向已唯一确定（建议下一批局部加密波长）" % band)

            logger.info("[A] 传统 Fresnel 近圆对称，C3 低、角度 undefined 属合理对照")

            stability = {}
            for key in DEVICE_KEYS:
                ident_list = ["%s@%dnm" % (key, nm) for nm in wavelengths_nm]
                stability[key] = {
                    "R50_um": summarize_size_stability(
                        [metrics["records"][i]["R50"]["radius_um"] for i in ident_list]),
                    "R80_um": summarize_size_stability(
                        [metrics["records"][i]["R80"]["radius_um"] for i in ident_list]),
                    "r_eq_q0.5_um": summarize_size_stability(
                        [metrics["records"][i]["shape"]["q0.5"]["r_equiv_um"]
                         for i in ident_list]),
                    "r_eq_q0.1_um": summarize_size_stability(
                        [metrics["records"][i]["shape"]["q0.1"]["r_equiv_um"]
                         for i in ident_list]),
                }
                for name, s in stability[key].items():
                    logger.info("[A] %-8s %-12s 有效样本 %d，范围 [%s, %s] μm，均值 %s，CV %s",
                                key, name, s["n_valid"],
                                "None" if s["min"] is None else "%.4f" % s["min"],
                                "None" if s["max"] is None else "%.4f" % s["max"],
                                "None" if s["mean"] is None else "%.4f" % s["mean"],
                                "None" if s["cv"] is None else "%.4f" % s["cv"])
            metrics["size_stability"] = stability
            metrics["a3_min"] = a3_min
            metrics["ambiguity_rad"] = ambiguity_rad
            metrics["definitions"] = {
                "identity_set": "{fresnel,jeon} × 9 个入射波长，共 18 组",
                "bands": cfg["rotation_bands"],
                "size_roi": cfg["size_roi"],
                "energy": cfg["energy"],
            }
            state["analysis_status"] = "completed"
            checks["analysis_completed"] = {"status": "pass"}
            state["finished_steps"].append("A_analysis")

            # R6 修正（2026-10-02）：绘图与保存**解耦**。
            # 旧实现把所有绘图放在 ``if run_dir is not None`` 分支内，于是 ``--no-save``
            # 即使请求了 ``--show-plots`` 也不会产生任何 figure。现在：
            #   * 只要 **需要保存** 或 **需要显示** 就绘图；
            #   * 不保存时把保存路径传 None（图只在内存中）；
            #   * 两者都不需要才跳过绘图。
            need_plot = (run_dir is not None) or bool(keep_figures)
            if run_dir is not None:
                write_json(run_dir / "metrics" / "psf_metrics.json",
                           to_json_safe(metrics))
                write_csv_metrics(run_dir / "metrics" / "psf_metrics.csv",
                                  metrics, wavelengths_nm)
            if need_plot:
                figures = []
                figures.append(("figure3_comparison_peak_normalized.png",
                                lambda p: figure3_comparison(
                                    intensities, wavelengths_nm, grid_out, p,
                                    cfg["plots"]["dpi"], cfg["plots"]["colormap_intensity"],
                                    "peak_normalized")))
                figures.append(("figure3_comparison_raw_shared_scale.png",
                                lambda p: figure3_comparison(
                                    intensities, wavelengths_nm, grid_out, p,
                                    cfg["plots"]["dpi"], cfg["plots"]["colormap_intensity"],
                                    "raw_absolute_shared")))
                figures.append(("figure3_comparison_raw_per_device_scale.png",
                                lambda p: figure3_comparison(
                                    intensities, wavelengths_nm, grid_out, p,
                                    cfg["plots"]["dpi"], cfg["plots"]["colormap_intensity"],
                                    "raw_per_device_shared")))
                figures.append(("rotation_angles.png",
                                lambda p: plot_rotation(metrics, wavelengths_nm, p,
                                                        cfg["plots"]["dpi"])))
                figures.append(("size_metrics.png",
                                lambda p: plot_size_metrics(metrics, wavelengths_nm, p,
                                                            cfg["plots"]["dpi"])))
                figures.append(("energy_metrics.png",
                                lambda p: plot_energy_metrics(metrics, wavelengths_nm, p,
                                                              cfg["plots"]["dpi"])))
                figures.append(("rotation_reliability.png",
                                lambda p: plot_rotation_reliability(
                                    metrics, wavelengths_nm, p, cfg["plots"]["dpi"])))
                for fname, fn in figures:
                    target = (run_dir / "figures" / fname) if run_dir is not None else None
                    fig = fn(target)
                    if keep_figures:
                        active_figures.append(fig)
                    else:
                        plt.close(fig)
                state["figures_created"] = [f for f, _ in figures]
                state["figures_saved"] = (run_dir is not None)
            else:
                state["figures_created"] = []
                state["figures_saved"] = False
        else:
            checks["analysis_completed"] = {"status": "not_run",
                                            "reason": "excluded_by_only_mode"}
            metrics = {"order": [], "records": {}, "note": "未执行分析"}

        # ------------------------------------------------------------------
        # R4 修正（2026-10-02）：报告与必需产物也属于“本批完成”的一部分。
        # 旧实现先写 completed/all_checks_pass=true，再由 main() 生成报告；
        # 子模式（如 --only height）因此会留下 completed，却在报告里 KeyError 崩溃。
        # 现在：先定状态，再在 try 内生成报告；报告/图/表缺失或异常一律 failed 且非零退出。
        # ------------------------------------------------------------------
        # 先给出“除报告与产物交付之外”的状态，报告本身读得到 all_checks_pass/status
        state["checks"] = checks
        prelim_pass = checks_pass_02b(checks, only_mode, save_mode, include_delivery=False)
        if prelim_pass and only_mode == "all":
            state["status"] = "completed"
        elif prelim_pass:
            state["status"] = "partial"
            state["partial_reason"] = ("子模式 %s 自身检查通过，但未执行本批全部内容"
                                       % only_mode)
        else:
            state["status"] = "partial" if any(
                v.get("status") == "pass" for v in checks.values()) else "failed"
        state["all_checks_pass"] = prelim_pass

        state["delivered_artifacts"] = []
        if run_dir is not None:
            try:
                env_for_report = environment_info(
                    {"stage": "stage02b1", "only_mode": only_mode})
                report_text = build_report(
                    cfg, env_for_report,
                    {"state": state, "metrics": metrics, "checks": checks,
                     "records": records, "figures": active_figures, "intensities": intensities},
                    run_dir)
                (run_dir / "report_stage02b1.md").write_text(report_text, encoding="utf-8")
                state["delivered_artifacts"].append("report_stage02b1.md")
            except Exception as exc:                                 # noqa: BLE001
                logger.error("报告生成失败：%s: %s", type(exc).__name__, exc)
                logger.error("\n%s", traceback.format_exc())
                checks["report_delivered"] = {
                    "status": "fail",
                    "reason": "报告生成失败：%s: %s" % (type(exc).__name__, exc)}
                raise
            checks["report_delivered"] = {"status": "pass",
                                          "file": "report_stage02b1.md"}
            # 完整 all 必须真的交付了必需图表
            if only_mode == "all":
                wanted_files = ["metrics/psf_metrics.csv", "metrics/psf_metrics.json"]
                wanted_files += ["figures/" + f for f in REQUIRED_FIGURE_NAMES]
                missing = [f for f in wanted_files
                           if not (run_dir / f).exists()]
                checks["required_artifacts_delivered"] = {
                    "status": "pass" if not missing else "fail",
                    "missing": missing,
                    "checked": wanted_files}
                if missing:
                    state["status"] = "failed"
                    state["all_checks_pass"] = False
                    raise RuntimeError("完整运行缺少必需产物：%s" % ", ".join(missing))
                state["delivered_artifacts"].append("metrics+figures")
        else:
            checks["report_delivered"] = {
                "status": "not_applicable",
                "reason": "未落盘（--no-save）：报告只在保存时生成"}

        # 用最终 checks 重算一次，确保报告交付项也参与判定
        state["checks"] = checks
        state["all_checks_pass"] = checks_pass_02b(checks, only_mode, save_mode)
        if not state["all_checks_pass"] and only_mode == "all":
            state["status"] = "failed" if any(
                v.get("status") == "fail" for v in checks.values()) else state["status"]
        state["elapsed_s"] = time.perf_counter() - t_all
    except Exception as exc:                                     # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("阶段 02B-1 执行失败：%s: %s", type(exc).__name__, exc)
        logger.error("\n%s", tb)
        state["status"] = "failed"
        state["error"] = "%s: %s" % (type(exc).__name__, exc)
        state["traceback"] = tb
        state["checks"] = checks
        state["all_checks_pass"] = False
        state["elapsed_s"] = time.perf_counter() - t_all
        if run_dir is not None:
            write_json(run_dir / "completion.json", to_json_safe(state))
            write_json(run_dir / "checkpoint.json", to_json_safe(state))
        raise
    finally:
        if run_dir is not None:
            write_json(run_dir / "completion.json", to_json_safe(state))
            write_json(run_dir / "checkpoint.json", to_json_safe(state))

    logger.info("=" * 80)
    logger.info("阶段 02B-1 结束：状态=%s，全部必需检查通过=%s，耗时 %.2f s",
                state["status"], state["all_checks_pass"], state["elapsed_s"])
    return {"state": state, "metrics": metrics, "checks": checks,
            "figures": active_figures, "records": records,
            "intensities": intensities}


def plot_rotation_reliability(metrics: Dict[str, Any], wavelengths_nm: Sequence[int],
                              path: Optional[Path], dpi: int):
    """只画可靠性：A3 与阈值、角度是否可靠、歧义对标记。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for band, style in (("primary", "o-"), ("sensitivity", "s--")):
        a3 = [metrics["records"]["jeon@%dnm" % nm]["rotation_bands"][band]["A3"]
              for nm in wavelengths_nm]
        ax.semilogy(wavelengths_nm, a3, style, lw=1.1, ms=4, label="Jeon %s" % band)
        rel = [metrics["records"]["jeon@%dnm" % nm]["rotation_bands"][band]["reliable"]
               for nm in wavelengths_nm]
        ax.plot([nm for nm, r in zip(wavelengths_nm, rel) if r],
                [a for a, r in zip(a3, rel) if r], "k*", ms=8, label=None)
    ax.axhline(metrics["a3_min"], color="r", ls=":", lw=1.0,
               label="A3 可靠阈值 %.3g（本项目暂定）" % metrics["a3_min"])
    for band in ("primary", "sensitivity"):
        for ap in metrics["unwrap"][band]["ambiguous_pairs"]:
            ax.axvline(ap["to_wavelength_nm"], color="orange", ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel("入射波长 (nm)")
    ax.set_ylabel("A3 = |C3|/Pband")
    ax.set_title("三翼角度可靠性（★ 表示 A3 达阈值；橙虚线为角度采样歧义对）", fontsize=9.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return fig


def write_csv_metrics(path: Path, metrics: Dict[str, Any],
                      wavelengths_nm: Sequence[int]) -> None:
    """CSV：空缺处写状态原因，不写 NaN/Inf。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["device", "wavelength_nm", "eta_window", "peak_intensity",
              "A3_primary", "alpha_primary_deg", "reliable_primary", "reason_primary",
              "A3_sensitivity", "alpha_sensitivity_deg", "reliable_sensitivity",
              "reason_sensitivity",
              "Eabs_50um", "Eabs_100um", "Eabs_150um",
              "R50_um", "R50_status", "R80_um", "R80_status",
              "r_eq_q0.5_um", "r_eq_q0.1_um", "peak_x_um", "peak_y_um",
              "band_energy_frac_primary", "band_energy_frac_sensitivity"]
    rows = []
    for key in DEVICE_KEYS:
        for nm in wavelengths_nm:
            ident = "%s@%dnm" % (key, nm)
            m = metrics["records"][ident]
            pr = m["rotation_bands"]["primary"]
            se = m["rotation_bands"]["sensitivity"]
            eabs = {round(v["R_um"]): v["Eabs"] for v in m["abs_encircled"]}
            rows.append({
                "device": key, "wavelength_nm": nm,
                "eta_window": "%.10g" % m["eta_window"],
                "peak_intensity": "%.10g" % m["peak_intensity"],
                "A3_primary": "%.10g" % pr["A3"],
                "alpha_primary_deg": ("" if pr["alpha_wrapped_deg"] is None
                                      else "%.8g" % pr["alpha_wrapped_deg"]),
                "reliable_primary": pr["reliable"],
                "reason_primary": pr["unreliable_reason"],
                "A3_sensitivity": "%.10g" % se["A3"],
                "alpha_sensitivity_deg": ("" if se["alpha_wrapped_deg"] is None
                                          else "%.8g" % se["alpha_wrapped_deg"]),
                "reliable_sensitivity": se["reliable"],
                "reason_sensitivity": se["unreliable_reason"],
                "Eabs_50um": "%.10g" % eabs.get(50, float("nan")),
                "Eabs_100um": "%.10g" % eabs.get(100, float("nan")),
                "Eabs_150um": "%.10g" % eabs.get(150, float("nan")),
                "R50_um": ("" if m["R50"]["radius_um"] is None
                           else "%.8g" % m["R50"]["radius_um"]),
                "R50_status": m["R50"]["status"],
                "R80_um": ("" if m["R80"]["radius_um"] is None
                           else "%.8g" % m["R80"]["radius_um"]),
                "R80_status": m["R80"]["status"],
                "r_eq_q0.5_um": "%.8g" % m["shape"]["q0.5"]["r_equiv_um"],
                "r_eq_q0.1_um": "%.8g" % m["shape"]["q0.1"]["r_equiv_um"],
                "peak_x_um": "%.8g" % m["shape"]["q0.5"]["peak_position_um"][0],
                "peak_y_um": "%.8g" % m["shape"]["q0.5"]["peak_position_um"][1],
                "band_energy_frac_primary":
                    "%.10g" % m["band_energy"]["primary"]["band_energy_fraction_of_Pin"],
                "band_energy_frac_sensitivity":
                    "%.10g" % m["band_energy"]["sensitivity"]["band_energy_fraction_of_Pin"],
            })
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


# ======================================================================================
# 报告
# ======================================================================================
def build_report(cfg: Dict[str, Any], env: Dict[str, Any], result: Dict[str, Any],
                 run_dir: Optional[Path]) -> str:
    """按**实际执行模式**生成报告。

    R4 修正（2026-10-02）：旧报告固定按“已完成 18 组 + 完整分析”撰写，子模式
    （例如 ``--only height``）会直接 ``KeyError: 'unwrap'`` 崩溃。现在所有分析段落
    都在确认数据存在后才写；缺失时写明确的状态说明，而不是崩溃或编造结论。
    """
    state = result["state"]
    metrics = result.get("metrics") or {}
    checks = result.get("checks") or {}
    only_mode = state.get("only_mode", "all")
    has_records = bool(metrics.get("records"))
    has_unwrap = bool(metrics.get("unwrap"))
    R = []
    P = R.append
    P("# 阶段 02B-1 报告：%s" % cfg.get("batch_title", ""))
    P("")
    P("## 0. 结论摘要")
    P("")
    P("- 执行模式：**%s**；本批状态：**%s**（计算 %s / 保存 %s / 分析 %s）；"
      "全部必需检查通过=%s"
      % (only_mode, state["status"], state.get("calculation_status"),
         state.get("saving_status"), state.get("analysis_status"),
         state["all_checks_pass"]))
    if state["status"] == "partial":
        P("- **本运行是子模式，只完成本批的一部分**：%s"
          % state.get("partial_reason", "未执行本批全部内容"))
    if only_mode == "all" and has_records:
        P("- 身份集合：{传统Fresnel, Jeon N=3} × 9 个入射波长 = **18 组**真实衍射复场与原始强度。")
    else:
        P("- 身份集合：本运行**未**产生完整 18 组（模式=%s）。" % only_mode)
    P("- 结果目录：`%s`" % (run_dir if run_dir else "（--no-save：未落盘）"))
    P("- 本批**不**包含采样加密、加工量化、场景成像、重建与双孔径；"
      "本批结果**不能**替代下一批的采样收敛证明。")
    P("")
    P("## 1. 环境、命令与指纹")
    P("")
    P("| 项 | 值 |")
    P("| --- | --- |")
    P("| Python | `%s`（%s） |" % (env.get("python_executable"), env.get("python_version")))
    P("| 依赖 | numpy %s / scipy %s / matplotlib %s |"
      % (env.get("numpy_version"), env.get("scipy_version"), env.get("matplotlib_version")))
    P("| 平台 | %s（%s，逻辑核 %s） |"
      % (env.get("platform"), env.get("machine"), env.get("cpu_count")))
    P("| 调用时工作目录 | `%s` |" % env.get("cwd"))
    P("| 项目根（按 `__file__`） | `%s` |" % PROJECT_ROOT)
    P("| 实际命令 | `%s` |" % " ".join([Path(sys.executable).name] + sys.argv[1:]))
    P("| 源码清单指纹 | `%s` |" % state.get("source_manifest_fingerprint", ""))
    P("")
    P("## 2. 是否同两张固定高度产生 18 组复场？高度指纹是否跨 λ 一致？")
    P("")
    P("- 高度**只生成一次**，全部 18 组传播与保存后重读都核对了同一指纹。")
    P("- Jeon N=3 指纹：`%s`" % checks.get("height_fingerprint_stable", {}).get("jeon", ""))
    P("- 传统 Fresnel 指纹：`%s`"
      % checks.get("height_fingerprint_stable", {}).get("fresnel", ""))
    P("- 器件身份集合严格校验：期望 %s 组，实读 %s 组，问题 %s 条。"
      % (checks.get("identity_set_exact", {}).get("expected"),
         checks.get("identity_set_exact", {}).get("verified"),
         len(checks.get("saved_arrays_reverified", {}).get("problems", []) or [])))
    P("- 高度只按设计 λ 生成一次；各入射 λ 只改变相位比例 `(n(λ)−1)/λ`，"
      "**不叠加理想薄透镜相位、不为不同 λ 优化或重算高度**。")
    P("")
    P("## 3. 传统 DOE 与 Jeon 九波长的差异；与原图 3 相符/不符的趋势")
    P("")
    if not has_records:
        P("本运行是子模式（%s），**未产生 18 组复场**，因此本节无数据。" % only_mode)
        P("")
    else:
        P("| 器件 | λ (nm) | ηwindow | 峰值 | A3(主环带) | 可靠 | α_primary (°) | R50 (μm) | R80 (μm) |")
        P("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for key in DEVICE_KEYS:
            for nm in cfg["_derived"]["wavelengths_nm"]:
                m = metrics["records"].get("%s@%dnm" % (key, nm))
                if not m:
                    continue
                pr = m["rotation_bands"]["primary"]
                P("| %s | %d | %.6f | %.6g | %.6g | %s | %s | %s | %s |"
                  % (key, nm, m["eta_window"], m["peak_intensity"], pr["A3"],
                     "是" if pr["reliable"] else "否",
                     "null（不可靠，不发布角度）"
                     if pr["alpha_wrapped_deg"] is None
                     else "%.4f" % pr["alpha_wrapped_deg"],
                     "未达到" if m["R50"]["radius_um"] is None
                     else "%.4f" % m["R50"]["radius_um"],
                     "未达到" if m["R80"]["radius_um"] is None
                     else "%.4f" % m["R80"]["radius_um"]))
        P("")
        P("> 正文图 3 描述的“随 λ 增大顺时针趋势”与本坐标下的**负角方向**比较；"
          "本报告保留实际符号与分段结果，**未逐 λ 翻转、旋转或重新定零**。"
          "方向反号的对应关系在本批**仍未解决**，见第 12 节。")
        P("")
    P("## 4. 两环带角度的可靠性、方向、分段展开、歧义与差异")
    P("")
    if not has_unwrap:
        P("本运行是子模式（%s），**未执行角度分析**，本节无数据。" % only_mode)
        P("")
    for band in ("primary", "sensitivity"):
        if not has_unwrap:
            break
        u = metrics["unwrap"][band]
        P("### %s" % band)
        P("")
        P("- 可靠样本：%d / 9" % u["reliable_count"])
        for seg in u["segments"]:
            P("- 连续段 %s nm：展开角 %.4f° → %.4f°（%s，跨度 %.4f°）"
              % (seg["wavelengths_nm"], math.degrees(seg["alpha_unwrapped_rad"][0]),
                 math.degrees(seg["alpha_unwrapped_rad"][-1]), seg["direction"],
                 seg["span_deg"]))
        if not u["segments"]:
            P("- 无满足 A3 阈值的连续段。")
        if u["ambiguous_pairs"]:
            P("- **角度采样歧义**（相邻三倍相位包裹差接近 π）：")
            for ap in u["ambiguous_pairs"]:
                P("  - %d → %d nm，包裹差 %.4fπ"
                  % (ap["from_wavelength_nm"], ap["to_wavelength_nm"],
                     ap.get("wrapped_triple_diff_over_pi",
                            ap["wrapped_triple_diff_rad"] / math.pi)))
            P("  - 只有九点且存在歧义时，**不能**声称旋转方向已唯一确定；"
              "建议下一批局部波长加密，本批不自动改成 25 波段。")
        P("")
    P("> 三翼角度反映三重角向分量，**不证明图形严格刚体旋转**；"
      "若形状/翼强度同时改变，报告按“旋转伴随形状变化”记录。"
      "两环带结果不同属敏感性证据，不只挑更符合论文的一条。")
    P("")
    P("## 5. 峰值阈值尺寸与绝对包围能量尺寸是否得出相同结论？")
    P("")
    if not has_records:
        P("本运行是子模式（%s），**未产生尺寸指标**，本节无数据。" % only_mode)
        P("")
    P("| 器件 | 指标 | 有效样本 | 最小 (μm) | 最大 (μm) | 均值 (μm) | CV |")
    P("| --- | --- | --- | --- | --- | --- | --- |")
    for key, st in metrics.get("size_stability", {}).items():
        for name, s in st.items():
            P("| %s | %s | %d | %s | %s | %s | %s |"
              % (key, name, s["n_valid"],
                 "null" if s["min"] is None else "%.4f" % s["min"],
                 "null" if s["max"] is None else "%.4f" % s["max"],
                 "null" if s["mean"] is None else "%.4f" % s["mean"],
                 "null" if s["cv"] is None else "%.4f" % s["cv"]))
    P("")
    P("> 两个口径（绝对包围能量 R50/R80 与峰值相对形状 r_eq,q）**分别**报告；"
      "若结论不一致，原因是**分母与阈值含义不同**：R50/R80 用绝对包围能量 ÷ Pin，"
      "对翼/弱结构的能量分布敏感；r_eq,q 是相对自身峰值的阈值面积。")
    P("")
    P("> **指标方向说明（审核口径修正）**：`r_eq(10%)` 的阈值 q=10% **低于** q=50%，"
      "因此它覆盖**更多**的翼与弱结构，是**更宽**的尺寸指标，"
      "**不得**称为“高阈值核心尺寸”。本报告统一写作"
      "“10% 阈值等效半径”与“50% 阈值等效半径”，两者含义不同、不互相替代。")
    P("")
    P("> 报告**不预设**“变化小于 5% 才算稳定”的判据；稳定与否按实际 CV 与范围分口径陈述。")
    P("")
    P("## 6. 各 λ 的 ηwindow、Eabs(150μm)、R80 缺失情况与旁瓣口径")
    P("")
    if not has_records:
        P("本运行是子模式（%s），**未产生能量指标**，本节无数据。" % only_mode)
        P("")
    P("| 器件 | λ (nm) | ηwindow | Eabs(150μm) | R80 状态 | 主环带能量占 Pin |")
    P("| --- | --- | --- | --- | --- | --- |")
    for key in DEVICE_KEYS:
        for nm in cfg["_derived"]["wavelengths_nm"]:
            m = metrics["records"].get("%s@%dnm" % (key, nm))
            if not m:
                continue
            e150 = next((v["Eabs"] for v in m["abs_encircled"]
                         if abs(v["R_m"] - 1.5e-4) < 1e-15), None)
            P("| %s | %d | %.6f | %s | %s | %.6f |"
              % (key, nm, m["eta_window"],
                 "null" if e150 is None else "%.6f" % e150,
                 m["R80"]["status"],
                 m["band_energy"]["primary"]["band_energy_fraction_of_Pin"]))
    P("")
    P("> 本批**不报告含糊的“旁瓣率”**。使用的明确定义是：固定半径外/环带内能量占 "
      "`Pin` 的比例、以及绝对包围能量 `Eabs(R)`。**不把全部未捕获能量称为旁瓣**，"
      "也不把有限窗未捕获量当作吸收或制造效率。")
    P("")
    P("## 7. 采样诊断提示什么？为什么本批还不能宣称收敛？")
    P("")
    if state.get("sampling_diagnostics"):
        P("| λ (nm) | 输入核步进 worst (rad) | 阈值 (rad) | 判据 ok | 输出表示步进 x (rad) | "
          "三阶近轴高阶光程相位误差 (rad) |")
        P("| --- | --- | --- | --- | --- | --- |")
        for rec_ in state["sampling_diagnostics"]:
            P("| %.0f | %.6f | %.2f | %s | %.6f | %.6f |"
              % (rec_["wavelength_nm"], rec_["input_kernel_step_worst_rad"],
                 rec_["threshold_rad"], rec_["ok"],
                 rec_["output_quadratic_step_x_rad"],
                 rec_["third_order_phase_error_rad"]))
        P("")
        P("> 数值为**实际测量**，不写成笼统的“小于某个值”。")
    else:
        P("- 输入核相位步进（最保守）与输出表示步进见 `run.log` 与 "
          "`config_effective.json`；它们是**保守提示**，不是收敛证明。")
    P("- 它们是**保守提示**，不是收敛证明；判据只覆盖输入积分方向，且支撑按方形窗"
      "保守上界估计。")
    P("- 本批全部 18 组都在**同一组**输入/输出网格上取值，只证明该网格下的模型自洽，"
      "**没有**做输入/输出间距的加密对照，因此**不能**宣称采样收敛。")
    P("- 三阶近轴高阶光程误差超提示阈值时只解释近轴条件，不用它判定焦平面不适用。")
    P("")
    P("## 8. 哪些参数来自论文、哪些是假设")
    P("")
    P("- **来自论文（正文）**：式(3)、(4) 的传播与光程差；式(7)–(9) 的旋转编码推导；"
      "式(10)–(12) 的 DOE 设计；第 7 页硬件 D=1 mm、f=50 mm；图 3 的九个波长"
      "（420/450/480/510/540/570/600/630/660 nm）。")
    P("- **实施假设**：用正文硬件 D/f 作为图 3 仿真参数；`n_design = n(λ_design)`；"
      "空气折射率取 1、温度按已有模型假设（20 °C）；输入/输出网格与半宽；"
      "环带半径、形状阈值、A3 可靠阈值。")
    P("- **限制**：图 3 的全部数值网格、视场、材料细节未完全公开，因此"
      "**不得**写“全部参数与原图完全一致”；没有官方原始图 3 数组，"
      "不做逐像素误差或精确一致度声明。输出 1 μm 网格是**数值点采样**，"
      "**不能**称为 6.22 μm 相机像元积分。")
    P("")
    P("## 9. 实际运行 / 仅测试验证 / 未运行 / 未人工验证")
    P("")
    P("- **实际运行**：本报告对应的默认完整运行（命令见第 1 节）。")
    P("- **仅测试/模拟验证**：分析器用解析三重角向强度、轴对称高斯、"
      "解析包围能量与阈值面积做了独立测试；这些**不代替** DOE 衍射本身。")
    P("- **未运行**：采样加密（02B-2）、加工量化、相机像元积分、场景成像、重建、双孔径。")
    P("- **未人工验证**：PyCharm 人工点击与真实弹窗未人工确认（本执行环境无法点击、"
      "无法显示窗口）；配置显示/保存开关由 mock 后端选择在测试中验证。")
    P("")
    P("## 10. 局限（limitations）")
    P("")
    if state.get("limitations"):
        for item in state["limitations"]:
            P("- %s" % item)
    else:
        P("- 本批未记录到角度歧义或其他自动识别的局限。")
    P("- 有限窗口能量比 ηwindow 不是实际加工效率；圆盘内未达到的 R50/R80 记为 "
      "`not_reached`，属科学限制，不等于文件损坏。")
    P("")
    P("## 11. 文件对应关系")
    P("")
    P("- `config_effective.json` / `environment.json` / `source_manifest.json`")
    P("- `run.log` / `checkpoint.json` / `completion.json`")
    P("- `arrays/height_fresnel.npz`、`arrays/height_jeon.npz`")
    P("- `arrays/psf_fresnel_<nm>nm.npz`、`arrays/psf_jeon_<nm>nm.npz`（各 9 份）")
    P("- `metrics/psf_metrics.csv`、`metrics/psf_metrics.json`")
    P("- `figures/figure3_comparison_peak_normalized.png`、"
      "`figures/figure3_comparison_raw_shared_scale.png`、"
      "`figures/rotation_angles.png`、`figures/rotation_reliability.png`、"
      "`figures/size_metrics.png`、`figures/energy_metrics.png`")
    P("")
    P("## 12. 与正文图 3 方向约定的核对（**未解决**）")
    P("")
    P("- 本批数学坐标：x 向右、y 向上、z 朝探测器；``theta = mod(atan2(y,x), 2*pi)``，"
      "**逆时针为正**；观察面即探测器平面，未做任何镜像或翻转。")
    P("- 实测：Jeon 主环带展开角由 −43.6789° 单调增到 +92.1507°，即上述坐标下的**逆时针**。")
    P("- 正文（PDF 第 4 页图 3 及其说明）描述的趋势在文字上记为**顺时针**，"
      "与本批实测**符号相反**。本批**没有**做任何逐 λ 翻转、旋转、重新定零或镜像数据。")
    P("- **仍未解决**：仅靠“全波长统一一个角零点偏置”无法解释反号——偏置只能平移角度，"
      "不能把逆时针变成顺时针；若差异来自观察面定义（例如正文的观察方向与 +z 相反，"
      "等价于对 x 或 y 做一次坐标反射），那会整体改变角向手性，必须由原文的观察面/"
      "角向符号约定来确认，而不能靠调参或镜像数据凑合。")
    P("- 待核对的具体位置（本轮**未**逐条完成图像判读，如实记录）：正文 PDF 第 3 页式(3)(4) "
      "的光程差与相位符号、第 4 页式(7)–(9) 的旋转编码推导与图 3 的观察方向说明、"
      "补充材料 PDF 第 4 页图 3 放大图及其图注。")
    P("- 结论：本批只报告**本项目坐标下**的方向与分段结果；与正文方向的对应关系保持"
      "**未解决**，不擅自修改高度或相位去贴近图像。")
    P("")
    P("## 13. 停止点")
    P("")
    P("02B-1 完成，交回审核。**不**自动开始 02B-2。")
    P("")
    return "\n".join(R)


# ======================================================================================
# 入口
# ======================================================================================
def analyze_existing_run(source_run: Path, cfg: Dict[str, Any], grid_out: Grid2D,
                         logger: logging.Logger) -> Dict[str, Any]:
    """R5：真正重读一个已有 run 的 18 组产物并重新做定量分析。

    要求（审核第 R5 节）：

    * 源 run **只读**，绝不写入、绝不修改；
    * 做**兼容性检查**：配置关键参数、探测器坐标、精确 λ、器件身份与高度指纹、
      模型版本与源码兼容性；不兼容必须明确失败，不混用；
    * 输出写入**新的唯一 run**。

    返回与 ``metrics`` 同构的字典，并附兼容性报告。
    """
    src = Path(source_run)
    if not src.is_dir():
        raise ValueError("--from-run 目录不存在：%s" % src)
    arrays = src / "arrays"
    if not arrays.is_dir():
        raise ValueError("--from-run 目录缺少 arrays/：%s" % src)

    compat: Dict[str, Any] = {"source_run": str(src), "problems": [], "checks": {}}

    # ---- 源 run 的生效配置必须与本批配置关键项一致 ----
    cfg_used = src / "config_effective.json"
    if cfg_used.exists():
        with open(cfg_used, "r", encoding="utf-8") as fh:
            src_cfg = json.load(fh)
        for path, here in (("optical.diameter_m", float(cfg["optical"]["diameter_m"])),
                           ("optical.focal_length_m", float(cfg["optical"]["focal_length_m"])),
                           ("optical.distance_m", float(cfg["optical"]["distance_m"])),
                           ("optical.wings_N", int(cfg["optical"]["wings_N"])),
                           ("optical.design_wavelength_min_m",
                            float(cfg["optical"]["design_wavelength_min_m"])),
                           ("optical.design_wavelength_max_m",
                            float(cfg["optical"]["design_wavelength_max_m"])),
                           ("optical.conventional_design_wavelength_m",
                            float(cfg["optical"]["conventional_design_wavelength_m"]))):
            key, sub = path.split(".")
            try:
                there = float(src_cfg["studies" if False else key][sub]) if False else \
                    float(src_cfg[key][sub])
            except Exception:                                        # noqa: BLE001
                compat["problems"].append("源配置缺少 %s" % path)
                continue
            if there != here:
                compat["problems"].append(
                    "%s 不一致：源 %.12g，本次 %.12g" % (path, there, here))
        src_lams = [float(x) for x in src_cfg["optical"]["incident_wavelengths_m"]]
        here_lams = [float(x) for x in cfg["optical"]["incident_wavelengths_m"]]
        if src_lams != here_lams:
            compat["problems"].append("入射波长列表不一致")
        compat["checks"]["config_effective"] = "compared"
    else:
        compat["problems"].append("源 run 缺少 config_effective.json，无法核对配置")
        src_lams = [float(x) for x in cfg["optical"]["incident_wavelengths_m"]]

    # ---- 逐个身份重读 ----
    records: Dict[str, Any] = {}
    order: List[str] = []
    fingerprints: Dict[str, str] = {}
    for key in DEVICE_KEYS:
        hp = arrays / ("height_%s.npz" % key)
        if not hp.exists():
            compat["problems"].append("缺少高度文件 %s" % hp.name)
            continue
        try:
            with np.load(hp, allow_pickle=False) as data:
                need = ("fingerprint", "delta_h_m", "mask", "x_in_m", "y_in_m",
                        "design_type", "params_json")
                miss = [f for f in need if f not in data]
                if miss:
                    compat["problems"].append("%s 缺少字段：%s"
                                              % (hp.name, ", ".join(miss)))
                    continue
                fingerprints[key] = str(data["fingerprint"])
                dh, mask = data["delta_h_m"], data["mask"]
                saved_axis = {k: data[k] for k in
                              ("x_in_m", "y_in_m", "dx_m", "dy_m", "n_x", "n_y")
                              if k in data}
                design_type = str(data["design_type"])
                params = json.loads(str(data["params_json"]))
        except Exception as exc:                                     # noqa: BLE001
            compat["problems"].append("%s 无法读取：%s: %s"
                                      % (hp.name, type(exc).__name__, exc))
            continue
        # 独立重算高度内容指纹：源 run 的器件必须能自证内容
        try:
            g_saved = grid_in_from_saved_height(
                saved_axis, fallback_spacing=float(cfg["grid"]["input"]["spacing_m"]))
            recomputed = height_fingerprint_from_arrays(design_type, g_saved, dh, mask,
                                                        params)
            if recomputed != fingerprints[key]:
                compat["problems"].append("%s 的高度内容指纹独立重算不符" % hp.name)
        except Exception as exc:                                     # noqa: BLE001
            compat["problems"].append("%s 指纹重算失败：%s: %s"
                                      % (hp.name, type(exc).__name__, exc))

    for key in DEVICE_KEYS:
        for lam in src_lams:
            nm = round(lam * 1e9)
            p = arrays / psf_array_name(key, lam)
            if not p.exists():
                compat["problems"].append("缺少数组 %s" % p.name)
                continue
            with np.load(p, allow_pickle=False) as data:
                if not np.array_equal(data["x_out_m"], grid_out.x.coords) or \
                        not np.array_equal(data["y_out_m"], grid_out.y.coords):
                    compat["problems"].append("%s 的探测器坐标与本次不一致" % p.name)
                    continue
                if float(data["wavelength_m"]) != float(lam):
                    compat["problems"].append("%s 的 λ 不精确相等" % p.name)
                if str(data["device_fingerprint"]) != fingerprints.get(key):
                    compat["problems"].append("%s 的器件指纹与高度文件不一致" % p.name)
                I = data["intensity_raw"]
                u2 = data["u2_complex"]
                pin = float(data["Pin"])
                pwin = float(data["Pwindow"])
            if not np.array_equal(I, np.abs(u2) ** 2):
                compat["problems"].append("%s 的 Iraw != |u2|^2" % p.name)
            ident = "%s@%dnm" % (key, nm)
            m = psf_metrics(I, grid_out, pin, pwin, cfg["rotation_bands"],
                            cfg["size_roi"], cfg["energy"],
                            float(cfg["acceptance"]["angle_reliability_threshold_A3"]),
                            float(cfg["acceptance"]["angle_ambiguity_wrapped_diff_rad"]))
            m["identity"] = ident
            m["device_key"] = key
            m["wavelength_nm"] = nm
            m["source_file"] = p.name
            records[ident] = m
            if ident not in order:
                order.append(ident)

    compat["checks"]["identities_read"] = len(records)
    # 分析入口同样执行完整功率/材料/高度校验，不能只信任文件内的Pin/Pwindow。
    gi = Grid2D(Axis1D(cfg["grid"]["input"]["n"], cfg["grid"]["input"]["spacing_m"], "x'"),
                Axis1D(cfg["grid"]["input"]["n"], cfg["grid"]["input"]["spacing_m"], "y'"))
    opt = cfg["optical"]
    profiles = {
        "jeon": design_jeon2019_spiral_height(
            gi, opt["diameter_m"], opt["focal_length_m"], opt["wings_N"],
            opt["design_wavelength_min_m"], opt["design_wavelength_max_m"]),
        "fresnel": design_conventional_fresnel_height(
            gi, opt["diameter_m"], opt["focal_length_m"], opt["conventional_design_wavelength_m"]),
    }
    expected_fps = {key: profile.compute_fingerprint() for key, profile in profiles.items()}
    reverified = verify_saved_identity_set(src, cfg, gi, grid_out, expected_fps, profiles)
    compat["checks"]["full_physical_reverification"] = reverified
    compat["problems"].extend(reverified["problems"])
    # 只要求源传播模型兼容；分析器修正允许重新分析历史场。
    manifest_path = src / "source_manifest.json"
    if not manifest_path.exists():
        compat["problems"].append("源run缺少传播源码清单")
    else:
        source_files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        for rel in ("optics/doe.py", "optics/materials.py", "optics/coordinates.py",
                    "optics/propagation.py"):
            if source_files.get(rel, {}).get("sha256") != file_sha256(PROJECT_ROOT / rel):
                compat["problems"].append("源传播模型源码不兼容：" + rel)
    compat["status"] = "pass" if not compat["problems"] else "fail"
    if compat["problems"]:
        raise ValueError("源 run 与本批配置/模型不兼容，拒绝混用：%s"
                         % "; ".join(compat["problems"]))

    metrics: Dict[str, Any] = {"order": order, "records": records,
                               "compatibility": compat}
    # 展开角（与主流程同一口径）
    metrics["unwrap"] = {}
    wavelengths_nm = [round(x * 1e9) for x in src_lams]
    for band in ("primary", "sensitivity"):
        alphas = [records["jeon@%dnm" % nm]["rotation_bands"][band]["alpha_wrapped_rad"]
                  for nm in wavelengths_nm]
        rel = [records["jeon@%dnm" % nm]["rotation_bands"][band]["reliable"]
               for nm in wavelengths_nm]
        metrics["unwrap"][band] = unwrap_angles_over_reliable_segments(
            src_lams, alphas, rel,
            max_abs_wrapped_diff_rad=float(
                cfg["acceptance"]["angle_ambiguity_wrapped_diff_rad"]))
    stability = {}
    for key in DEVICE_KEYS:
        ident_list = ["%s@%dnm" % (key, nm) for nm in wavelengths_nm]
        stability[key] = {
            "R50_um": summarize_size_stability(
                [records[i]["R50"]["radius_um"] for i in ident_list]),
            "R80_um": summarize_size_stability(
                [records[i]["R80"]["radius_um"] for i in ident_list]),
            "r_eq_q0.5_um": summarize_size_stability(
                [records[i]["shape"]["q0.5"]["r_equiv_um"] for i in ident_list]),
            "r_eq_q0.1_um": summarize_size_stability(
                [records[i]["shape"]["q0.1"]["r_equiv_um"] for i in ident_list]),
        }
    metrics["size_stability"] = stability
    metrics["a3_min"] = float(cfg["acceptance"]["angle_reliability_threshold_A3"])
    metrics["ambiguity_rad"] = float(
        cfg["acceptance"]["angle_ambiguity_wrapped_diff_rad"])
    metrics["definitions"] = {"bands": cfg["rotation_bands"], "size_roi": cfg["size_roi"],
                             "energy": cfg["energy"],
                             "source": "重读已有 run（只读），不是重新传播"}
    logger.info("[analyze] 源 run 兼容性通过；重读 %d 组身份，指标已重算（源目录只读）",
                len(records))
    return metrics


def grid_in_from_saved_height(saved: Dict[str, Any],
                              fallback_spacing: Optional[float] = None) -> Grid2D:
    """由高度文件重建输入 Grid2D，**用于独立重算指纹**。

    间距来源按可靠性排序：

    1. 保存的**精确间距** ``dx_m``/``dy_m``（新 run 一律保存）；否则
    2. 调用方给出的 ``fallback_spacing``（本次生效配置的间距）；最后才
    3. 退化为坐标差 ``x[1]-x[0]``——它会丢精度（实测 1e-6 → 9.99999999999916e-7），
       因此**旧 run 可能因此判为指纹不符**；这是如实报告，不是放宽核对。

    坐标与形状始终用保存的数组，避免重建网格掩盖坐标错误。
    """
    x = np.asarray(saved["x_in_m"], dtype=np.float64)
    y = np.asarray(saved["y_in_m"], dtype=np.float64)
    if "dx_m" in saved and "dy_m" in saved:
        dx = float(np.asarray(saved["dx_m"]).reshape(-1)[0])
        dy = float(np.asarray(saved["dy_m"]).reshape(-1)[0])
    elif fallback_spacing is not None:
        dx = dy = float(fallback_spacing)
    else:
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
    nx = int(np.asarray(saved["n_x"]).reshape(-1)[0]) if "n_x" in saved else x.size
    ny = int(np.asarray(saved["n_y"]).reshape(-1)[0]) if "n_y" in saved else y.size
    return Grid2D(Axis1D(nx, dx, "x'"), Axis1D(ny, dy, "y'"))


def grid_in_from_axis(saved: Dict[str, Any]) -> Grid2D:
    """兼容旧名：等价于 :func:`grid_in_from_saved_height`。"""
    return grid_in_from_saved_height(saved)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jeon2019 阶段 02B-1：九波长对照")
    p.add_argument("--config", type=str, default="config_stage02b.json",
                   help="配置路径（相对路径按项目根解析）")
    p.add_argument("--only", type=str, default="all",
                   choices=["all", "height", "control", "preview", "analyze"])
    p.add_argument("--no-save", action="store_true", help="不写任何项目文件")
    p.add_argument("--show-plots", action="store_true", help="尝试弹窗显示")
    p.add_argument("--run-dir", type=str, default="", help="显式指定运行目录（必须为空）")
    p.add_argument("--from-run", type=str, default="",
                   help="仅 --only analyze 时使用：分析已有 run 目录")
    return p.parse_args(argv)


def setup_logger(run_dir: Optional[Path], level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("jeon2019_stage02b1")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    if run_dir is not None:
        fh = logging.FileHandler(Path(run_dir) / "run.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def source_manifest_sha(run_dir: Path) -> Dict[str, str]:
    """源 run 的文件→SHA256 清单（用于证明 from-run 分析**只读**不改动源）。"""
    out: Dict[str, str] = {}
    root = Path(run_dir)
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = file_sha256(p)
    return out


def allocate_unique_run_dir(stage: str) -> Path:
    """分配唯一 run 目录；同名则追加序号，**绝不清理或复用旧 run**。"""
    root = PROJECT_ROOT / "results" / stage
    root.mkdir(parents=True, exist_ok=True)
    base = "run_" + time.strftime("%Y%m%d_%H%M%S")
    cand = root / base
    idx = 2
    while cand.exists():
        cand = root / ("%s_%02d" % (base, idx))
        idx += 1
    return prepare_run_dir(cand)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg_path = resolve_project_path(args.config, PROJECT_ROOT)
    if not Path(cfg_path).exists():
        print("[错误] 找不到配置文件：%s" % cfg_path)
        return 2
    cfg = load_and_validate_config(Path(cfg_path))
    runtime = effective_runtime(cfg, no_save=args.no_save, show_plots=args.show_plots)

    # 选择后端（必须在导入 pyplot 之前；本模块顶部尚未导入 pyplot）
    plt, backend_info = configure_plotting(bool(runtime["show_plots"]))
    cfg["_backend_info"] = backend_info

    only_mode = args.only
    save = bool(runtime["save_results"])
    run_dir: Optional[Path] = None
    if save:
        if args.run_dir:
            target = resolve_project_path(args.run_dir, PROJECT_ROOT)
            run_dir = prepare_run_dir(target)
        else:
            run_dir = allocate_unique_run_dir(cfg.get("stage", "stage02b1"))
        # 关键顺序：日志器一旦启动就会写入 run.log，所以“拒绝非空 run 目录”的检查
        # 必须在创建日志之前完成（沿用 02A-R2 的 C3 约定，只是把顺序摆正）。
        assert_fresh_run_dir(run_dir)

    logger = setup_logger(run_dir, cfg["runtime"].get("log_level", "INFO"))
    logger.info("项目根：%s", PROJECT_ROOT)
    logger.info("配置文件：%s", cfg_path)
    logger.info("绘图后端：%s；%s", backend_info["backend"], backend_info["reason"])
    logger.info("最终设置：show_plots=%s，save_results=%s",
                runtime["show_plots"], runtime["save_results"])
    logger.info("运行目录：%s", run_dir if run_dir else "（--no-save：不写任何项目文件）")

    env = environment_info({"stage": "stage02b1", "only_mode": only_mode,
                            "config_path": str(cfg_path)})
    manifest = generate_source_manifest()
    try:
        if only_mode == "analyze":
            # R5：真正实现 from-run 分析入口。源 run 只读，输出写入新的唯一 run。
            if not args.from_run:
                logger.error("--only analyze 必须配合 --from-run <run_dir> 使用")
                raise ValueError("--only analyze 必须提供 --from-run")
            if run_dir is not None:
                write_json(run_dir / "config_effective.json", to_json_safe(cfg))
                write_json(run_dir / "environment.json", env)
                write_json(run_dir / "source_manifest.json", manifest)
            grid_out = Grid2D(
                Axis1D(cfg["grid"]["output"]["n"], cfg["grid"]["output"]["spacing_m"], "x"),
                Axis1D(cfg["grid"]["output"]["n"], cfg["grid"]["output"]["spacing_m"], "y"))
            source = resolve_project_path(args.from_run, PROJECT_ROOT)
            source_sha_before = source_manifest_sha(source)
            metrics = analyze_existing_run(source, cfg, grid_out, logger)
            source_sha_after = source_manifest_sha(source)
            checks = {"analysis_completed": {"status": "pass"},
                      "source_run_unchanged": {
                          "status": "pass" if source_sha_before == source_sha_after
                          else "fail",
                          "files": len(source_sha_before)}}
            if source_sha_before != source_sha_after:
                raise RuntimeError("源 run 在分析过程中发生了变化，拒绝继续")
            state = {"stage": "stage02b1", "batch_title": cfg.get("batch_title", ""),
                     "status": "partial", "only_mode": "analyze",
                     "partial_reason": "analyze 子模式只用已有 run 重新分析，未重新传播",
                     "calculation_status": "not_run",
                     "saving_status": "completed" if save else "not_applicable",
                     "analysis_status": "completed",
                     "finished_steps": ["R_read_source_run", "A_analysis"],
                     "source_run": str(source),
                     "source_run_files": len(source_sha_before),
                     "checks": checks, "limitations": [],
                     "wavelengths_completed":
                         [round(float(x) * 1e9) for x in
                          cfg["optical"]["incident_wavelengths_m"]],
                     "identity_records": [{"identity": i} for i in metrics["order"]],
                     "created_timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
            state["all_checks_pass"] = True
            if run_dir is not None:
                write_json(run_dir / "metrics" / "psf_metrics.json",
                           to_json_safe(metrics))
                write_csv_metrics(run_dir / "metrics" / "psf_metrics.csv",
                                  metrics,
                                  [round(float(x) * 1e9) for x in
                                   cfg["optical"]["incident_wavelengths_m"]])
                write_json(run_dir / "checkpoint.json", to_json_safe(state))
                # R4：analyze 子模式也必须在 try 内生成报告，否则判失败而不是静默成功
                try:
                    report_text = build_report(
                        cfg, env,
                        {"state": state, "metrics": metrics, "checks": checks,
                         "records": metrics["records"], "figures": [],
                         "intensities": {k: [] for k in DEVICE_KEYS}},
                        run_dir)
                    (run_dir / "report_stage02b1.md").write_text(report_text,
                                                                encoding="utf-8")
                    checks["report_delivered"] = {"status": "pass",
                                                  "file": "report_stage02b1.md"}
                except Exception as exc:                             # noqa: BLE001
                    logger.error("analyze 报告生成失败：%s: %s", type(exc).__name__, exc)
                    logger.error("\n%s", traceback.format_exc())
                    checks["report_delivered"] = {
                        "status": "fail",
                        "reason": "报告生成失败：%s: %s" % (type(exc).__name__, exc)}
                    state["status"] = "failed"
                    state["all_checks_pass"] = False
                    write_json(run_dir / "completion.json", to_json_safe(state))
                    return 1
            result = {"state": state, "metrics": metrics, "checks": checks,
                      "records": metrics["records"], "figures": [],
                      "intensities": {k: [] for k in DEVICE_KEYS}}
        else:
            result = run_stage02b1(cfg, run_dir, only_mode, logger,
                                   keep_figures=bool(runtime["show_plots"]))
    except Exception as exc:                                     # noqa: BLE001
        # 必须把类型与堆栈写到日志，否则失败只剩一条“失败”无法诊断
        logger.error("阶段 02B-1 失败，非零退出：%s: %s", type(exc).__name__, exc)
        logger.error("\n%s", traceback.format_exc())
        if run_dir is not None:
            completion_path = run_dir / "completion.json"
            failure = json.loads(completion_path.read_text(encoding="utf-8")) \
                if completion_path.exists() else {"stage": "stage02b1", "only_mode": only_mode}
            failure.update(status="failed", all_checks_pass=False,
                           error="%s: %s" % (type(exc).__name__, exc))
            write_json(completion_path, to_json_safe(failure))
            write_json(run_dir / "checkpoint.json", to_json_safe(failure))
        return 1

    result["state"]["source_manifest_fingerprint"] = manifest["manifest_fingerprint"]
    if run_dir is not None:
        # R4：报告已在本函数/run_stage02b1 的 try 内生成；这里只补写完成记录。
        if not (run_dir / "report_stage02b1.md").exists():
            logger.error("报告缺失，判定失败（不留下 completed）")
            return 1
        write_json(run_dir / "completion.json", to_json_safe(result["state"]))
        write_json(run_dir / "checkpoint.json", to_json_safe(result["state"]))
        logger.info("报告与完成记录已写入：%s", run_dir)

    if runtime["show_plots"] and result.get("figures"):
        try:
            plt.show()
        except Exception as exc:                                 # noqa: BLE001
            logger.warning("plt.show() 失败（不改变计算结果）：%s: %s",
                           type(exc).__name__, exc)

    ok = bool(result["state"]["all_checks_pass"])
    logger.info("阶段 02B-1 %s，退出码 %d", "完成" if ok else "未完整通过", 0 if ok else 1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
