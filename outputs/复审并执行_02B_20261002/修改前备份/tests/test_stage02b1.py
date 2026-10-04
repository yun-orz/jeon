# -*- coding: utf-8 -*-
"""阶段 02B-1 专项测试：分析器独立验证 + 身份集合/完整性拒绝路径。

设计原则（任务书第 11 节）：**优先用独立解析结果测试分析器**，不写仅重复实现公式的
空测试。这里用到的解析结果与 DOE 衍射**无关**，因此它们只验证“分析器算得对不对”，
不构成对 DOE 物理结论的证明。

独立解析参照
------------
1. 三重角向强度 ``I = g(r)·[1 + a·cos(3(θ−α))]``：理论 ``arg(C3)/3 = α``（模 120°），
   用于检查跨 120° 周期的角度、符号与连续段展开。
2. 轴对称高斯：``C3 ≈ 0``，方向必须判为不可靠（不能把像素误差当成稳定朝向）。
3. 二维高斯绝对包围能量 ``E(R) = 1 − exp[−R²/(2σ²)]``：
   ``Pin`` 取解析全平面功率 ``2πσ²A``，检查 R50/R80 与窗口不足时的 ``not_reached``。
4. 高斯阈值面积 ``A_q = −2πσ²ln(q)``：检查 50% / 10% 面积与固定 ROI 截断。
5. 数据完整性：NaN / 错误形状 / 无能量 / 缺一个波长 / 重复身份 / 伪换高度 /
   错误物理坐标都必须被拒绝；**两器件都检查功率**，不只检查 Jeon。

运行::

    python -B -m unittest discover -s tests -p "test_stage02b1.py" -v
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optics.coordinates import Axis1D, Grid2D                       # noqa: E402
from _stage02b1_support import (retained_evidence_dir, write_evidence,  # noqa: E402
                                describe_encoding_setup, run_child, child_env)
from optics.materials import refractive_index_fused_silica          # noqa: E402
from optics.propagation import fresnel_kernel_separable              # noqa: E402
from optics.psf_analysis import (                                   # noqa: E402
    absolute_encircled_energy, c3_rotation_metric, find_first_radius_reaching,
    summarize_size_stability, threshold_area_radius,
    unwrap_angles_over_reliable_segments,
)

LAM = 550e-9


def three_fold_intensity(grid: Grid2D, alpha_rad: float, a: float = 0.3,
                         sigma: float = 1.2e-4) -> np.ndarray:
    """解析三重角向强度 ``g(r)[1 + a·cos(3(θ−α))]``（只用于分析器测试）。"""
    X, Y = grid.meshgrid()
    r = np.hypot(X, Y)
    th = np.mod(np.arctan2(Y, X), 2.0 * np.pi)
    g = np.exp(-(r / sigma) ** 2)
    return g * (1.0 + a * np.cos(3.0 * (th - alpha_rad)))


def gaussian_intensity(grid: Grid2D, sigma_m: float, amplitude: float = 1.0
                       ) -> np.ndarray:
    """轴对称二维高斯强度（``I = A·exp(−r²/(2σ²))``）。"""
    X, Y = grid.meshgrid()
    return amplitude * np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma_m ** 2))


def make_grid(n: int = 801, d: float = 0.6e-6) -> Grid2D:
    return Grid2D(Axis1D(n, d, "x"), Axis1D(n, d, "y"))


# ======================================================================================
# 1. 三重角向解析强度
# ======================================================================================
class TestC3AgainstAnalyticThreeFold(unittest.TestCase):
    def setUp(self):
        self.grid = make_grid()

    def test_alpha_recovered_modulo_120deg(self):
        """理论 arg(C3)/3 = α（模 120°）：跨周期的正负角都必须正确。"""
        for deg in (0.0, 25.0, 70.0, 110.0, -40.0, -100.0, 179.0):
            with self.subTest(alpha_deg=deg):
                I = three_fold_intensity(self.grid, math.radians(deg))
                m = c3_rotation_metric(I, self.grid, 30e-6, 120e-6, a3_min=0.05)
                self.assertTrue(m["reliable"])
                got = m["alpha_wrapped_deg"]
                diff = (got - deg + 60.0) % 120.0 - 60.0
                self.assertLess(abs(diff), 1e-6,
                                "α=%.2f° 测得 %.6f°，模 120° 差 %.3e°" % (deg, got, diff))

    def test_a3_equals_analytic_value(self):
        """对 ``I = g(r)[1+a·cos(3(θ−α))]``，``A3 = |C3|/Pband = a/2``。"""
        for a in (0.1, 0.3, 0.6):
            with self.subTest(a=a):
                I = three_fold_intensity(self.grid, math.radians(33.0), a=a)
                m = c3_rotation_metric(I, self.grid, 30e-6, 120e-6, a3_min=1e-6)
                self.assertAlmostEqual(m["A3"], a / 2.0, places=6)

    def test_sign_convention_counterclockwise_positive(self):
        """逆时针为正：α 增大时 arg(C3) 增大（同号）。"""
        I1 = three_fold_intensity(self.grid, math.radians(10.0))
        I2 = three_fold_intensity(self.grid, math.radians(40.0))
        a1 = c3_rotation_metric(I1, self.grid, 30e-6, 120e-6, a3_min=0.05)["alpha_wrapped_rad"]
        a2 = c3_rotation_metric(I2, self.grid, 30e-6, 120e-6, a3_min=0.05)["alpha_wrapped_rad"]
        self.assertGreater(a2 - a1, 0.0)
        self.assertAlmostEqual(math.degrees(a2 - a1), 30.0, places=6)

    def test_axisymmetric_gaussian_gives_zero_c3(self):
        """轴对称高斯：C3 应接近 0，且必须判为方向不可靠。"""
        I = gaussian_intensity(self.grid, 1.2e-4)
        m = c3_rotation_metric(I, self.grid, 30e-6, 120e-6, a3_min=0.05)
        self.assertLess(m["A3"], 1e-10)
        self.assertFalse(m["reliable"])
        self.assertTrue(m["unreliable_reason"])

    def test_band_excludes_axis_and_outside(self):
        """环带外权重为 0、光轴 r=0 不计入：只改环带外的像素不得改变 C3。"""
        I = three_fold_intensity(self.grid, math.radians(20.0))
        m1 = c3_rotation_metric(I, self.grid, 30e-6, 120e-6, a3_min=0.05)
        I2 = I.copy()
        X, Y = self.grid.meshgrid()
        r = np.hypot(X, Y)
        I2[r < 25e-6] *= 7.0            # 环带内界以内
        I2[r > 130e-6] *= 7.0           # 环带外界以外
        m2 = c3_rotation_metric(I2, self.grid, 30e-6, 120e-6, a3_min=0.05)
        self.assertAlmostEqual(m1["C3_real"], m2["C3_real"], places=18)
        self.assertAlmostEqual(m1["C3_imag"], m2["C3_imag"], places=18)

    def test_unwrap_across_120deg_period(self):
        """跨 120° 周期的连续段必须正确展开，且分段不跨不可靠间隙。"""
        degs = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 118.0, 138.0, 158.0]
        alphas = [math.radians(d % 120.0 - 60.0) for d in degs]
        rel = [True] * len(degs)
        u = unwrap_angles_over_reliable_segments(
            [400e-9 + i * 20e-9 for i in range(len(degs))], alphas, rel)
        self.assertEqual(len(u["segments"]), 1)
        span = u["segments"][0]["span_deg"]
        self.assertAlmostEqual(span, 158.0, places=6)
        self.assertEqual(u["segments"][0]["direction"], "逆时针")
        self.assertFalse(u["has_ambiguity"])

    def test_unwrap_does_not_cross_unreliable_gap(self):
        """低可靠点把序列切断，方向不得跨间隙续接。"""
        degs = [0.0, 20.0, 40.0, 60.0]
        alphas = [math.radians(d) for d in degs]
        rel = [True, True, False, True]
        u = unwrap_angles_over_reliable_segments(
            [400e-9, 420e-9, 440e-9, 460e-9], alphas, rel)
        self.assertEqual(len(u["segments"]), 2)
        self.assertIsNone(u["alpha_unwrapped_rad"][2])
        self.assertEqual(u["segments"][0]["indices"], [0, 1])
        self.assertEqual(u["segments"][1]["indices"], [3])

    def test_ambiguity_flagged_near_pi(self):
        """相邻三倍相位包裹差接近 π 时必须标记角度采样歧义。"""
        alphas = [0.0, math.radians(58.0)]      # 3×(58°) = 174° ≈ 0.967π
        u = unwrap_angles_over_reliable_segments([400e-9, 420e-9], alphas, [True, True],
                                                max_abs_wrapped_diff_rad=0.9 * math.pi)
        self.assertTrue(u["has_ambiguity"])
        self.assertEqual(len(u["ambiguous_pairs"]), 1)


# ======================================================================================
# 2. 高斯能量的解析参照
# ======================================================================================
class TestGaussianEnergyAnalytic(unittest.TestCase):
    def setUp(self):
        self.grid = make_grid(1201, 0.5e-6)
        self.sigma = 8.0e-5
        self.amp = 3.0
        self.I = gaussian_intensity(self.grid, self.sigma, self.amp)
        # 解析全平面功率：Pin = Σ I dA 的连续极限 2πσ²A
        self.pin_analytic = 2.0 * math.pi * self.sigma ** 2 * self.amp

    def test_absolute_encircled_energy_matches_analytic(self):
        """E(R) = 1 − exp[−R²/(2σ²)]，分母用解析全平面功率。"""
        radii = [0.5 * self.sigma, self.sigma, 2.0 * self.sigma]
        got = absolute_encircled_energy(self.I, self.grid, self.pin_analytic, radii)
        for rec, R in zip(got, radii):
            expect = 1.0 - math.exp(-R ** 2 / (2.0 * self.sigma ** 2))
            self.assertLess(abs(rec["Eabs"] - expect), 2e-4,
                            "R=%.4g: 数值 %.6f 解析 %.6f" % (R, rec["Eabs"], expect))

    def test_R50_R80_match_analytic_radius(self):
        """R50/R80 应等于 σ·sqrt(2·ln(1/(1−q)))。"""
        r50 = find_first_radius_reaching(self.I, self.grid, self.pin_analytic, 0.5)
        r80 = find_first_radius_reaching(self.I, self.grid, self.pin_analytic, 0.8)
        exp50 = self.sigma * math.sqrt(2.0 * math.log(2.0))
        exp80 = self.sigma * math.sqrt(2.0 * math.log(5.0))
        self.assertEqual(r50["status"], "reached")
        self.assertEqual(r80["status"], "reached")
        # 容差：累计分组按像素半径量化，径向最坏偏差约为一个像素间距量级，
        # 因此用 2 个网格间距作容差，并在报告里写明这是离散量化所致、不插值提高精度。
        tol = 2.0 * max(self.grid.dx, self.grid.dy)
        self.assertLess(abs(r50["radius_m"] - exp50), tol,
                        "R50 数值 %.6g 解析 %.6g，差 %.3e m（容差 %.3e m）"
                        % (r50["radius_m"], exp50, abs(r50["radius_m"] - exp50), tol))
        self.assertLess(abs(r80["radius_m"] - exp80), tol,
                        "R80 数值 %.6g 解析 %.6g，差 %.3e m（容差 %.3e m）"
                        % (r80["radius_m"], exp80, abs(r80["radius_m"] - exp80), tol))

    def test_not_reached_reported_when_window_too_small(self):
        """窗口不足时必须报 not_reached，**不插值**、不假装测出。"""
        r99 = find_first_radius_reaching(self.I, self.grid, 1e6 * self.pin_analytic, 0.5)
        self.assertEqual(r99["status"], "not_reached")
        self.assertIsNone(r99["radius_m"])
        self.assertIsNotNone(r99["Rlimit_m"])

    def test_threshold_area_radius_matches_analytic(self):
        """高斯阈值面积 A_q = −2πσ²ln(q)，r_eq = σ·sqrt(−2 ln q)。"""
        ipeak = float(np.max(self.I))
        for q in (0.5, 0.1):
            with self.subTest(q=q):
                rec = threshold_area_radius(self.I, self.grid, q, peak=ipeak,
                                            r_max=0.5e-3)
                expect = self.sigma * math.sqrt(-2.0 * math.log(q))
                self.assertLess(abs(rec["r_equiv_m"] - expect) / expect, 2e-3,
                                "q=%.1f: 数值 %.6g 解析 %.6g"
                                % (q, rec["r_equiv_m"], expect))

    def test_threshold_area_truncated_by_roi(self):
        """固定 ROI 截断必须真的生效并如实反映在面积上。"""
        ipeak = float(np.max(self.I))
        full = threshold_area_radius(self.I, self.grid, 0.1, peak=ipeak, r_max=1.0)
        small = threshold_area_radius(self.I, self.grid, 0.1, peak=ipeak,
                                      r_max=1.5 * self.sigma)
        self.assertLess(small["area_m2"], full["area_m2"])
        self.assertEqual(small["r_max_m"], 1.5 * self.sigma)


# ======================================================================================
# 3. 尺寸统计与输入校验
# ======================================================================================
class TestSizeSummaryAndValidation(unittest.TestCase):
    def test_summarize_reports_valid_sample_count(self):
        s = summarize_size_stability([1.0, 2.0, None, 3.0, float("nan")])
        self.assertEqual(s["n_valid"], 3)
        self.assertAlmostEqual(s["mean"], 2.0)
        self.assertAlmostEqual(s["min"], 1.0)
        self.assertAlmostEqual(s["max"], 3.0)
        self.assertAlmostEqual(s["range"], 2.0)

    def test_summarize_empty(self):
        s = summarize_size_stability([None, float("nan")])
        self.assertEqual(s["n_valid"], 0)
        self.assertIsNone(s["mean"])

    def test_nan_intensity_rejected(self):
        g = make_grid(101, 1e-6)
        I = gaussian_intensity(g, 5e-5)
        I[3, 4] = float("nan")
        with self.assertRaises(ValueError):
            c3_rotation_metric(I, g, 10e-6, 40e-6)
        with self.assertRaises(ValueError):
            absolute_encircled_energy(I, g, 1.0, [1e-5])
        with self.assertRaises(ValueError):
            find_first_radius_reaching(I, g, 1.0, 0.5)
        with self.assertRaises(ValueError):
            threshold_area_radius(I, g, 0.5)

    def test_wrong_shape_rejected(self):
        g = make_grid(101, 1e-6)
        I = np.zeros((50, 50))
        for fn in (lambda: c3_rotation_metric(I, g, 10e-6, 40e-6),
                   lambda: absolute_encircled_energy(I, g, 1.0, [1e-5]),
                   lambda: threshold_area_radius(I, g, 0.5),
                   lambda: find_first_radius_reaching(I, g, 1.0, 0.5)):
            with self.assertRaises(ValueError):
                fn()

    def test_zero_energy_and_bad_pin_rejected(self):
        g = make_grid(101, 1e-6)
        Z = np.zeros(g.shape)
        m = c3_rotation_metric(Z, g, 10e-6, 40e-6)
        self.assertFalse(m["reliable"])
        with self.assertRaises(ValueError):
            absolute_encircled_energy(gaussian_intensity(g, 5e-5), g, 0.0, [1e-5])
        with self.assertRaises(ValueError):
            find_first_radius_reaching(gaussian_intensity(g, 5e-5), g, -1.0, 0.5)


# ======================================================================================
# 4. 身份集合与保存后重读的拒绝路径（用主入口模块的真实函数）
# ======================================================================================
class TestIdentitySetAndReverify(unittest.TestCase):
    """用真实 ``main_stage02b`` 函数检查“伪造数据必须被拒绝”。

    这里构造一个**完整的小型 run 目录**（301×301 太慢，改用配置副本里的 9 个波长但
    实际写小网格是不可行的——因为重读会核对坐标），因此本测试只针对
    ``verify_saved_identity_set`` 的**判定逻辑**：用真实网格尺寸但只写少量文件，
    验证“缺文件/重复/错 λ/错器件/伪换高度”都会被判 fail。
    """

    @classmethod
    def setUpClass(cls):
        sys.argv = ["main_stage02b.py"]
        import main_stage02b as M
        importlib.reload(M)
        cls.M = M
        cfg_path = PROJECT_ROOT / "config_stage02b.json"
        cls.cfg = M.load_and_validate_config(cfg_path)

    def _grids(self):
        gc = self.cfg["grid"]
        gi = Grid2D(Axis1D(gc["input"]["n"], gc["input"]["spacing_m"], "x'"),
                    Axis1D(gc["input"]["n"], gc["input"]["spacing_m"], "y'"))
        go = Grid2D(Axis1D(gc["output"]["n"], gc["output"]["spacing_m"], "x"),
                    Axis1D(gc["output"]["n"], gc["output"]["spacing_m"], "y"))
        return gi, go

    def _write_minimal(self, root: Path, gi, go, fingerprints, profiles,
                       *, drop=None, wrong_lambda=None, forged_duplicate=False,
                       fake_height=False, bad_eta=False):
        """写一份合成 run 目录，可按需注入缺陷。

        ``forged_duplicate``：把 ``psf_fresnel_420nm.npz`` 的**文件名保持不变**、
        但内部身份字段改成 ``jeon@660nm``；同时正常写出 ``psf_jeon_660nm.npz``，
        于是重读会看到 ``jeon@660nm`` 这个身份**出现两次**。
        这正是“重命名/复制文件充数”的伪造方式，必须被身份集合检查拒绝。
        """
        M = self.M
        arrays = root / "arrays"
        arrays.mkdir(parents=True, exist_ok=True)
        for key in ("fresnel", "jeon"):
            M.save_height_arrays(root, key, profiles[key], fingerprints[key], gi, self.cfg)
        if fake_height:
            p = arrays / "height_jeon.npz"
            with np.load(p, allow_pickle=False) as d:
                payload = {k: d[k] for k in d.files}
            payload["delta_h_m"] = payload["delta_h_m"] * 0.5
            np.savez_compressed(p, **payload)
        lams = [float(x) for x in self.cfg["optical"]["incident_wavelengths_m"]]
        for key in ("fresnel", "jeon"):
            for lam in lams:
                nm = round(lam * 1e9)
                if drop is not None and (key, nm) == drop:
                    continue
                lam_write, key_write = lam, key
                if wrong_lambda is not None and (key, nm) == wrong_lambda[0]:
                    lam_write = wrong_lambda[1]
                if forged_duplicate and (key, nm) == ("fresnel", 420):
                    # 文件名仍是 fresnel@420，但内部身份写成 jeon@660
                    lam_write = 660e-9
                    key_write = "jeon"
                I = np.full(go.shape, 1.0, dtype=np.float64)
                u2 = np.ones(go.shape, dtype=np.complex128)
                # 合成样本必须**真实自洽**（审核 R2 明确要求）：Pin 由该高度与该 λ 重建
                # u1 得到，Pwindow 由写入的强度积分得到。这样"正常样本通过"才有意义，
                # 而不是靠编造一个与强度无关的功率数值。
                u1_syn = self.M.compute_doe_transmission_field(
                    profiles[key], lam_write,
                    amplitude=float(self.cfg["optical"].get("amplitude", 1.0)),
                    h_offset=float(self.cfg["optical"].get("h_offset_m", 0.0)))
                pin = float(np.sum(np.abs(u1_syn) ** 2) * gi.cell_area)
                pw = float(np.sum(I) * go.cell_area)
                if bad_eta:
                    pw = pin * 1.5          # 只把保存值改成越界，制造字段不一致
                n_lam = float(refractive_index_fused_silica(lam_write))
                M.save_psf(root, key_write, lam_write, {
                    "u2": u2, "intensity": I, "x_out_m": go.x.coords,
                    "y_out_m": go.y.coords, "device_fingerprint": fingerprints[key],
                    "refractive_index": n_lam, "Pin": pin, "Pwindow": pw,
                    "include_global_phase": bool(
                        self.cfg["propagation"]["include_global_phase"]),
                    "amplitude": float(self.cfg["optical"].get("amplitude", 1.0)),
                    "eta_window": pw / pin,
                    "propagation_distance_m": float(self.cfg["optical"]["distance_m"]),
                    "design_focal_length_m": float(self.cfg["optical"]["focal_length_m"]),
                    "peak_intensity": float(np.max(I)),
                    "t_propagation_s": 0.0})

    def _run_verify(self, mutator, expect_fail=True):
        # R8：证据目录**永久保留**，不自动删除
        tmp = retained_evidence_dir("jeon2019_02b1_verify")
        gi, go = self._grids()
        profiles = {
            "jeon": self.M.design_jeon2019_spiral_height(
                gi, float(self.cfg["optical"]["diameter_m"]),
                float(self.cfg["optical"]["focal_length_m"]),
                int(self.cfg["optical"]["wings_N"]),
                float(self.cfg["optical"]["design_wavelength_min_m"]),
                float(self.cfg["optical"]["design_wavelength_max_m"])),
            "fresnel": self.M.design_conventional_fresnel_height(
                gi, float(self.cfg["optical"]["diameter_m"]),
                float(self.cfg["optical"]["focal_length_m"]),
                float(self.cfg["optical"]["conventional_design_wavelength_m"])),
        }
        fps = {k: v.compute_fingerprint() for k, v in profiles.items()}
        kw = {}
        if mutator:
            mutator(kw)
        self._write_minimal(tmp, gi, go, fps, profiles, **kw)
        res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps, profiles)
        write_evidence(tmp, "verify_result.json",
                       json.dumps(res, ensure_ascii=False, indent=2, default=str))
        if expect_fail:
            self.assertEqual(res["status"], "fail", "伪造数据必须被拒绝")
            self.assertTrue(res["problems"])
        return res

    def test_complete_synthetic_set_passes(self):
        res = self._run_verify(None, expect_fail=False)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["verified_identity_count"], 18)

    def test_missing_one_wavelength_rejected(self):
        self._run_verify(lambda kw: kw.update(drop=("jeon", 540)))

    def test_wrong_wavelength_label_rejected(self):
        self._run_verify(lambda kw: kw.update(wrong_lambda=(("jeon", 540), 541e-9)))

    def test_duplicate_identity_rejected(self):
        self._run_verify(lambda kw: kw.update(forged_duplicate=True))

    def test_tampered_height_rejected(self):
        self._run_verify(lambda kw: kw.update(fake_height=True))

    def test_bad_eta_on_either_device_rejected(self):
        """两个器件都要检查功率：不能只检查 Jeon。"""
        for key in ("fresnel", "jeon"):
            with self.subTest(device=key):
                # R8：证据目录永久保留，不自动删除
                tmp = retained_evidence_dir("jeon2019_02b1_eta", tag=key)
                gi, go = self._grids()
                profiles = {
                    "jeon": self.M.design_jeon2019_spiral_height(
                        gi, float(self.cfg["optical"]["diameter_m"]),
                        float(self.cfg["optical"]["focal_length_m"]),
                        int(self.cfg["optical"]["wings_N"]),
                        float(self.cfg["optical"]["design_wavelength_min_m"]),
                        float(self.cfg["optical"]["design_wavelength_max_m"])),
                    "fresnel": self.M.design_conventional_fresnel_height(
                        gi, float(self.cfg["optical"]["diameter_m"]),
                        float(self.cfg["optical"]["focal_length_m"]),
                        float(self.cfg["optical"]["conventional_design_wavelength_m"])),
                }
                fps = {k: v.compute_fingerprint() for k, v in profiles.items()}
                self._write_minimal(tmp, gi, go, fps, profiles)
                # 只把该器件的一个文件改成 η 越界
                lam = float(self.cfg["optical"]["incident_wavelengths_m"][0])
                I = np.ones(go.shape)
                pin = float(np.sum(I) * gi.cell_area)
                self.M.save_psf(tmp, key, lam, {
                    "u2": np.ones(go.shape, dtype=np.complex128), "intensity": I,
                    "x_out_m": go.x.coords, "y_out_m": go.y.coords,
                    "device_fingerprint": fps[key], "refractive_index": 1.46,
                    "Pin": pin, "Pwindow": pin * 1.5,
                    "include_global_phase": bool(
                        self.cfg["propagation"]["include_global_phase"]),
                    "amplitude": 1.0, "eta_window": 1.5,
                    "propagation_distance_m": float(self.cfg["optical"]["distance_m"]),
                    "design_focal_length_m": float(self.cfg["optical"]["focal_length_m"]),
                    "peak_intensity": 1.0, "t_propagation_s": 0.0})
                res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps,
                                                       profiles)
                write_evidence(tmp, "verify_result.json",
                               json.dumps(res, ensure_ascii=False, indent=2,
                                          default=str))
                self.assertEqual(res["status"], "fail")


# ======================================================================================
# 5. 配置校验与运行模式
# ======================================================================================
class TestConfigAndModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.argv = ["main_stage02b.py"]
        import main_stage02b as M
        importlib.reload(M)
        cls.M = M
        cls.base = json.loads((PROJECT_ROOT / "config_stage02b.json").read_text("utf-8"))

    def _write_cfg(self, tmp: Path, mutate) -> Path:
        cfg = json.loads(json.dumps(self.base))
        mutate(cfg)
        p = tmp / "cfg.json"
        p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        return p

    def test_valid_config_loads(self):
        cfg = self.M.load_and_validate_config(PROJECT_ROOT / "config_stage02b.json")
        self.assertEqual(cfg["_derived"]["wavelengths_nm"],
                         [420, 450, 480, 510, 540, 570, 600, 630, 660])

    def test_grid_inconsistency_rejected(self):
        # R8：证据目录永久保留，不自动删除
        tmp = retained_evidence_dir("jeon2019_02b1_cfg_grid")
        p = self._write_cfg(tmp, lambda c: c["grid"]["input"].update(
            {"half_width_m": 0.0006}))       # 与 (n//2)*d 不自洽
        with self.assertRaises(ValueError):
            self.M.load_and_validate_config(p)
        self.assertTrue((tmp / "cfg.json").exists(), "证据必须留在磁盘上")

    def test_aperture_not_covered_rejected(self):
        tmp = retained_evidence_dir("jeon2019_02b1_cfg_aperture")

        def mut(c):
            c["grid"]["input"].update({"n": 301, "spacing_m": 1e-6,
                                       "half_width_m": 150e-6})
        p = self._write_cfg(tmp, mut)
        with self.assertRaises(ValueError):
            self.M.load_and_validate_config(p)
        self.assertTrue((tmp / "cfg.json").exists(), "证据必须留在磁盘上")

    def test_wrong_wavelength_count_rejected(self):
        tmp = retained_evidence_dir("jeon2019_02b1_cfg_wavelengths")
        p = self._write_cfg(tmp, lambda c: c["optical"].update(
            {"incident_wavelengths_m": [5.4e-7]}))
        with self.assertRaises(ValueError):
            self.M.load_and_validate_config(p)
        self.assertTrue((tmp / "cfg.json").exists(), "证据必须留在磁盘上")

    def test_wings_N_must_be_three(self):
        tmp = retained_evidence_dir("jeon2019_02b1_cfg_wings")
        p = self._write_cfg(tmp, lambda c: c["optical"].update({"wings_N": 1}))
        with self.assertRaises(ValueError):
            self.M.load_and_validate_config(p)
        self.assertTrue((tmp / "cfg.json").exists(), "证据必须留在磁盘上")

    def test_required_checks_respect_save_mode(self):
        all_save = self.M.required_checks_02b("all", True)
        all_nosave = self.M.required_checks_02b("all", False)
        self.assertIn("identity_set_exact", all_save)
        self.assertNotIn("identity_set_exact", all_nosave)
        self.assertIn("height_generated", all_nosave)
        self.assertEqual(self.M.required_checks_02b("height"),
                         {"height_generated", "aperture_area", "local_phase_err",
                          "device_design_lambda", "height_fingerprint_stable"})

    def test_identity_expected_size_is_18(self):
        cfg = self.M.load_and_validate_config(PROJECT_ROOT / "config_stage02b.json")
        self.assertEqual(int(cfg["acceptance"]["identity_set_expected_size"]), 18)


# ======================================================================================
# 6. 修正批反例：R1 圆盘边界、R2 伪造功率、R3 传播参数、R7 不可靠角度
# ======================================================================================
class TestAuditCounterexamples(unittest.TestCase):
    """逐条复现审核给出的反例，确认修正后行为正确。"""

    @classmethod
    def setUpClass(cls):
        sys.argv = ["main_stage02b.py"]
        import main_stage02b as M
        importlib.reload(M)
        cls.M = M
        cls.cfg = M.load_and_validate_config(PROJECT_ROOT / "config_stage02b.json")

    # ---------------- R1 ----------------
    def test_r1_corner_only_field_returns_not_reached(self):
        """审核反例：强度只在 r>155 μm 的方窗角落时，150 μm 圆盘能量为 0。

        旧实现返回 ``R80=185.31 μm`` 且 ``reached``；修正后必须 ``not_reached``、
        ``radius_m=None``、``Eabs_at_Rlimit=0``。
        """
        g = make_grid(301, 1e-6)
        X, Y = g.meshgrid()
        r = np.hypot(X, Y)
        I = np.where(r > 155e-6, 1.0, 0.0)
        pin = float(np.sum(I) * g.cell_area)
        res = find_first_radius_reaching(I, g, pin, 0.8)
        self.assertEqual(res["status"], "not_reached")
        self.assertIsNone(res["radius_m"])
        self.assertAlmostEqual(res["Eabs_at_Rlimit"], 0.0, places=15)
        self.assertAlmostEqual(res["Rlimit_m"], 150e-6, places=15)

    def test_r1_square_reaches_but_inscribed_circle_does_not(self):
        """宽分布：方窗内达标但内切圆（150 μm）内不达标 —— 必须 not_reached。"""
        g = make_grid(301, 1e-6)
        X, Y = g.meshgrid()
        r = np.hypot(X, Y)
        # 能量全部落在 150 μm 之外、方窗之内（角区环带）
        I = np.where((r > 150e-6) & (r <= 212e-6), 1.0, 0.0)
        pin_total = float(np.sum(I) * g.cell_area)     # 用全平面做分母
        res = find_first_radius_reaching(I, g, pin_total, 0.5)
        self.assertEqual(res["status"], "not_reached",
                         "圆盘内不该达标：%r" % (res,))

    def test_r1_radius_scale_is_not_float_noise(self):
        """径向分辨率必须是明确采样尺度，不是约 1e-20 m 的浮点最小差。"""
        g = make_grid(301, 1e-6)
        X, Y = g.meshgrid()
        I = np.exp(-(np.hypot(X, Y) / 1e-4) ** 2)
        pin = float(np.sum(I) * g.cell_area)
        res = find_first_radius_reaching(I, g, pin, 0.5)
        self.assertGreater(res["radius_scale_m"], 1e-9,
                           "采样尺度不该是浮点噪声量级")
        self.assertLessEqual(res["radius_scale_m"], 1e-6 + 1e-18)

    def test_r1_disk_beyond_Rlimit_rejected(self):
        """超过完整圆盘上限的评价半径必须被拒绝，不能给出看似有效的 Eabs。"""
        g = make_grid(301, 1e-6)
        X, Y = g.meshgrid()
        I = np.exp(-(np.hypot(X, Y) / 1e-4) ** 2)
        pin = float(np.sum(I) * g.cell_area)
        with self.assertRaises(ValueError):
            absolute_encircled_energy(I, g, pin, [200e-6])
        # 150 μm 恰好等于 Rlimit，必须允许
        ok = absolute_encircled_energy(I, g, pin, [150e-6])
        self.assertAlmostEqual(ok[0]["Rlimit_m"], 150e-6, places=15)

    def test_r1_gaussian_still_matches_analytic(self):
        """保留高斯解析验证（回归）：修正不得破坏正确行为。"""
        g = make_grid(1201, 0.5e-6)
        sigma, amp = 8.0e-5, 3.0
        I = gaussian_intensity(g, sigma, amp)
        pin = 2.0 * math.pi * sigma ** 2 * amp
        r50 = find_first_radius_reaching(I, g, pin, 0.5)
        exp50 = sigma * math.sqrt(2.0 * math.log(2.0))
        self.assertEqual(r50["status"], "reached")
        self.assertLess(abs(r50["radius_m"] - exp50), 2.0 * g.dx)

    # ---------------- R2 ----------------
    def _build_run(self, mutate_power=None):
        """构造一个自洽的 18 组 run，并可选地篡改某一组的功率字段。"""
        tmp = retained_evidence_dir("jeon2019_02b1_r2")
        (tmp / "arrays").mkdir(exist_ok=True)
        gc = self.cfg["grid"]
        gi = Grid2D(Axis1D(gc["input"]["n"], gc["input"]["spacing_m"], "x'"),
                    Axis1D(gc["input"]["n"], gc["input"]["spacing_m"], "y'"))
        go = Grid2D(Axis1D(gc["output"]["n"], gc["output"]["spacing_m"], "x"),
                    Axis1D(gc["output"]["n"], gc["output"]["spacing_m"], "y"))
        profiles = {
            "jeon": self.M.design_jeon2019_spiral_height(
                gi, float(self.cfg["optical"]["diameter_m"]),
                float(self.cfg["optical"]["focal_length_m"]),
                int(self.cfg["optical"]["wings_N"]),
                float(self.cfg["optical"]["design_wavelength_min_m"]),
                float(self.cfg["optical"]["design_wavelength_max_m"])),
            "fresnel": self.M.design_conventional_fresnel_height(
                gi, float(self.cfg["optical"]["diameter_m"]),
                float(self.cfg["optical"]["focal_length_m"]),
                float(self.cfg["optical"]["conventional_design_wavelength_m"])),
        }
        fps = {k: v.compute_fingerprint() for k, v in profiles.items()}
        for k in ("fresnel", "jeon"):
            self.M.save_height_arrays(tmp, k, profiles[k], fps[k], gi, self.cfg)
        amps = float(self.cfg["optical"].get("amplitude", 1.0))
        hoff = float(self.cfg["optical"].get("h_offset_m", 0.0))
        z = float(self.cfg["optical"]["distance_m"])
        n_lam_list = {}
        for k in ("fresnel", "jeon"):
            for lam in self.cfg["optical"]["incident_wavelengths_m"]:
                lam = float(lam)
                u1 = self.M.compute_doe_transmission_field(profiles[k], lam,
                                                           amplitude=amps, h_offset=hoff)
                pin = float(np.sum(np.abs(u1) ** 2) * gi.cell_area)
                I = np.full(go.shape, 1.0, dtype=np.float64)
                u2 = np.ones(go.shape, dtype=np.complex128)
                pw = float(np.sum(I) * go.cell_area)
                if mutate_power is not None and mutate_power(k, round(lam * 1e9)):
                    pin, pw = pin * 2.0, pw * 2.0
                n_lam_list[(k, round(lam * 1e9))] = (
                    float(refractive_index_fused_silica(lam)), z)
                self.M.save_psf(tmp, k, lam, {
                    "u2": u2, "intensity": I, "x_out_m": go.x.coords,
                    "y_out_m": go.y.coords, "device_fingerprint": fps[k],
                    "refractive_index": float(refractive_index_fused_silica(lam)),
                    "Pin": pin, "Pwindow": pw,
                    "include_global_phase": bool(
                        self.cfg["propagation"]["include_global_phase"]),
                    "amplitude": amps, "eta_window": pw / pin,
                    "propagation_distance_m": z,
                    "design_focal_length_m": float(self.cfg["optical"]["focal_length_m"]),
                    "peak_intensity": float(np.max(I)),
                    "t_propagation_s": 0.0})
        return tmp, gi, go, fps, profiles

    def test_r2_self_consistent_synthetic_passes(self):
        tmp, gi, go, fps, profiles = self._build_run()
        res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps, profiles)
        write_evidence(tmp, "verify_result.json",
                       json.dumps(res, ensure_ascii=False, indent=2, default=str))
        self.assertEqual(res["status"], "pass",
                         "真实自洽的合成样本必须通过：%r" % res["problems"][:5])

    def test_r2_both_power_fields_doubled_rejected(self):
        """审核反例：Pin 与 Pwindow **同时乘 2** 必须被拒绝（旧实现仍 pass）。"""
        tmp, gi, go, fps, profiles = self._build_run(
            mutate_power=lambda k, nm: (k, nm) == ("jeon", 540))
        res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps, profiles)
        write_evidence(tmp, "verify_result.json",
                       json.dumps(res, ensure_ascii=False, indent=2, default=str))
        self.assertEqual(res["status"], "fail", "同时放大两项功率必须被拒绝")
        self.assertTrue(any("Pin" in m for m in res["problems"]), res["problems"])

    def test_r2_single_power_field_changed_rejected(self):
        """只改 Pwindow（保持 Pin）也必须被拒绝：重积分会发现不符。"""
        tmp, gi, go, fps, profiles = self._build_run()
        lam = 540e-9
        p = tmp / "arrays" / self.M.psf_array_name("jeon", lam)
        with np.load(p, allow_pickle=False) as d:
            payload = {k: d[k] for k in d.files}
        payload["Pwindow"] = np.array(float(payload["Pwindow"]) * 2.0)
        np.savez_compressed(p, **payload)
        res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps, profiles)
        write_evidence(tmp, "verify_result.json",
                       json.dumps(res, ensure_ascii=False, indent=2, default=str))
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("Pwindow" in m for m in res["problems"]), res["problems"])

    def test_r2_metadata_tampering_rejected(self):
        """改 η / n(λ) / 振幅 / y 轴 / 设计 λ / dtype 都必须被拒绝。"""
        cases = {
            "eta": ("eta_window", lambda v: np.array(0.123456)),
            "n_lambda": ("refractive_index", lambda v: np.array(1.999)),
            "amplitude": ("amplitude", lambda v: np.array(7.0)),
            "y_axis": ("y_out_m", lambda v: v + 1e-6),
            "dtype": ("intensity_raw", lambda v: np.asarray(v, dtype=np.float32)),
        }
        for tag, (field, mut) in cases.items():
            with self.subTest(field=tag):
                tmp, gi, go, fps, profiles = self._build_run()
                lam = 540e-9
                p = tmp / "arrays" / self.M.psf_array_name("jeon", lam)
                with np.load(p, allow_pickle=False) as d:
                    payload = {k: d[k] for k in d.files}
                payload[field] = mut(payload[field])
                np.savez_compressed(p, **payload)
                res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps,
                                                       profiles)
                write_evidence(tmp, "verify_result_%s.json" % tag,
                               json.dumps(res, ensure_ascii=False, indent=2,
                                          default=str))
                self.assertEqual(res["status"], "fail",
                                 "%s 被篡改后必须拒绝：%r" % (tag, res["problems"][:3]))

    def test_r2_design_lambda_tampering_rejected(self):
        """改高度文件里的设计 λ 图（保留旧指纹字段）必须被独立重算发现。"""
        tmp, gi, go, fps, profiles = self._build_run()
        p = tmp / "arrays" / "height_jeon.npz"
        with np.load(p, allow_pickle=False) as d:
            payload = {k: d[k] for k in d.files}
        payload["lambda_design_m"] = payload["lambda_design_m"] * 1.001
        np.savez_compressed(p, **payload)
        res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps, profiles)
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("lambda_design" in m for m in res["problems"]),
                        res["problems"])

    def test_r2_tampered_height_content_rejected(self):
        """改高度内容（不改指纹字段）必须被独立重算指纹发现。"""
        tmp, gi, go, fps, profiles = self._build_run()
        p = tmp / "arrays" / "height_jeon.npz"
        with np.load(p, allow_pickle=False) as d:
            payload = {k: d[k] for k in d.files}
        payload["delta_h_m"] = payload["delta_h_m"] * 0.5
        np.savez_compressed(p, **payload)
        res = self.M.verify_saved_identity_set(tmp, self.cfg, gi, go, fps, profiles)
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("指纹" in m or "delta_h" in m for m in res["problems"]),
                        res["problems"])

    # ---------------- R3 ----------------
    def test_r3_propagation_uses_configured_distance(self):
        """小网格反例：z=60 mm 必须与 z=50 mm 不同，且与直接参考一致。"""
        gi = Grid2D(Axis1D(401, 1e-6, "x'"), Axis1D(401, 1e-6, "y'"))
        go = Grid2D(Axis1D(101, 2e-6, "x"), Axis1D(101, 2e-6, "y"))
        prof = self.M.design_jeon2019_spiral_height(gi, 1e-3, 50e-3, 3, 420e-9, 660e-9)
        logger = self.M.setup_logger(None, "ERROR")
        z50 = 50e-3
        z60 = 60e-3
        r50 = self.M.propagate_device(prof, 540e-9, gi, go, True, 1.0, 0.0, z50,
                                      logger, "test")
        r60 = self.M.propagate_device(prof, 540e-9, gi, go, True, 1.0, 0.0, z60,
                                      logger, "test")
        self.assertEqual(r50["distance_m"], z50)
        self.assertEqual(r60["distance_m"], z60)
        rel = float(np.sqrt(np.sum(np.abs(r50["u2"] - r60["u2"]) ** 2))
                    / np.sqrt(np.sum(np.abs(r50["u2"]) ** 2)))
        self.assertGreater(rel, 0.5, "改变配置 z 必须显著改变结果")
        # 与直接调用传播核一致（同一 z、同一网格）
        ref = fresnel_kernel_separable(r50["u1"], gi, 540e-9, z50,
                                       go.x.coords, go.y.coords)
        self.assertTrue(np.array_equal(r50["u2"], ref))

    def test_r3_psf_records_both_z_and_f(self):
        """PSF 必须同时记录实际传播距离与器件设计焦距。"""
        gi = Grid2D(Axis1D(201, 2e-6, "x'"), Axis1D(201, 2e-6, "y'"))
        go = Grid2D(Axis1D(51, 4e-6, "x"), Axis1D(51, 4e-6, "y"))
        prof = self.M.design_jeon2019_spiral_height(gi, 1e-3, 50e-3, 3, 420e-9, 660e-9)
        logger = self.M.setup_logger(None, "ERROR")
        r = self.M.propagate_device(prof, 540e-9, gi, go, True, 1.0, 0.0, 60e-3,
                                    logger, "test")
        self.assertEqual(r["distance_m"], 60e-3)
        self.assertEqual(r["design_focal_length_m"], 50e-3)

    # ---------------- R7 ----------------
    def test_r7_unreliable_alpha_is_null(self):
        """轴对称高斯：A3 极低 → alpha 必须为 null，原始相位另名且标注无效。"""
        g = make_grid(401, 1e-6)
        I = gaussian_intensity(g, 1.2e-4)
        m = c3_rotation_metric(I, g, 30e-6, 120e-6, a3_min=0.05)
        self.assertFalse(m["reliable"])
        self.assertIsNone(m["alpha_wrapped_rad"])
        self.assertIsNone(m["alpha_wrapped_deg"])
        self.assertIsNotNone(m["alpha_raw_rad"])
        self.assertIn("不是", m["alpha_raw_note"])

    def test_r7_reliable_alpha_present(self):
        """可靠时 alpha 必须给出数值（不能一律置空）。"""
        g = make_grid(401, 1e-6)
        I = three_fold_intensity(g, math.radians(25.0))
        m = c3_rotation_metric(I, g, 30e-6, 120e-6, a3_min=0.05)
        self.assertTrue(m["reliable"])
        self.assertIsNotNone(m["alpha_wrapped_deg"])
        self.assertAlmostEqual((m["alpha_wrapped_deg"] - 25.0 + 60) % 120 - 60, 0.0,
                               places=6)

    def test_r7_empty_band_and_mixed_sequence(self):
        """无环带能量、以及可靠/不可靠混合序列都必须正常处理，不崩溃。"""
        g = make_grid(201, 1e-6)
        zero = np.zeros(g.shape)
        m = c3_rotation_metric(zero, g, 30e-6, 120e-6, a3_min=0.05)
        self.assertFalse(m["reliable"])
        self.assertIsNone(m["alpha_wrapped_deg"])
        # 混合序列：不可靠点必须为 None，且展开不跨间隙
        alphas = [0.0, math.radians(20.0), math.radians(40.0), math.radians(60.0)]
        rel = [True, False, True, True]
        u = unwrap_angles_over_reliable_segments(
            [420e-9, 450e-9, 480e-9, 510e-9], alphas, rel)
        self.assertIsNone(u["alpha_unwrapped_rad"][1])
        self.assertEqual(len(u["segments"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
