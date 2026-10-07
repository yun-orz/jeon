"""接手后的物理积分及验收漏判回归测试，不删除测试文件。"""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from optics.coordinates import Axis1D, Grid2D, circular_aperture, ideal_thin_lens_phase
from optics.propagation import fresnel_kernel_matrix
from optics.validation import finite_window_energy
from tests.test_negative_paths import run_main


class TestFiniteEnergy(unittest.TestCase):
    def test_off_center_rectangular_support_matches_full_kernel(self):
        """裁去全零输入后仍使用原坐标，不能把非对称支撑重新居中。"""
        g = Grid2D(Axis1D(23, 8e-6, "x"), Axis1D(17, 15e-6, "y"))
        u = np.zeros(g.shape, dtype=complex)
        rng = np.random.default_rng(23)
        u[2:10, 11:19] = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        rec = finite_window_energy(u, g, 550e-9, .05, .2e-3, 150e-6, .5, 10)
        out = rec["grid"]
        ref = fresnel_kernel_matrix(u, g, 550e-9, .05, out.x.coords, out.y.coords)
        self.assertLess(np.linalg.norm(rec["field"] - ref) / np.linalg.norm(ref), 1e-11)
        self.assertEqual(rec["nonzero_support_shape_yx"], [8, 8])
        self.assertAlmostEqual(rec["P_in"], np.sum(abs(u) ** 2) * g.cell_area)

    def test_absolute_airy_energy_and_output_grid_refinement(self):
        g = Grid2D(Axis1D(401, 1e-6, "x"), Axis1D(401, 1e-6, "y"))
        lam, z, D = 550e-9, .05, .1e-3
        u = circular_aperture(g, D) * ideal_thin_lens_phase(g, lam, z)
        radius = 2 * 3.8317059702075125 * lam * z / (np.pi * D)
        records = [finite_window_energy(u, g, lam, z, D, radius, .5, n) for n in (20, 40)]
        for rec in records:
            self.assertLess(rec["absolute_difference"], .005)
            self.assertTrue(rec["window_covers_disk"])
            self.assertTrue(rec["disk_does_not_select_whole_window"])
            self.assertIsNone(rec["P_total_native_fft"])
            self.assertAlmostEqual(rec["E_numeric_disk_over_Pin"], rec["P_disk"] / rec["P_in"])
            np.testing.assert_allclose(rec["radial_P_cum"][-1], rec["P_square_window"], rtol=1e-12)
        self.assertLess(abs(records[0]["E_numeric_disk_over_Pin"] - records[1]["E_numeric_disk_over_Pin"]), .005)
        self.assertAlmostEqual(records[0]["output_spacing_m"] / records[1]["output_spacing_m"], 2)

    def test_zero_input_rejected(self):
        g = Grid2D(Axis1D(9, 1e-6, "x"), Axis1D(9, 1e-6, "y"))
        with self.assertRaises(ValueError):
            finite_window_energy(np.zeros(g.shape), g, 550e-9, .05, .1e-3, 100e-6, .5)


class TestCompletionAcceptance(unittest.TestCase):
    """用内存返回值注入验收漏洞，光学正确性由独立物理测试保证。"""
    def execute(self, group, result, enabled=True):
        def patch(mod):
            mod._CFG_EARLY["studies"]["direct_vs_fft" if group == "direct" else group]["enabled"] = enabled
            name = {"airy": "study_airy", "direct": "study_direct_vs_fft",
                    "energy": "study_energy", "sampling": "study_sampling"}[group]
            setattr(mod, name, lambda *a, **k: copy.deepcopy(result))
        return run_main(["--only", group, "--no-save"], monkeypatch=patch)

    @staticmethod
    def airy():
        return {"cases": [{"name": name, "first_ring_rel_err": .001,
                            "line_L1_vs_airy": .001,
                            "encircled_energy": {"absolute_difference": .0001}}
                           for name in ("D0.1mm_f50mm", "D0.2mm_f25mm", "D1mm_f50mm")]}

    def test_valid_airy_fixture_passes(self):
        self.assertEqual(self.execute("airy", self.airy())["exit_code"], 0)

    def test_non_first_airy_nan_and_missing_measurement_fail(self):
        for value in (float("nan"), float("inf"), None):
            with self.subTest(value=value):
                result = self.airy()
                result["cases"][1]["first_ring_rel_err"] = value
                self.assertNotEqual(self.execute("airy", result)["exit_code"], 0)

    def test_negative_ring_error_is_checked_by_absolute_value(self):
        result = self.airy()
        result["cases"][2]["first_ring_rel_err"] = -.5
        self.assertNotEqual(self.execute("airy", result)["exit_code"], 0)

    def test_missing_paper_scale_case_fails(self):
        result = self.airy()
        result["cases"] = result["cases"][:2]
        self.assertNotEqual(self.execute("airy", result)["exit_code"], 0)

    def test_missing_requested_direct_measurement_fails(self):
        self.assertNotEqual(self.execute("direct", {})["exit_code"], 0)

    def test_disabled_explicitly_requested_group_fails(self):
        self.assertNotEqual(self.execute("airy", self.airy(), enabled=False)["exit_code"], 0)

    def test_bad_energy_window_fails_despite_small_numeric_error(self):
        result = {"parseval_worst_rel_err": 1e-14, "absolute_energy_worst_abs_diff": .001,
                  "absolute_energy_windows_ok": False, "absolute_energy_grid_stable": True}
        self.assertNotEqual(self.execute("energy", result)["exit_code"], 0)

    def test_bad_padding_trend_fails_despite_small_finest_error(self):
        result = {"identity": {"worst_rel_L2": 1e-13},
                  "input_sampling_L1": {"last_pair_raw_intensity_L1": .001},
                  "padding_sampling_L1": {"best": .001, "finer_not_worse": False,
                                          "window_covered": True}}
        self.assertNotEqual(self.execute("sampling", result)["exit_code"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
