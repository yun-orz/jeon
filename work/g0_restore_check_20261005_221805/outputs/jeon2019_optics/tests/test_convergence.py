# -*- coding: utf-8 -*-
"""采样比较器的独立边界和解析插值验证，不生成或清理文件。"""
import unittest
import numpy as np
from optics.coordinates import make_grid
from optics.convergence import common_indices,compare_intensities,periodic_angle_difference_deg


class TestConvergence(unittest.TestCase):
    def test_three_fold_angle_period(self):
        self.assertAlmostEqual(periodic_angle_difference_deg(59,-59),2)
        self.assertIsNone(periodic_angle_difference_deg(None,1))

    def test_native_extraction_rejects_shift(self):
        g=make_grid(21,1e-6)
        with self.assertRaises(ValueError):common_indices(g.x.coords,g.x.coords+.2e-6)

    def test_linear_intensity_interpolates_exactly(self):
        coarse,fine=make_grid(11,2e-6),make_grid(21,1e-6)
        X,Y=coarse.meshgrid();a=3+X*1e4+Y*2e4
        X,Y=fine.meshgrid();b=3+X*1e4+Y*2e4
        got=compare_intensities(a,coarse,b,fine)
        self.assertLess(got["intensity_l1"],1e-14)
        self.assertLess(got["common_point_relative_max"],1e-14)

    def test_energy_scaling_detected(self):
        g=make_grid(21,1e-6);a=np.ones(g.shape)
        self.assertAlmostEqual(compare_intensities(a*2,g,a,g)["intensity_l1"],1)
