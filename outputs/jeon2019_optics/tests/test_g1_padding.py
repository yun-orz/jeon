# -*- coding: utf-8 -*-
"""解析周期平面模、功率守恒及非法物理输入检查。"""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g1_padding import periodic_fresnel,power_on_grid


class TestPeriodicFresnel(unittest.TestCase):
    def test_plane_mode_analytic_phase(self):
        n=64;dx=1e-6;wave=540e-9;z=.05
        y,x=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
        field=np.exp(2j*np.pi*(3*x-2*y)/n)
        expected=field*np.exp(-1j*np.pi*wave*z*((3/(n*dx))**2+(2/(n*dx))**2))*np.exp(1j*(2*np.pi/wave)*z)
        np.testing.assert_allclose(periodic_fresnel(field,dx,wave,z,13),expected,atol=2e-12,rtol=0)

    def test_power_and_input_preservation(self):
        rng=np.random.default_rng(7);field=rng.normal(size=(64,64))+1j*rng.normal(size=(64,64));old=field.copy()
        output=periodic_fresnel(field,1e-6,660e-9,.05,17)
        self.assertAlmostEqual(power_on_grid(output,1e-6)/power_on_grid(field,1e-6),1.,places=13)
        np.testing.assert_array_equal(field,old)

    def test_invalid_units_and_grid(self):
        with self.assertRaises(ValueError):periodic_fresnel(np.ones((3,3)),1e-6,540e-9,.05)
        with self.assertRaises(ValueError):periodic_fresnel(np.ones((4,4)),1e-6,540e-9,0.)


if __name__=='__main__':unittest.main()
