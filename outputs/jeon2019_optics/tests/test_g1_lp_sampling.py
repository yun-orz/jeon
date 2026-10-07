# -*- coding: utf-8 -*-
"""解析Fresnel原函数与独立Gauss求积交叉检查。"""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g1_lp_sampling import cell_matrix


class TestCellIntegral(unittest.TestCase):
    def test_matches_independent_gauss_integral(self):
        nodes,weights=np.polynomial.legendre.leggauss(24)
        inputs=np.array([-.5e-3,0.,.4e-3]);outputs=np.array([-200e-6,0.,100e-6])
        dx=1e-6;wave=540e-9;distance=.05
        shifts=outputs[:,None,None]-inputs[None,:,None]+dx*nodes
        independent=dx*np.sum(weights*np.exp(1j*np.pi*shifts**2/(wave*distance)),axis=-1)/2
        np.testing.assert_allclose(cell_matrix(outputs,inputs,dx,wave,distance),independent,atol=1e-18,rtol=1e-10)

    def test_small_cell_approaches_point_kernel(self):
        dx=1e-9;wave=540e-9;distance=.05
        result=cell_matrix(np.array([20e-6]),np.array([300e-6]),dx,wave,distance)[0,0]/dx
        expected=np.exp(1j*np.pi*(20e-6-300e-6)**2/(wave*distance))
        self.assertLess(abs(result-expected),1e-8)

    def test_rejects_invalid_physical_units(self):
        with self.assertRaises(ValueError):cell_matrix([0.],[0.],0.,540e-9,.05)
        with self.assertRaises(ValueError):cell_matrix([np.nan],[0.],1e-6,540e-9,.05)


if __name__=='__main__':unittest.main()
