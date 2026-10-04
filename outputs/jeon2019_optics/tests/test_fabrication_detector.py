# -*- coding: utf-8 -*-
"""量化边界、对称坐标及解析面积积分的独立验证；不删除文件。"""
import unittest
import numpy as np
from scipy.special import erf
from optics.fabrication_detector import quantize_relative_height,detector_grids,integrate_square_pixels
from optics.coordinates import make_grid
from optics.doe import design_jeon2019_spiral_height,compute_doe_transmission_field
from optics.materials import refractive_index_fused_silica
from optics.propagation import fresnel_kernel_separable


class TestFabricationDetector(unittest.TestCase):
    def test_half_level_rule_and_depth_floor(self):
        h=-np.array([[0,50,150,1450]],dtype=float)*1e-9
        mask=np.ones_like(h)
        nearest,codes=quantize_relative_height(h,mask,100e-9,16,"nearest_depth")
        self.assertEqual(codes.tolist(),[[0,1,2,15]])
        floor,codes=quantize_relative_height(h,mask,100e-9,16,"floor_depth")
        self.assertEqual(codes.tolist(),[[0,0,1,14]])
        self.assertLessEqual(np.abs(nearest-h).max(),50e-9+1e-20)
        self.assertLessEqual(np.abs(floor-h).max(),100e-9+1e-20)

    def test_range_rejected_without_clipping(self):
        for h in (1e-9,-1600e-9):
            with self.assertRaises(ValueError):
                quantize_relative_height(np.array([[h]]),np.ones((1,1)),100e-9,16,"nearest_depth")

    def test_even_nodes_are_symmetric_and_inside_fixed_edges(self):
        camera,nodes,edges=detector_grids(97,6.22e-6,8)
        np.testing.assert_allclose(nodes.x.coords,-nodes.x.coords[::-1],rtol=0,atol=1e-20)
        self.assertEqual(camera.x.coords[48],0)
        self.assertAlmostEqual(edges[-1]*1e6,301.67)
        self.assertTrue(np.all(nodes.x.coords>edges[0]) and np.all(nodes.x.coords<edges[-1]))
        _,_,coarse_edges=detector_grids(97,6.22e-6,4)
        np.testing.assert_array_equal(edges,coarse_edges)

    def test_constant_intensity_integrates_area(self):
        _,nodes,_=detector_grids(7,6.22e-6,4)
        power,average=integrate_square_pixels(np.full(nodes.shape,3.),7,6.22e-6,4)
        np.testing.assert_allclose(power,3*(6.22e-6)**2,rtol=2e-15)
        np.testing.assert_allclose(average,3.,rtol=2e-15)

    def test_gaussian_matches_erf_and_converges(self):
        errors=[]
        pitch,sigma=6.22e-6,20e-6
        for q in (4,8,16):
            _,nodes,edges=detector_grids(25,pitch,q)
            X,Y=nodes.meshgrid()
            power,_=integrate_square_pixels(np.exp(-(X*X+Y*Y)/(2*sigma**2)),25,pitch,q)
            integral=sigma*np.sqrt(np.pi/2)*np.diff(erf(edges/(np.sqrt(2)*sigma)))
            exact=np.outer(integral,integral)
            errors.append(np.abs(power-exact).sum()/exact.sum())
        self.assertLess(errors[-1],5e-5)
        self.assertLess(errors[1],errors[0]*.3)
        self.assertLess(errors[2],errors[1]*.3)

    def test_bad_shape_and_negative_intensity_rejected(self):
        for I in (np.zeros((2,2)),-np.ones((28,28)),np.full((28,28),np.nan)):
            with self.assertRaises(ValueError):integrate_square_pixels(I,7,6.22e-6,4)

    def test_uniform_substrate_only_adds_global_phase(self):
        g=make_grid(101,10e-6);out=make_grid(41,5e-6);lam=540e-9
        p=design_jeon2019_spiral_height(g,1e-3,.05,3,420e-9,660e-9)
        u=compute_doe_transmission_field(p,lam)
        shifted=compute_doe_transmission_field(p,lam,h_offset=.5e-3)
        factor=np.exp(2j*np.pi*(refractive_index_fused_silica(lam)-1)*.5e-3/lam)
        self.assertLess(np.linalg.norm(shifted-u*factor)/np.linalg.norm(u),1e-11)
        a=fresnel_kernel_separable(u,g,lam,.05,out.x.coords,out.y.coords)
        b=fresnel_kernel_separable(shifted,g,lam,.05,out.x.coords,out.y.coords)
        self.assertLess(np.linalg.norm(abs(a)**2-abs(b)**2)/np.linalg.norm(abs(a)**2),1e-11)
