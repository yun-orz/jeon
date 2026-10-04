# -*- coding: utf-8 -*-
"""用非对称解析核验证平移、裁剪、伴随与通量。"""
import unittest
import numpy as np
from optics.imaging import SpectralImager, direct_shift_sum, operator_checks


class ImagingTests(unittest.TestCase):
    def setUp(self):
        self.k = np.arange(1,28,dtype=float).reshape(3,3,3)/100
        self.op = SpectralImager(self.k,[4,6],2e-6,[1,2,.5])

    def test_corner_impulses_do_not_wrap_or_flip(self):
        for b,y,x in [(0,0,0),(1,3,5),(2,1,2)]:
            cube = np.zeros(self.op.cube_shape); cube[b,y,x] = 1
            expected = np.zeros(self.op.full_shape)
            expected[y:y+3,x:x+3] = self.op.response[b]*self.k[b]
            np.testing.assert_allclose(self.op.forward(cube),expected,atol=2e-15,rtol=2e-14)

    def test_weighted_incoherent_superposition(self):
        cube = np.zeros(self.op.cube_shape); cube[:,1,2] = [1,2,.5]
        np.testing.assert_allclose(self.op.forward(cube),direct_shift_sum(cube,self.k,self.op.response),atol=2e-15)

    def test_signed_direct_reference_and_flux(self):
        x = np.random.default_rng(42).standard_normal(self.op.cube_shape)
        actual = self.op.forward(x)
        np.testing.assert_allclose(actual,direct_shift_sum(x,self.k,self.op.response),atol=1e-14)
        expected = np.dot(x.sum(axis=(1,2)),self.op.response*self.k.sum(axis=(1,2)))
        self.assertAlmostEqual(float(actual.sum()),float(expected),places=12)

    def test_full_and_crop_adjoint_and_gradient(self):
        for crop in [None,[1,5,2,7]]:
            op = SpectralImager(self.k,[4,6],2e-6,[1,2,.5],crop)
            r = operator_checks(op,10,np.ones(op.output_shape))
            self.assertLess(r['inner_product_relative_error'],1e-10)
            self.assertLess(r['gradient_relative_error'],1e-5)

    def test_crop_is_explicit_projection(self):
        op = SpectralImager(self.k,[4,6],2e-6,[1,2,.5],[1,5,2,7])
        cube = np.ones(op.cube_shape)
        np.testing.assert_array_equal(op.forward(cube),self.op.forward(cube)[1:5,2:7])
        y = np.ones(op.output_shape); extended = np.zeros(op.full_shape); extended[1:5,2:7] = y
        np.testing.assert_array_equal(op.adjoint(y),self.op.adjoint(extended))

    def test_even_scene_coordinates_and_impulse_center(self):
        c = self.op.coordinates()
        np.testing.assert_allclose(c['scene_x_m'],-c['scene_x_m'][::-1],atol=1e-20)
        self.assertNotEqual(c['scene_x_m'][3],0.)
        self.assertAlmostEqual(c['full_x_m'][3+1],c['scene_x_m'][3])
        np.testing.assert_allclose(np.diff(c['full_y_m']),2e-6,atol=1e-20)

    def test_zero_scene_exact(self):
        np.testing.assert_array_equal(self.op.forward(np.zeros(self.op.cube_shape)),np.zeros(self.op.full_shape))

    def test_invalid_shapes_and_values_rejected(self):
        for x in [np.zeros((2,4,6)),np.full(self.op.cube_shape,np.nan),np.zeros(self.op.cube_shape,dtype=complex)]:
            with self.assertRaises(ValueError): self.op.forward(x)
        with self.assertRaises(ValueError): self.op.adjoint(np.zeros((4,6)))
        for k in [self.k[:,:,:2],np.zeros_like(self.k),-self.k]:
            with self.assertRaises(ValueError): SpectralImager(k,[4,6],1.,[1,1,1])
        for crop in [[0,0,0,2],[-1,2,0,2],[0,99,0,2],[0,2,0,1.5]]:
            with self.assertRaises(ValueError): SpectralImager(self.k,[4,6],1.,[1,1,1],crop)


if __name__ == '__main__': unittest.main()
