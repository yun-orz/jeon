# -*- coding: utf-8 -*-
"""G2物理单位、边界及伴随的独立小尺寸验证。"""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.signal import convolve2d
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g2_rgb import RGBForward,synthetic_response


class TestG2(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(53)
        self.k=self.rng.random((2,3,5));self.k/=self.k.sum(axis=(1,2))[:,None,None]*1.25
        self.r=self.rng.random((3,2));self.op=RGBForward(self.k,self.r)

    def test_forward_direct_reference(self):
        x=self.rng.random((8,7,2));bands=np.stack([convolve2d(x[:,:,b],self.k[b],mode='same',boundary='fill') for b in range(2)],axis=-1)
        np.testing.assert_allclose(self.op.forward(x),bands@self.r.T,rtol=1e-13,atol=1e-14)

    def test_adjoint_boundary_and_even_images(self):
        for h,w in [(4,5),(5,4),(4,4),(5,5)]:
            x=self.rng.normal(size=(h,w,2));y=self.rng.normal(size=(h,w,3))
            self.assertAlmostEqual(float(np.sum(self.op.forward(x)*y)),float(np.sum(x*self.op.adjoint(y))),delta=1e-12)
        # 非对称核与边角单位脉冲独立验证裁剪转置。
        y=np.zeros((8,7,3));y[0,0,1]=1.
        expected=np.zeros((8,7,2))
        for b in range(2):
            for iy in range(8):
                for ix in range(7):
                    ky=1-iy;kx=2-ix
                    if 0<=ky<3 and 0<=kx<5:expected[iy,ix,b]=self.r[1,b]*self.k[b,ky,kx]
        np.testing.assert_allclose(self.op.adjoint(y),expected,atol=1e-15)

    def test_point_flux_and_crop_loss(self):
        x=np.zeros((11,13,2));x[5,6,:]=1.;full=self.r@self.k.sum(axis=(1,2))
        np.testing.assert_allclose(self.op.forward(x).sum(axis=(0,1)),full,atol=1e-14)
        x[:]=0.;x[0,0,:]=1.;self.assertTrue(np.all(self.op.forward(x).sum(axis=(0,1))<full))

    def test_single_band_and_incoherent_linearity(self):
        a=self.rng.random((8,7,2));a[:,:,1]=0.;b=self.rng.random((8,7,2));b[:,:,0]=0.
        np.testing.assert_allclose(self.op.forward(a+b),self.op.forward(a)+self.op.forward(b),atol=1e-14)
        image=convolve2d(a[:,:,0],self.k[0],mode='same')
        np.testing.assert_allclose(self.op.forward(a),image[:,:,None]*self.r[:,0],atol=1e-14)

    def test_density_bandwidth_and_zero_response(self):
        x=self.rng.random((8,7,2));density=RGBForward(self.k,self.r,'photons_per_nm',10.)
        np.testing.assert_allclose(density.forward(x),self.op.forward(x*10),atol=1e-13)
        self.assertTrue(np.array_equal(RGBForward(self.k,np.zeros((3,2))).forward(x),np.zeros((8,7,3))))
        self.assertTrue(np.array_equal(self.op.forward(np.zeros_like(x)),np.zeros((8,7,3))))

    def test_invalid_data(self):
        for k,r in [(self.k[:,::2,:],self.r),(self.k,self.r[:2]),(self.k,-self.r),(self.k*np.nan,self.r)]:
            with self.assertRaises(ValueError):RGBForward(k,r)
        with self.assertRaises(ValueError):self.op.forward(np.zeros((2,2,3)))
        with self.assertRaises(ValueError):self.op.adjoint(np.full((2,2,3),np.inf))
        with self.assertRaises(ValueError):self.op.check_cube(-np.ones((2,2,2)),physical=True)
        with self.assertRaises(ValueError):RGBForward(self.k,self.r,'power')

    def test_response_is_predeclared(self):
        c=dict(kind='synthetic_gaussian',centers_nm=[610,540,460],sigma_nm=[45,35,30],peak_qe=[.6]*3)
        r=synthetic_response(np.array([460,540,610])*1e-9,c)
        self.assertAlmostEqual(r[0,2],.6);self.assertAlmostEqual(r[1,1],.6);self.assertAlmostEqual(r[2,0],.6)
        with self.assertRaises(ValueError):synthetic_response(np.array([540,460])*1e-9,c)


if __name__=='__main__':unittest.main()
