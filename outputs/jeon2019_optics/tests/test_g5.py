# -*- coding: utf-8 -*-
"""G5指标独立解析与逐窗口参考检查，可直接在PyCharm运行。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g5_metrics import assess,ssim_band


def reference_ssim(a,b,L=1.):
    """用逐窗口中心化偏差计算方差，避免复用实现的原始二阶矩算法。"""
    y,x=np.mgrid[-5:6,-5:6];w=np.exp(-(x*x+y*y)/(2*1.5**2));w/=w.sum()
    result=np.zeros((a.shape[0]-10,a.shape[1]-10))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            u=a[i:i+11,j:j+11];v=b[i:i+11,j:j+11]
            mu=(w*u).sum();mv=(w*v).sum()
            vu=(w*(u-mu)**2).sum();vv=(w*(v-mv)**2).sum();cov=(w*(u-mu)*(v-mv)).sum()
            result[i,j]=((2*mu*mv+(.01*L)**2)*(2*cov+(.03*L)**2))/((mu*mu+mv*mv+(.01*L)**2)*(vu+vv+(.03*L)**2))
    return result


class TestG5(unittest.TestCase):
    def test_perfect_and_strict_json(self):
        import json
        a=np.ones((11,12,3));r,m=assess(a,a)
        self.assertIsNone(r['psnr_cube_db']);self.assertTrue(r['psnr_cube_is_positive_infinity'])
        self.assertAlmostEqual(r['ssim_mean'],1.);self.assertLess(r['sam_mean_rad'],3e-8)
        json.dumps(r,allow_nan=False)

    def test_scale_angle_and_psnr(self):
        a=np.ones((12,12,2));b=2*a;r,_=assess(a,b)
        self.assertAlmostEqual(r['mse'],1.);self.assertAlmostEqual(r['psnr_cube_db'],0.)
        self.assertLess(r['sam_mean_rad'],3e-8)
        self.assertEqual(r['prediction_above_reference_values'],288)

    def test_orthogonal_signed_and_zero_spectra(self):
        a=np.zeros((11,11,2));a[:,:,0]=1.;b=np.zeros_like(a);b[:,:,1]=1.
        r,_=assess(a,b);self.assertAlmostEqual(r['sam_mean_rad'],np.pi/2)
        r,_=assess(a,-a);self.assertAlmostEqual(r['sam_mean_rad'],np.pi)
        r,m=assess(a,np.zeros_like(a));self.assertIsNone(r['sam_mean_rad'])
        self.assertEqual(r['sam_undefined_pixels'],121);self.assertTrue(np.isnan(m['sam_rad']).all())

    def test_ssim_independent_centered_moments(self):
        rng=np.random.default_rng(15);a=rng.random((13,15));b=rng.normal(.3,.2,(13,15))
        np.testing.assert_allclose(ssim_band(a,b,1.),reference_ssim(a,b),atol=4e-14,rtol=4e-14)

    def test_reference_range_is_not_refitted(self):
        a=np.full((11,11,2),2.);b=a+.1
        r,_=assess(a,b,1.);s,_=assess(a,b,2.)
        self.assertAlmostEqual(s['psnr_cube_db']-r['psnr_cube_db'],20*np.log10(2))
        self.assertEqual(r['target_above_reference_values'],242)

    def test_reject_invalid_inputs(self):
        a=np.ones((11,11,2))
        for b in [a[:,:,:1],a*np.nan]:
            with self.assertRaises(ValueError):assess(a,b)
        for L in [0.,-1.,float('inf'),True]:
            with self.assertRaises(ValueError):assess(a,a,L)
        with self.assertRaises(ValueError):assess(-a,a)
        with self.assertRaises(ValueError):assess(a[:10],a[:10])


if __name__=='__main__':unittest.main()
