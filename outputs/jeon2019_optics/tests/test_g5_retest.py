# -*- coding: utf-8 -*-
"""测量一致性须包含ROI外的光，零信号和带符号预测显式处理。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g2_rgb import RGBForward
from optics.g5_retest_metrics import measurement_fit,summary_difference


class TestRetest(unittest.TestCase):
    def test_float32_measurements_use_float64_norm(self):
        op=RGBForward(np.ones((1,1,1)),np.ones((3,1)))
        p=np.full((1,3,3,1),.3,dtype=np.float32)
        y=np.linspace(.1,.7,27,dtype=np.float32).reshape(1,3,3,3)
        row,rgb=measurement_fit(op,p,y,0,3)
        reference=np.linalg.norm((rgb-y.astype(float)).ravel())/np.linalg.norm(y.astype(float).ravel())
        self.assertAlmostEqual(row['center_relative_l2'],float(reference),places=14)

    def test_float_roundoff_does_not_hide_count_errors(self):
        expected=dict(sam=.493,valid=4096,clipped=False,undefined=None)
        actual=dict(expected,sam=float(np.nextafter(.493,1.)))
        self.assertLess(summary_difference(actual,expected),1e-15)
        for bad in [dict(expected,sam=.494),dict(expected,valid=4095),dict(expected,clipped=0)]:
            with self.assertRaises(ValueError):summary_difference(bad,expected)

    def test_halo_contributes_to_center(self):
        op=RGBForward(np.ones((1,3,3))/9,np.ones((3,1)))
        p=np.zeros((1,5,5,1));p[0,1,2,0]=1
        y=np.stack([op.forward(p[0])]);row,recoded=measurement_fit(op,p,y,2,1)
        self.assertAlmostEqual(row['center_rgb_mse'],0);self.assertGreater(recoded[0,2,2,0],0)
        cropped=op.forward(p[0,2:3,2:3])
        self.assertAlmostEqual(float(cropped.sum()),0)

    def test_signed_prediction_and_zero_signal(self):
        op=RGBForward(np.ones((1,1,1)),np.ones((3,1)))
        p=-np.ones((1,3,3,1));y=np.zeros((1,3,3,3))
        row,recoded=measurement_fit(op,p,y,1,1)
        self.assertIsNone(row['center_relative_l2']);self.assertEqual(row['zero_measurement_patches'],1)
        self.assertAlmostEqual(row['center_rgb_mse'],1.);self.assertTrue((recoded<0).all())

    def test_invalid_region_and_shapes(self):
        op=RGBForward(np.ones((1,1,1)),np.ones((3,1)));p=np.ones((1,3,3,1));y=np.ones((1,3,3,3))
        for h,s in [(3,1),(-1,1),(0,0)]:
            with self.assertRaises(ValueError):measurement_fit(op,p,y,h,s)
        with self.assertRaises(ValueError):measurement_fit(op,p,y[:,:,:,:2],0,1)


if __name__=='__main__':unittest.main()
