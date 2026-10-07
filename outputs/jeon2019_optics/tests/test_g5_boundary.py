# -*- coding: utf-8 -*-
"""固定物理中心、掩码排除和有限核支持的独立检查。"""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.signal import convolve2d
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g5_boundary import center_crop,support_tiles,change_metrics


class TestBoundary(unittest.TestCase):
    def test_same_physical_center(self):
        field=np.arange(32*32).reshape(32,32,1)
        np.testing.assert_array_equal(center_crop(center_crop(field,24),8),field[12:20,12:20])
        with self.assertRaises(ValueError): center_crop(field,9)
        with self.assertRaises(ValueError): center_crop(field,40)

    def test_support_mask_exclusion_without_reselection(self):
        raw=np.ones((40,40,31));mask=np.ones((40,40));mask[0,0]=0
        tiles,records=support_tiles(raw,mask,[[4,4],[20,20]],8,16,np.ones(31),2.)
        self.assertEqual([r['accepted'] for r in records],[False,True])
        self.assertEqual(tiles.shape,(1,16,16,25))
        self.assertEqual(records[0]['invalid_pixels'],1)
        self.assertAlmostEqual(float(tiles[0,0,0,12]),.5)

    def test_shared_measurement_has_exact_finite_support(self):
        rng=np.random.default_rng(31); scene=rng.random((40,40));kernel=rng.random((5,5))
        large=convolve2d(scene,kernel,mode='same')
        # 20像素观测域加两侧2像素物体支持，已覆盖5像素核。
        support=scene[8:32,8:32]
        local=convolve2d(support,kernel,mode='same')[2:-2,2:-2]
        np.testing.assert_allclose(local,large[10:30,10:30],atol=1e-14,rtol=0)
        truncated=convolve2d(scene[10:30,10:30],kernel,mode='same')
        self.assertGreater(np.max(np.abs(truncated-local)),.1)

    def test_relative_change_signed_and_undefined(self):
        result=change_metrics(np.array([-2.,2.]),np.array([-1.,1.]))
        self.assertEqual(result['relative_l2'],1.)
        self.assertEqual(result['rmse'],1.)
        self.assertIsNone(change_metrics(np.ones(2),np.zeros(2))['relative_l2'])


if __name__=='__main__':unittest.main()
