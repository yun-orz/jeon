# -*- coding: utf-8 -*-
"""测试场景不拟合尺度、掩码及非重叠选择检查。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g5_test_data import select_test


class TestHeldoutData(unittest.TestCase):
    def test_fixed_train_scale_and_determinism(self):
        raw=np.ones((8,8,31));lbl=np.ones((8,8));s=np.ones(31)
        a,origins,n=select_test(raw,lbl,s,2.,4,2,19)
        b,again,_=select_test(raw,lbl,s,2.,4,2,19)
        np.testing.assert_array_equal(a,b);self.assertEqual(origins,again);self.assertEqual(n,4)
        np.testing.assert_allclose(a[0,0,0],np.arange(420,661,10)/1080.,rtol=1e-7)
        # 测试亮度倍增后输出也倍增，证明没有在测试数据上重新拟合峰值。
        doubled,_,_=select_test(2*raw,lbl,s,2.,4,2,19)
        np.testing.assert_array_equal(doubled,2*a)

    def test_mask_nonoverlap(self):
        raw=np.ones((8,8,31));lbl=np.ones((8,8));lbl[0,0]=0
        _,origins,n=select_test(raw,lbl,np.ones(31),1.,4,3,0)
        self.assertEqual(n,3);self.assertNotIn((0,0),origins)
        occupied=np.zeros((8,8))
        for y,x in origins:occupied[y:y+4,x:x+4]+=1
        self.assertLessEqual(occupied.max(),1)

    def test_invalid_and_insufficient_data(self):
        raw=np.ones((8,8,31));lbl=np.ones((8,8))
        for scale in [0.,np.nan,True]:
            with self.assertRaises(ValueError):select_test(raw,lbl,np.ones(31),scale,4,1,0)
        with self.assertRaises(ValueError):select_test(raw,lbl*0,np.ones(31),1.,4,1,0)
        raw[0,0,0]=np.nan
        with self.assertRaises(ValueError):select_test(raw,lbl,np.ones(31),1.,4,4,0)


if __name__=='__main__':unittest.main()
