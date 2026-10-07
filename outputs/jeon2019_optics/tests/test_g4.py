# -*- coding: utf-8 -*-
"""G4真实数据预处理的谱单位、掩码与分割检查。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g4_data import photon_relative,valid_tile_origins


class TestG4(unittest.TestCase):
    def test_sensitivity_and_photon_conversion(self):
        wave=np.array([420,540,660]);s=np.array([.2,.5,.8]);energy=np.array([2.,3.,4.])
        raw=energy*s
        np.testing.assert_allclose(photon_relative(raw,s,wave),energy*wave/540.)

    def test_motion_mask_excludes_entire_context(self):
        mask=np.ones((8,8),dtype=bool);mask[1,1]=False
        self.assertEqual(valid_tile_origins(mask,4),[(0,4),(4,0),(4,4)])

    def test_no_overlapping_tiles(self):
        rows=valid_tile_origins(np.ones((13,15)),4)
        seen=np.zeros((13,15),dtype=int)
        for y,x in rows:seen[y:y+4,x:x+4]+=1
        self.assertLessEqual(seen.max(),1)

    def test_invalid_calibration_rejected(self):
        for sensitivity in [[1,0],[1,np.nan]]:
            with self.assertRaises(ValueError):photon_relative([1,2],sensitivity,[420,430])
        with self.assertRaises(ValueError):photon_relative([-1,2],[1,1],[420,430])


if __name__=='__main__':unittest.main()
