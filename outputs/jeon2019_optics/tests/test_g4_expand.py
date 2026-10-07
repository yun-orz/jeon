# -*- coding: utf-8 -*-
"""扩充数据选择的可比较性与确定性检查。"""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g4_expand_data import extend_origins


class TestExpand(unittest.TestCase):
    def test_originals_preserved_and_unique(self):
        grid=[(i,j) for i in range(4) for j in range(4)];old=[[0,0],[2,3]]
        chosen=extend_origins(grid,old,10,7)
        self.assertEqual(chosen[:2],[(0,0),(2,3)]);self.assertEqual(len(set(chosen)),10)
        self.assertEqual(chosen,extend_origins(grid,old,10,7))

    def test_same_count_and_all_tiles(self):
        grid=[(0,0),(0,4),(4,0)];old=[(0,4)]
        self.assertEqual(extend_origins(grid,old,1,0),old)
        self.assertEqual(set(extend_origins(grid,old,3,0)),set(grid))

    def test_bad_old_or_count_rejected(self):
        grid=[(0,0),(0,4)]
        for old,count in [([(0,0),(0,0)],2),([(8,0)],2),([(0,0)],3),([(0,0)],0)]:
            with self.assertRaises(ValueError):extend_origins(grid,old,count,0)


if __name__=='__main__':unittest.main()
