# -*- coding: utf-8 -*-
"""只读场景源重验、错误配置及真值来源完整性。"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from optics.stage02d1_source import load_imaging_source
from optics.stage02c_source import tree_sha
from main_stage02d2 import validate_config

ROOT=Path(__file__).resolve().parents[1]


class SourceReconstructionTests(unittest.TestCase):
    def test_source_identity_and_readonly(self):
        source=ROOT/'results/stage02d1/run_20261002_203904'
        before=tree_sha(source)
        items,evidence=load_imaging_source(source,ROOT)
        self.assertEqual(len(items),6);self.assertTrue(evidence['source_unchanged'])
        self.assertEqual(before,tree_sha(source))

    def test_reject_missing_or_modified_model(self):
        with self.assertRaises(FileNotFoundError):load_imaging_source(ROOT/'不存在的成像run',ROOT)
        from optics.imaging import SpectralImager
        original=SpectralImager.per_band_full
        # 直接修改前向数值，但保持源文件不动；必须拒绝与保存测量不符的模型。
        with patch.object(SpectralImager,'per_band_full',new=lambda op,x:2*original(op,x)):
            with self.assertRaisesRegex(ValueError,'前向模型'):load_imaging_source(ROOT/'results/stage02d1/run_20261002_203904',ROOT)

    def test_config_validation(self):
        c=json.loads((ROOT/'config_stage02d2.json').read_text(encoding='utf-8'));validate_config(c)
        c['evaluation']['spectrum_norm_threshold']=0
        with self.assertRaises(ValueError):validate_config(c)


if __name__=='__main__':unittest.main()
