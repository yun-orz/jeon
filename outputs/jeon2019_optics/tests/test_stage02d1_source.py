# -*- coding: utf-8 -*-
"""源PSF物理重验及配置拒绝，不修改源run或删除测试证据。"""
import copy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from main_stage02d1 import validate_config
from optics.stage02c_source import load_source,tree_sha
from optics.doe import compute_doe_transmission_field

ROOT = Path(__file__).resolve().parents[1]


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'config_stage02d1.json').read_text(encoding='utf-8'))
        self.source = ROOT/self.cfg['source_run']

    def test_real_source_verifies_and_preserves_eta(self):
        before = tree_sha(self.source)
        kernels,evidence = load_source(self.source,ROOT,self.cfg)
        self.assertEqual(len(evidence['validated_fields']),9)
        expected = json.loads((self.source/'metrics/validation.json').read_text(encoding='utf-8'))
        for device,k in kernels.items():
            self.assertEqual(k.shape,(3,97,97))
            for b,lam in enumerate(self.cfg['wavelengths_m']):
                rec = expected['records'][device+'_'+str(round(lam*1e9))+'nm_q8']
                self.assertAlmostEqual(float(k[b].sum()),rec['eta_window'],places=13)
        self.assertEqual(before,tree_sha(self.source))

    def test_reject_wrong_input_power_even_with_saved_metadata(self):
        with patch('optics.stage02c_source.compute_doe_transmission_field',
                   side_effect=lambda p,lam: 2*compute_doe_transmission_field(p,lam)):
            with self.assertRaisesRegex(ValueError,'输入功率'): load_source(self.source,ROOT,self.cfg)

    def test_reject_center_point_and_wrong_pitch(self):
        for key,value in [('quadrature_per_axis',1),('pitch_m',6e-6)]:
            cfg = copy.deepcopy(self.cfg); cfg[key] = value
            with self.assertRaises(ValueError): load_source(self.source,ROOT,cfg)

    def test_reject_missing_source(self):
        with self.assertRaises(FileNotFoundError): load_source(ROOT/'不存在的02C源run',ROOT,self.cfg)

    def test_config_negative_paths(self):
        validate_config(self.cfg)
        for key,value in [('devices',['continuous','continuous']),('scene_shape',[8,32]),
                          ('response',[1,2,1]),('crop',[0,129,0,128]),('seed',-1),
                          ('wavelengths_m',[660e-9,540e-9,420e-9])]:
            cfg = copy.deepcopy(self.cfg); cfg[key] = value
            with self.assertRaises(ValueError): validate_config(cfg)


if __name__=='__main__': unittest.main()
