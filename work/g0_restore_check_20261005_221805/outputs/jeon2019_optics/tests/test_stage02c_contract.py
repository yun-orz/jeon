# -*- coding: utf-8 -*-
"""02C配置防错及低方向可靠性情况下的验收语义。"""
import copy
import json
from pathlib import Path
import unittest
from main_stage02c import validate_config, convergence_pass


class Stage02CContract(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]/'config_stage02c.json').read_text(encoding='utf-8'))

    def test_valid_default(self):
        validate_config(self.cfg)

    def test_reject_unimplemented_camera(self):
        for key, value in [('response','photons'), ('fill_factor',.9), ('pixels_per_axis',7)]:
            c = copy.deepcopy(self.cfg); c['detector'][key] = value
            with self.assertRaises(ValueError): validate_config(c)

    def test_reject_invalid_physics(self):
        for section, key, value in [('optical','distance_m',0), ('input','n',501),
                                    ('fabrication','levels',65536), ('optical','wavelengths_m',[540e-9,420e-9])]:
            c = copy.deepcopy(self.cfg); c[section][key] = value
            with self.assertRaises(ValueError): validate_config(c)

    def test_unreliable_angles_are_not_invented(self):
        row = dict(pixel_l1=0., pixel_l2=0., eta_absolute=0.)
        for b in ['primary','sensitivity']:
            row['angle_'+b+'_deg'] = None; row['reliability_'+b+'_match'] = True
        for r in ['R50','R80']:
            row[r+'_status_match'] = True; row[r+'_difference_um'] = None
        self.assertTrue(convergence_pass(row, self.cfg['thresholds'], self.cfg['detector']['pitch_m']))
        row['reliability_primary_match'] = False
        self.assertFalse(convergence_pass(row, self.cfg['thresholds'], self.cfg['detector']['pitch_m']))


if __name__ == '__main__': unittest.main()
