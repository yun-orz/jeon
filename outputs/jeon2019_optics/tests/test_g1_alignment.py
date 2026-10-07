# -*- coding: utf-8 -*-
"""对齐分析的独立测角、尺寸分母及固定高度性质检查。"""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from optics.g1_alignment import clockwise_profile,angular_registration,size_diagnostics,polar_samples
from optics.g1_forward import profile_for
from optics.doe import compute_doe_transmission_field


class TestAlignment(unittest.TestCase):
    def test_registration_angle_sign(self):
        theta=np.arange(720)*2*np.pi/720;r=np.linspace(20e-6,100e-6,25)
        def pattern(degree):
            # 解析角向图样只校验相关符号，不用来替代DOE的衍射PSF。
            t=theta-np.radians(degree)
            return np.exp(-r[:,None]/80e-6)*(2+np.cos(3*t)+.3*np.cos(6*t))
        for degree in [12.5,-8.]:
            row=angular_registration(pattern(0),pattern(degree),r)
            self.assertTrue(row['reliable']);self.assertAlmostEqual(row['angle_deg'],degree)

    def test_isotropic_pattern_no_direction(self):
        row=angular_registration(np.ones((20,720)),np.ones((20,720)),np.arange(1,21)*1e-6)
        self.assertFalse(row['reliable']);self.assertIsNone(row['angle_deg'])

    def test_window_and_input_denominators(self):
        x=np.arange(-50,51)*1e-6;X,Y=np.meshgrid(x,x);image=np.exp(-(X**2+Y**2)/(2*(10e-6)**2))
        bank=np.stack([image/image.sum()*.7,image/image.sum()*.4])
        rows=size_diagnostics(bank,x,x,1e-6,dict(shape_roi_m=40e-6,peak_thresholds=[.5,.1]))
        self.assertGreater(rows[0]['R50_input_um'],rows[0]['R50_window_um'])
        self.assertAlmostEqual(rows[0]['R50_window_um'],10*np.sqrt(2*np.log(2)),delta=1)
        self.assertIsNone(rows[1]['R50_input_um'])
        self.assertAlmostEqual(rows[0]['R50_window_um'],rows[1]['R50_window_um'])

    def test_clockwise_is_fixed_opposite_chirality(self):
        c=json.loads((ROOT/'config_g1.json').read_text(encoding='utf-8'))
        c['input']['n']=201;c['input']['spacing_m']=5.5e-6
        old=profile_for(c);cw=clockwise_profile(old)
        np.testing.assert_allclose(cw.delta_h,old.delta_h[::-1],rtol=1e-12,atol=1e-18)
        fp=cw.compute_fingerprint()
        a=compute_doe_transmission_field(cw,420e-9);b=compute_doe_transmission_field(cw,660e-9)
        self.assertFalse(np.allclose(a,b));self.assertEqual(fp,cw.compute_fingerprint())

    def test_invalid_polar_and_negative_intensity(self):
        x=np.linspace(-100e-6,100e-6,21);bank=np.ones((1,21,21))
        with self.assertRaises(ValueError):
            polar_samples(bank,x,x,dict(r_min_m=10e-6,r_max_m=101e-6,r_samples=10,theta_samples=720))
        bank[0,0,0]=-1
        with self.assertRaises(ValueError):size_diagnostics(bank,x,x,10e-6,dict(shape_roi_m=50e-6,peak_thresholds=[.5]))


if __name__=='__main__':unittest.main()
