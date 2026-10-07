# -*- coding: utf-8 -*-
"""G1的配置、物理尺度及数值语义检查；不删除或清空任何证据。"""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.special import j1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optics.g1_forward import validate_config, profile_for, native_parseval, rotation_summary
from optics.coordinates import make_grid
from optics.propagation import fresnel_kernel_separable
from optics.psf_analysis import c3_rotation_metric


class TestG1(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT/'config_g1.json').read_text(encoding='utf-8'))

    def test_invalid_configs(self):
        for group, key, value in [('input','n',True),('input','spacing_m',float('nan')),
                                 ('detector','quadrature_main',4),('runtime','save_results',1),
                                 ('optical','wings_N',1),('plots','dpi',0)]:
            with self.subTest(key=key):
                c=copy.deepcopy(self.c);c[group][key]=value
                with self.assertRaises(ValueError): validate_config(c)
        self.c['wavelengths_nm']=self.c['wavelengths_nm'][:-1]
        with self.assertRaises(ValueError): validate_config(self.c)

    def test_refined_grid_same_device(self):
        a,b=profile_for(self.c),profile_for(self.c,True)
        np.testing.assert_allclose(a.delta_h,b.delta_h[::2,::2],rtol=1e-10,atol=1e-18)
        np.testing.assert_array_equal(a.mask,b.mask[::2,::2])
        self.assertAlmostEqual(a.grid.x.half_width,b.grid.x.half_width)

    def test_focused_pupil_airy_scale(self):
        # 理想薄透镜相位只用于独立传播验证，不加入Jeon DOE模型。
        g=make_grid(401,.5e-6);diameter=100e-6;lam=550e-9;z=.05
        pupil=(g.radius()<=diameter/2).astype(float)
        u1=pupil*np.exp(-1j*np.pi*g.radius()**2/(lam*z))
        x=np.arange(0,801,4)*1e-6
        u2=fresnel_kernel_separable(u1,g,lam,z,x,np.array([0.]))[0]
        t=np.pi*diameter*x/(lam*z)
        airy=np.ones_like(t);airy[1:]=(2*j1(t[1:])/t[1:])**2
        computed=np.abs(u2)**2/abs(u2[0])**2
        self.assertLess(np.linalg.norm(computed-airy)/np.linalg.norm(airy),.005)
        first=int(np.argmin(computed[60:100]))+60
        self.assertLess(abs(x[first]-1.21966989*lam*z/diameter),5e-6)
        self.assertTrue(native_parseval(u1,g,lam,z)['passed'])

    def test_angle_coordinate_sign_and_missing_segments(self):
        # 构造解析三点斑，仅测试测角符号；这不是替代衍射的器件PSF。
        g=make_grid(201,1e-6);X,Y=g.meshgrid()
        def spots(angle):
            return sum(np.exp(-((X-60e-6*np.cos(angle+k*2*np.pi/3))**2
                                 +(Y-60e-6*np.sin(angle+k*2*np.pi/3))**2)/(2*(3e-6)**2)) for k in range(3))
        rows=[]
        for angle in [0,.2,.4]:
            m=c3_rotation_metric(spots(angle),g,20e-6,90e-6)
            rows.append({'rotation_bands': {'primary':m,'sensitivity':m}})
        summary=rotation_summary(rows,[420e-9,430e-9,440e-9])
        self.assertGreater(summary['primary']['alpha_unwrapped_deg'][-1],summary['primary']['alpha_unwrapped_deg'][0])
        rows[1]['rotation_bands']['primary']=c3_rotation_metric(np.exp(-(X**2+Y**2)/(2*(40e-6)**2)),g,20e-6,90e-6)
        summary=rotation_summary(rows,[420e-9,430e-9,440e-9])
        self.assertIsNone(summary['primary']['alpha_unwrapped_deg'][1])
        self.assertEqual(len(summary['primary']['segments']),2)


if __name__ == '__main__':
    unittest.main()
