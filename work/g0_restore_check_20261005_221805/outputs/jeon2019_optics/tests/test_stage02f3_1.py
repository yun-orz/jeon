# -*- coding: utf-8 -*-
"""冻结参数、交叉项和优化误差区间的独立小阵列检查。"""
import json,copy,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from optics.electron_reconstruction import make_electron_operator,measured_weights
from optics.fixed_target_control import check_frozen_parameters,solve_fixed_target,decompose_errors
from optics.reconstruction import estimate_data_lipschitz

class FrozenControlTests(unittest.TestCase):
    def setUp(self):
        k=np.zeros((3,3,3));k[:,1,1]=[.5,.6,.7]
        self.base=make_electron_operator(k,[3,3],1e-6,[1,1,1],[420e-9,540e-9,660e-9],[.5,.5,.5],540e-9,100,[0,5,0,5])
        self.c=json.loads((Path(__file__).resolve().parents[1]/'config_stage02f2.json').read_text(encoding='utf-8'))['reconstruction'];self.w=np.ones((5,5))
        L=estimate_data_lipschitz(self.base,self.c['power'])['value'];self.ref=dict(weights=self.w,alpha=.001*L,L_data_estimate=L)
    def solve(self,z,solver=None):
        return solve_fixed_target(self.base,z,0.,self.w,self.ref['alpha'],self.ref['L_data_estimate'],1.05,solver or self.c['solver'],self.ref)
    def test_frozen_weight_recomputed_rejected(self):
        w,_=measured_weights(np.ones((5,5)),2.,4.)
        with self.assertRaises(ValueError):check_frozen_parameters(w,self.ref['alpha'],self.ref['L_data_estimate'],self.ref)
    def test_changed_alpha_or_L_rejected(self):
        with self.assertRaises(ValueError):check_frozen_parameters(self.w,self.ref['alpha']*2,self.ref['L_data_estimate'],self.ref)
        with self.assertRaises(ValueError):check_frozen_parameters(self.w,self.ref['alpha'],self.ref['L_data_estimate']*2,self.ref)
    def test_no_parameter_estimation_in_control(self):
        with patch('optics.electron_reconstruction.measured_weights',side_effect=AssertionError('不能重算W')),patch('optics.reconstruction.estimate_data_lipschitz',side_effect=AssertionError('不能重算L')):
            fit=self.solve(np.ones((5,5)))
        self.assertEqual(fit['result']['alpha'],self.ref['alpha']);np.testing.assert_array_equal(fit['operator'].weights,self.w)
    def test_negative_ideal_target_retained(self):
        fit=self.solve(np.full((5,5),-1e-13))
        self.assertTrue(np.all(fit['target']<0));self.assertTrue(np.all(fit['result']['reconstruction']==0))
    def test_truth_isolation(self):
        z=np.arange(25).reshape(5,5)
        def trial(truth):
            fit=self.solve(z);evaluation=float(np.sum(abs(fit['result']['reconstruction']-truth)));return fit,evaluation
        a,ea=trial(np.zeros((3,3,3)));b,eb=trial(np.ones((3,3,3)))
        np.testing.assert_array_equal(a['result']['reconstruction'],b['result']['reconstruction']);self.assertNotEqual(ea,eb)
    def test_cross_term_can_be_negative(self):
        t=np.ones((1,2,2));xn=np.full(t.shape,.75);x0=np.full(t.shape,.5)
        arrays,d=decompose_errors(t,xn,x0,.1,.2)
        np.testing.assert_array_equal(arrays['error_total'],arrays['error_baseline']+arrays['error_noise_perturbation']);self.assertLess(d['cross_term'],0)
        self.assertAlmostEqual(d['norms_absolute']['total']**2,d['norms_absolute']['baseline']**2+d['norms_absolute']['noise_perturbation']**2+d['cross_term'])
        self.assertAlmostEqual(d['baseline_interval']['lower'],.8);self.assertAlmostEqual(d['noise_perturbation_interval']['lower'],.2)
    def test_intervals_clip_lower_at_zero(self):
        t=np.ones((1,1,1));_,d=decompose_errors(t,t,t,1.,2.)
        self.assertEqual(d['baseline_interval']['lower'],0);self.assertEqual(d['noise_perturbation_interval']['upper'],3.)
    def test_nonzero_background_and_limit(self):
        c=copy.deepcopy(self.c['solver']);c['max_iterations']=1
        fit=solve_fixed_target(self.base,np.arange(25).reshape(5,5),3.,self.w,self.ref['alpha'],self.ref['L_data_estimate'],1.05,c,self.ref)
        np.testing.assert_array_equal(fit['target'],np.arange(25).reshape(5,5)-3);self.assertEqual(fit['result']['status'],'iteration_limit')
    def test_invalid_decomposition(self):
        t=np.ones((1,2,2))
        for tn,xn,x0,dn,d0 in [(t,t,t,-1,0),(t,t,t,0,np.nan),(t,np.ones((2,2,2)),t,0,0),(t,-t,t,0,0),(t*0,t,t,0,0)]:
            with self.assertRaises(ValueError):decompose_errors(tn,xn,x0,dn,d0)

if __name__=='__main__':unittest.main()
