# -*- coding: utf-8 -*-
"""采样伴随、信息隔离、选择准则与实际alpha转移的独立检查。"""
import copy,json,unittest
from pathlib import Path
import numpy as np
from optics.imaging import SpectralImager,operator_checks
from optics.detector_holdout import DetectorSubsetOperator,split_detector,prepare_problem,fit_candidate,choose_candidate,refit_selected
from optics.reconstruction import estimate_data_lipschitz

class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.base=SpectralImager(np.ones((1,1,1)),[4,4],1.,[3.])
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'config_stage02f3_2.json').read_text(encoding='utf-8'))['reconstruction']
        self.split=dict(seed=2019,identity='test_detector',validation_fraction=.25)
        self.z=np.arange(16).reshape(4,4)-2.
        self.problem=prepare_problem(self.base,self.z,1.,self.split)
    def test_split_partition_and_repeat(self):
        t,v=split_detector([4,4],2019,'test_detector',.25);t2,v2=split_detector([4,4],2019,'test_detector',.25)
        np.testing.assert_array_equal(t,t2);np.testing.assert_array_equal(v,v2);self.assertEqual(len(v),4)
        self.assertFalse(set(t)&set(v));self.assertEqual(set(t)|set(v),set(range(16)))
    def test_explicit_scatter_adjoint(self):
        op=DetectorSubsetOperator(self.base,[1,5,9]);y=np.array([[2.,-3.,4.]])
        full=np.zeros((4,4));full.ravel()[[1,5,9]]=y.ravel()
        np.testing.assert_array_equal(op.adjoint(y),self.base.adjoint(full))
        check=operator_checks(op,2019,np.ones((1,3)));self.assertLess(check['inner_product_relative_error'],1e-10);self.assertLess(check['gradient_relative_error'],1e-5)
    def test_invalid_indices_and_fraction(self):
        for idx in [[1,1],[-1],[16],[1.2],[]]:
            with self.assertRaises(ValueError):DetectorSubsetOperator(self.base,idx)
        for fraction in [0,1,.001,np.nan]:
            with self.assertRaises(ValueError):split_detector([4,4],2019,'A',fraction)
    def test_negative_observation_and_background(self):
        idx=self.problem['train'].indices
        np.testing.assert_array_equal(self.problem['target_train'],(self.z-1).ravel()[idx].reshape(1,-1));self.assertTrue(np.any(self.problem['target_train']<0))
    def test_validation_changes_do_not_change_training(self):
        p=self.problem;z2=self.z.copy();z2.ravel()[p['validation'].indices]+=50
        q=prepare_problem(self.base,z2,1.,self.split)
        est=estimate_data_lipschitz(p['train'],self.cfg['power']);est2=estimate_data_lipschitz(q['train'],self.cfg['power']);self.assertEqual(est,est2)
        for factor in [.01,.001,.0001,.00001]:
            a=fit_candidate(p,factor,self.cfg,est);b=fit_candidate(q,factor,self.cfg,est2)
            np.testing.assert_array_equal(a['result']['reconstruction'],b['result']['reconstruction']);self.assertEqual(a['alpha'],b['alpha']);self.assertNotEqual(a['validation_mse_e2'],b['validation_mse_e2'])
    def test_score_independent_and_strict_tie(self):
        est=estimate_data_lipschitz(self.problem['train'],self.cfg['power']);fit=fit_candidate(self.problem,.001,self.cfg,est)
        idx=self.problem['validation'].indices;x=fit['result']['reconstruction']
        expected=float(np.mean((3*x[0].ravel()[idx]+1-self.z.ravel()[idx])**2));self.assertAlmostEqual(expected,fit['validation_mse_e2'])
        records=[dict(alpha_factor=f,solver_status='converged',certificate_passed=True,validation_mse_e2=4.) for f in [.01,.001]]
        self.assertEqual(choose_candidate(records,[.01,.001]),0)
    def test_incomplete_candidates_never_select(self):
        records=[dict(alpha_factor=.01,solver_status='iteration_limit',certificate_passed=True,validation_mse_e2=1.)]
        with self.assertRaises(ValueError):choose_candidate(records,[.01])
        records[0]['solver_status']='converged';records[0]['certificate_passed']=False
        with self.assertRaises(ValueError):choose_candidate(records,[.01])
        with self.assertRaises(ValueError):choose_candidate([], [.01])
    def test_truth_and_ideal_context_isolation(self):
        def pipeline(truth,ideal):
            p=prepare_problem(self.base,self.z,1.,self.split);est=estimate_data_lipschitz(p['train'],self.cfg['power']);factors=[.01,.001]
            fits=[fit_candidate(p,f,self.cfg,est) for f in factors]
            recs=[dict(alpha_factor=f,solver_status=a['result']['status'],certificate_passed=a['certificate_passed'],validation_mse_e2=a['validation_mse_e2']) for f,a in zip(factors,fits)]
            index=choose_candidate(recs,factors);result=refit_selected(p,factors[index],self.cfg)
            return index,result['result']['reconstruction'],float(np.linalg.norm(result['result']['reconstruction']-truth))
        a,xa,ea=pipeline(np.zeros((1,4,4)),np.zeros((4,4)));b,xb,eb=pipeline(np.ones((1,4,4)),np.ones((4,4))*100)
        self.assertEqual(a,b);np.testing.assert_array_equal(xa,xb);self.assertNotEqual(ea,eb)
    def test_full_alpha_uses_full_operator(self):
        cfg=copy.deepcopy(self.cfg);fit=refit_selected(self.problem,.0001,cfg)
        self.assertEqual(fit['alpha'],.0001*fit['spectral_estimate']['value']);self.assertEqual(fit['result']['initialization'],'zeros')
        np.testing.assert_allclose(fit['result']['reconstruction'][0],np.maximum(self.z-1,0)*3/(9+fit['alpha']),rtol=1e-8,atol=1e-9)

if __name__=='__main__':unittest.main()
