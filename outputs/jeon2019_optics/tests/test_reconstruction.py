# -*- coding: utf-8 -*-
"""解析解及SciPy增广非负最小二乘独立参考；无清理操作。"""
import copy
import unittest
import numpy as np
from scipy.optimize import lsq_linear
from optics.imaging import SpectralImager
from optics.reconstruction import solve_nonnegative_ridge,evaluate_cube

CONFIG=dict(alpha_relative=.02,max_iterations=10000,gradient_tolerance=1e-9,
            objective_tolerance=1e-12,power_iterations=100,power_tolerance=1e-10,
            seed=3,initial_L_factor=1.05,backtracking_factor=2.,max_backtracks=50)


class Diagonal:
    cube_shape=(1,2,2); output_shape=(2,2)
    def __init__(self,d): self.d=np.asarray(d)
    def forward(self,x): return self.d*x[0]
    def adjoint(self,y): return (self.d*y)[None]


class ReconstructionTests(unittest.TestCase):
    def test_diagonal_analytic_solution(self):
        op=Diagonal([[1.,2.],[.5,3.]])
        y=np.array([[1.,-2.],[3.,4.]])
        result=solve_nonnegative_ridge(op,y,CONFIG)
        expected=np.maximum(0.,op.d*y/(op.d**2+result['alpha']))
        self.assertEqual(result['status'],'converged')
        np.testing.assert_allclose(result['reconstruction'][0],expected,rtol=1e-7,atol=1e-8)

    def test_explicit_augmented_nnls_reference(self):
        op=SpectralImager(np.array([[[.01,.04,.02],[.06,.2,.03],[.02,.05,.01]],
                                    [[.06,.01,.03],[.02,.12,.08],[.01,.03,.05]]]),[2,2],1.,[1,.8])
        n=np.prod(op.cube_shape)
        A=np.column_stack([op.forward(np.eye(n)[i].reshape(op.cube_shape)).ravel() for i in range(n)])
        y=np.random.default_rng(10).uniform(0,.2,op.output_shape)
        r=solve_nonnegative_ridge(op,y,CONFIG);alpha=r['alpha']
        ref=lsq_linear(np.vstack([A,np.sqrt(alpha)*np.eye(n)]),np.r_[y.ravel(),np.zeros(n)],
                       bounds=(0,np.inf),tol=1e-13,max_iter=500)
        self.assertTrue(ref.success);self.assertEqual(r['status'],'converged')
        f_ref=.5*np.sum((A@ref.x-y.ravel())**2)+.5*alpha*np.sum(ref.x**2)
        self.assertLess(abs(r['history'][-1]['objective']-f_ref)/f_ref,1e-5)
        np.testing.assert_allclose(r['reconstruction'].ravel(),ref.x,rtol=1e-6,atol=1e-6)
        self.assertLess(r['history'][-1]['projected_gradient_relative'],1e-5)

    def test_backtracking_and_monotone_objective(self):
        c=dict(CONFIG,initial_L_factor=1e-5)
        r=solve_nonnegative_ridge(Diagonal([[1.,2.],[.5,3.]]),np.ones((2,2)),c)
        self.assertGreater(r['history'][0]['backtracks'],0)
        self.assertTrue(np.all(np.diff([v['objective'] for v in r['history']])<=1e-14))

    def test_iteration_limit_is_not_converged(self):
        r=solve_nonnegative_ridge(Diagonal([[1.,2.],[.5,3.]]),np.ones((2,2)),dict(CONFIG,max_iterations=1))
        self.assertEqual(r['status'],'iteration_limit')

    def test_backtrack_exhaustion_raises(self):
        with self.assertRaisesRegex(RuntimeError,'回溯'):
            solve_nonnegative_ridge(Diagonal(np.ones((2,2))),np.ones((2,2)),dict(CONFIG,initial_L_factor=1e-8,max_backtracks=1))

    def test_zero_measurement_and_zero_operator(self):
        for op,y in [(Diagonal(np.ones((2,2))),np.zeros((2,2))),
                     (Diagonal(np.zeros((2,2))),np.ones((2,2)))]:
            r=solve_nonnegative_ridge(op,y,CONFIG)
            self.assertEqual(r['status'],'converged');self.assertTrue(np.all(r['reconstruction']==0))

    def test_reject_bad_measurement_or_config(self):
        op=Diagonal(np.ones((2,2)))
        for y in [np.zeros((3,3)),np.full((2,2),np.nan),np.ones((2,2),dtype=complex)]:
            with self.assertRaises(ValueError): solve_nonnegative_ridge(op,y,CONFIG)
        for key,value in [('alpha_relative',-1),('backtracking_factor',1),('max_iterations',0)]:
            c=copy.deepcopy(CONFIG);c[key]=value
            with self.assertRaises(ValueError):solve_nonnegative_ridge(op,np.ones((2,2)),c)

    def test_evaluation_does_not_change_reconstruction_and_sam_undefined(self):
        r=solve_nonnegative_ridge(Diagonal(np.ones((2,2))),np.ones((2,2)),CONFIG)
        saved=r['reconstruction'].copy()
        evaluate_cube(np.ones((1,2,2)),saved,1e-12)
        evaluate_cube(np.ones((1,2,2))*2,saved,1e-12)
        np.testing.assert_array_equal(saved,r['reconstruction'])
        e=evaluate_cube(np.ones((3,2,2)),np.zeros((3,2,2)),1e-12)
        self.assertIsNone(e['sam_mean_deg']);self.assertEqual(e['sam_undefined_reconstruction_zero_count'],4)


if __name__=='__main__':unittest.main()
