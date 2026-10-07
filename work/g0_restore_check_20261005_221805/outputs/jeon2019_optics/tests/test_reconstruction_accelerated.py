# -*- coding: utf-8 -*-
"""固定α加速法的解析/独立参考、单调重启及停止语义检查。"""
import unittest
import numpy as np
from scipy.optimize import lsq_linear
from optics.imaging import SpectralImager
from optics.reconstruction_accelerated import solve_accelerated_ridge
from tests.test_reconstruction import Diagonal

C=dict(max_iterations=10000,gradient_tolerance=1e-9,objective_tolerance=1e-12,
       backtracking_factor=2.,max_backtracks=50)


class AcceleratedTests(unittest.TestCase):
    def test_diagonal_and_restart(self):
        op=Diagonal([[1.,2.],[.5,3.]]);y=np.array([[1.,-2.],[3.,4.]]);alpha=.02
        r=solve_accelerated_ridge(op,y,alpha,C,9.02)
        self.assertEqual(r['status'],'converged');self.assertGreater(r['restarts'],0)
        np.testing.assert_allclose(r['reconstruction'][0],np.maximum(0,op.d*y/(op.d**2+alpha)),rtol=1e-7,atol=1e-8)
        self.assertTrue(all(b['objective']<=a['objective']+1e-14 for a,b in zip(r['history'],r['history'][1:])))
        self.assertTrue(all(v['momentum']==0 for v in r['history'] if v['restarted']))

    def test_full_crop_explicit_nnls(self):
        k=np.array([[[.01,.04,.02],[.06,.2,.03],[.02,.05,.01]],
                    [[.06,.01,.03],[.02,.12,.08],[.01,.03,.05]]])
        for crop in [None,[0,3,1,4]]:
            op=SpectralImager(k,[2,2],1.,[1,.8],crop);n=8
            A=np.column_stack([op.forward(np.eye(n)[i].reshape(op.cube_shape)).ravel() for i in range(n)])
            y=np.random.default_rng(10).uniform(0,.2,op.output_shape);alpha=.01
            r=solve_accelerated_ridge(op,y,alpha,C,.001)
            ref=lsq_linear(np.vstack([A,np.sqrt(alpha)*np.eye(n)]),np.r_[y.ravel(),np.zeros(n)],bounds=(0,np.inf),tol=1e-13,max_iter=500)
            self.assertTrue(ref.success);self.assertEqual(r['status'],'converged')
            np.testing.assert_allclose(r['reconstruction'].ravel(),ref.x,rtol=1e-6,atol=1e-6)
            f=.5*np.sum((A@ref.x-y.ravel())**2)+.5*alpha*np.sum(ref.x**2)
            self.assertLess(abs(r['history'][-1]['objective']-f)/f,1e-5)
            self.assertGreater(r['history'][0]['backtracks'],0)

    def test_zero_data_and_zero_operator(self):
        for op,y in [(Diagonal(np.ones((2,2))),np.zeros((2,2))),(Diagonal(np.zeros((2,2))),np.ones((2,2)))]:
            r=solve_accelerated_ridge(op,y,.01,C,1.)
            self.assertEqual(r['status'],'converged');self.assertTrue(np.all(r['reconstruction']==0))

    def test_iteration_and_backtrack_limits(self):
        op=Diagonal([[1.,2.],[.5,3.]])
        r=solve_accelerated_ridge(op,np.ones((2,2)),.01,dict(C,max_iterations=1),10.)
        self.assertEqual(r['status'],'iteration_limit')
        with self.assertRaisesRegex(RuntimeError,'回溯'):solve_accelerated_ridge(op,np.ones((2,2)),.01,dict(C,max_backtracks=1),1e-8)

    def test_bad_inputs(self):
        op=Diagonal(np.ones((2,2)))
        for y in [np.zeros((3,3)),np.full((2,2),np.nan),np.ones((2,2),dtype=complex)]:
            with self.assertRaises(ValueError):solve_accelerated_ridge(op,y,.01,C,1.)
        for alpha,L in [(-1.,1.),(.01,0.),(np.nan,1.)]:
            with self.assertRaises(ValueError):solve_accelerated_ridge(op,np.ones((2,2)),alpha,C,L)

    def test_negative_extrapolation_is_not_clipped(self):
        class MatrixOperator:
            cube_shape=(1,1,2);output_shape=(2,1)
            matrix=np.array([[1.,.9],[0.,.1]])
            def forward(self,x):return (self.matrix@x.ravel()).reshape(self.output_shape)
            def adjoint(self,y):return (self.matrix.T@y.ravel()).reshape(self.cube_shape)
        op=MatrixOperator();y=np.array([[1.],[-1.]])
        r=solve_accelerated_ridge(op,y,.001,C,2.)
        self.assertEqual(r['status'],'converged');self.assertTrue(np.all(r['reconstruction']>=0))
        self.assertTrue(any(h['extrapolated_negative_count']>0 for h in r['history']))
        ref=lsq_linear(np.vstack([op.matrix,np.sqrt(.001)*np.eye(2)]),np.r_[y.ravel(),0.,0.],bounds=(0,np.inf),tol=1e-13)
        np.testing.assert_allclose(r['reconstruction'].ravel(),ref.x,rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()
