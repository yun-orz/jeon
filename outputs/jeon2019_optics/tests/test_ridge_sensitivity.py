# -*- coding: utf-8 -*-
"""独立NNLS、强凸界与固定来源/配置的有效检查。"""
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from optics.ridge_certificate import ridge_certificate
from optics.reconstruction_accelerated import solve_accelerated_ridge
from optics.stage02d3_source import load_accelerated_source
from optics.stage02c_source import tree_sha
import main_stage02e1 as main

class MatrixOperator:
    def __init__(self,A):
        self.A=np.asarray(A,dtype=float);self.cube_shape=(self.A.shape[1],);self.output_shape=(self.A.shape[0],)
    def forward(self,x):return self.A@x
    def adjoint(self,y):return self.A.T@y

class CertificateTests(unittest.TestCase):
    def test_independent_nnls_bounds(self):
        rng=np.random.default_rng(2019)
        for alpha in [1e-3,.1,10.]:
            A=rng.normal(size=(8,5));y=rng.normal(size=8);op=MatrixOperator(A)
            extended=np.vstack([A,np.sqrt(alpha)*np.eye(5)])
            target=np.r_[y,np.zeros(5)];optimal,_=nnls(extended,target)
            for x in [np.zeros(5),rng.uniform(size=5),optimal,optimal*.93]:
                b=ridge_certificate(op,y,x,alpha)
                distance=np.linalg.norm(x-optimal)
                F=lambda z:.5*np.linalg.norm(A@z-y)**2+.5*alpha*np.linalg.norm(z)**2
                self.assertLessEqual(distance,b['distance_upper_bound']+1e-10)
                self.assertLessEqual(F(x)-F(optimal),b['objective_gap_upper_bound']+1e-10)
    def test_exact_zero_vs_tiny_positive(self):
        op=MatrixOperator(np.eye(2));y=np.array([-1.,-1.])
        zero=ridge_certificate(op,y,np.zeros(2),1.)
        tiny=ridge_certificate(op,y,np.array([1e-30,0.]),1.)
        self.assertEqual(zero['minimum_subgradient_norm'],0.)
        self.assertAlmostEqual(tiny['minimum_subgradient_norm'],1.)
        self.assertIsNone(zero['distance_upper_bound_relative'])
    def test_invalid_inputs(self):
        op=MatrixOperator(np.eye(2))
        for alpha in [0.,-1.,float('nan'),True]:
            with self.assertRaises(ValueError):ridge_certificate(op,np.ones(2),np.ones(2),alpha)
        for x in [np.array([-1.,1.]),np.ones(3),np.array([1.,float('inf')])]:
            with self.assertRaises(ValueError):ridge_certificate(op,np.ones(2),x,1.)
    def test_alpha_solutions_against_nnls(self):
        A=np.array([[1.,.9],[0.,.1],[.1,.2]]);y=np.array([1.,-.2,.3]);op=MatrixOperator(A)
        c=dict(max_iterations=5000,max_backtracks=50,backtracking_factor=2.,gradient_tolerance=1e-10,objective_tolerance=1e-12)
        for alpha in [.001,.01,.1,1.]:
            r=solve_accelerated_ridge(op,y,alpha,c,float(np.linalg.norm(A,2)**2+alpha))
            ref,_=nnls(np.vstack([A,np.sqrt(alpha)*np.eye(2)]),np.r_[y,np.zeros(2)])
            self.assertEqual(r['status'],'converged')
            np.testing.assert_allclose(r['reconstruction'],ref,atol=1e-7,rtol=0)

class SourceTests(unittest.TestCase):
    def test_real_source_and_readonly(self):
        c=json.loads((ROOT/'config_stage02e1.json').read_text(encoding='utf-8'))
        items,evidence=load_accelerated_source(ROOT/c['source_run'],ROOT)
        self.assertEqual(len(items),3)
        self.assertEqual({x['scene'] for x in items},{'lines_and_square','coincident_points','separated_points'})
        self.assertEqual(len(evidence['sources']),4)
        for source in evidence['sources']:self.assertEqual(tree_sha(Path(source['path'])),source['sha256'])
    def test_tampered_alpha_rejected(self):
        c=json.loads((ROOT/'config_stage02e1.json').read_text(encoding='utf-8'))
        original=np.load
        class ArrayView(dict):
            @property
            def files(self):return list(self)
        class Wrapped:
            def __init__(self,p,*a,**kw):self.z=original(p,*a,**kw)
            def __enter__(self):
                values=ArrayView({k:self.z[k].copy() for k in self.z.files})
                if 'reconstruction' in values:values['alpha']=values['alpha']*2
                self.z.close();return values
            def __exit__(self,*args):pass
        # 只篡改加速阶段数组的读取视图，不改变任何磁盘文件。
        def load(p,*a,**kw):
            if 'stage02d3' in str(p) and str(p).endswith('.npz'):return Wrapped(p,*a,**kw)
            return original(p,*a,**kw)
        with patch('optics.stage02d3_source.np.load',side_effect=load):
            with self.assertRaises(ValueError):load_accelerated_source(ROOT/c['source_run'],ROOT)
    def test_invalid_config(self):
        original=json.loads((ROOT/'config_stage02e1.json').read_text(encoding='utf-8'))
        main.validate_config(original)
        for key,value in [('alpha_factors',[.1,.1,1.]),('alpha_factors',[0.,1.]),('budget_seconds',0.),('check_seed',True)]:
            c=json.loads(json.dumps(original));c[key]=value
            with self.assertRaises(ValueError):main.validate_config(c)
        c=json.loads(json.dumps(original));c['stricter']['gradient_tolerance']=c['solver']['gradient_tolerance']
        with self.assertRaises(ValueError):main.validate_config(c)

if __name__=='__main__':unittest.main()
