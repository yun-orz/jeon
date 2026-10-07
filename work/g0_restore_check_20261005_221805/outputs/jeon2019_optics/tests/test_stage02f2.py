# -*- coding: utf-8 -*-
"""电子量纲、加权链及真值隔离的有意义小阵列检查。"""
import copy,json,unittest
from pathlib import Path
import numpy as np
from optics.electron_reconstruction import make_electron_operator,WeightedOperator,measured_weights,reconstruct_electrons
from optics.imaging import direct_shift_sum,operator_checks

class ElectronReconstructionTests(unittest.TestCase):
    def setUp(self):
        k=np.zeros((3,3,3));k[:,1,1]=[.5,.6,.7];k[:,1,2]=.1
        self.base=make_electron_operator(k,[4,4],1e-6,[1,.8,.9],[450e-9,540e-9,630e-9],[.5,.5,.5],540e-9,100,[0,6,0,6])
        self.c=json.loads((Path(__file__).resolve().parents[1]/'config_stage02f2.json').read_text(encoding='utf-8'))['reconstruction']
    def test_electron_direct_and_scale(self):
        x=np.arange(48).reshape(3,4,4)/100
        np.testing.assert_allclose(self.base.forward(x),direct_shift_sum(x,self.base.kernels,self.base.response),atol=1e-13)
        np.testing.assert_allclose(self.base.response,100*np.array([1,.8,.9])*np.array([450/540,1,630/540]))
    def test_signed_adjoint_and_gradient(self):
        for op in [self.base,WeightedOperator(self.base,np.arange(36).reshape(6,6)/36+.1)]:
            check=operator_checks(op,2019,np.ones(op.output_shape));self.assertLess(check['inner_product_relative_error'],1e-10);self.assertLess(check['gradient_relative_error'],1e-5)
    def test_weighted_chain_explicit(self):
        w=np.arange(36).reshape(6,6)+1.;op=WeightedOperator(self.base,w);y=np.arange(36).reshape(6,6)-10
        np.testing.assert_array_equal(op.adjoint(y),self.base.adjoint(w*y))
    def test_negative_target_and_background(self):
        z=np.full((6,6),-2.);fit=reconstruct_electrons(self.base,z,3.,2.,'fixed_weighted',self.c)
        np.testing.assert_array_equal(fit['weights'],np.full((6,6),.5));np.testing.assert_array_equal(fit['target'],np.full((6,6),-2.5))
        self.assertTrue(np.all(fit['result']['reconstruction']==0));self.assertEqual(fit['result']['status'],'converged')
    def test_weight_variance_floor(self):
        w,var=measured_weights([-3.,0.,10.],2.,4.)
        np.testing.assert_array_equal(var,[4,4,14]);np.testing.assert_allclose(w,1/np.sqrt([4,4,14]))
    def test_truth_isolation(self):
        # 外层评价真值改变，电子测量和器件参数不变；恢复接口没有真值参数。
        z=np.linspace(-2,30,36).reshape(6,6)
        def trial(truth):
            result=reconstruct_electrons(self.base,z,0.,2.,'fixed_weighted',self.c)
            evaluation=float(np.linalg.norm(result['result']['reconstruction']-truth))
            return result,evaluation
        a,ea=trial(np.zeros((3,4,4)));b,eb=trial(np.ones((3,4,4)))
        np.testing.assert_array_equal(a['result']['reconstruction'],b['result']['reconstruction']);self.assertEqual(a['alpha'],b['alpha']);self.assertNotEqual(ea,eb)
    def test_objective_certificate_and_status(self):
        x=np.zeros((3,4,4));x[:,2,2]=1.;z=self.base.forward(x)
        fit=reconstruct_electrons(self.base,z,0.,2.,'unweighted',self.c);r=fit['result'];h=r['history'];recovered=r['reconstruction'];res=self.base.forward(recovered)-z
        objective=.5*np.sum(res*res)+.5*fit['alpha']*np.sum(recovered*recovered)
        self.assertAlmostEqual(objective,h[-1]['objective']);self.assertTrue(np.all(recovered>=0));self.assertTrue(all(b['objective']<=a['objective']+1e-10 for a,b in zip(h,h[1:])))
        self.assertEqual(r['status'],'converged');self.assertLessEqual(h[-1]['projected_gradient_relative'],self.c['solver']['gradient_tolerance'])
        gradient=self.base.adjoint(res)+fit['alpha']*recovered;norm=np.linalg.norm(np.where(recovered>0,gradient,np.minimum(gradient,0)))
        self.assertAlmostEqual(norm/fit['alpha'],fit['certificate']['distance_upper_bound'])
    def test_bad_inputs(self):
        for w in [np.zeros((6,6)),np.ones((3,3)),np.full((6,6),np.nan)]:
            with self.assertRaises(ValueError):WeightedOperator(self.base,w)
        with self.assertRaises(ValueError):measured_weights([1],2.,0)
        with self.assertRaises(ValueError):reconstruct_electrons(self.base,np.ones((6,6)),-1.,2.,'unweighted',self.c)
    def test_iteration_limit_is_not_convergence(self):
        c=copy.deepcopy(self.c);c['solver']['max_iterations']=1;z=np.arange(36).reshape(6,6)
        self.assertEqual(reconstruct_electrons(self.base,z,0.,2.,'unweighted',c)['result']['status'],'iteration_limit')

if __name__=='__main__':unittest.main()
