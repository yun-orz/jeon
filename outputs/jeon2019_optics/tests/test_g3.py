# -*- coding: utf-8 -*-
"""G3：可微光学算子、HQS式(21)与先验梯度验证。"""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optics.g2_rgb import RGBForward
from optics.g3_hqs import TorchRGB,HQSDecoder,SpectralPrior


class TestG3(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(18);rng=np.random.default_rng(18)
        self.k=rng.random((3,3,5));self.k/=self.k.sum(axis=(1,2))[:,None,None]
        self.r=rng.random((3,3));self.op=TorchRGB(self.k,self.r)

    def test_matches_numpy_and_transpose(self):
        rng=np.random.default_rng(31);x=rng.normal(size=(10,9,3));y=rng.normal(size=(10,9,3))
        npop=RGBForward(self.k,self.r)
        tensor=lambda a:torch.tensor(a.transpose(2,0,1)[None])
        out=self.op(tensor(x));adj=self.op.adjoint(tensor(y))
        np.testing.assert_allclose(out[0].permute(1,2,0),npop.forward(x),atol=1e-13)
        np.testing.assert_allclose(adj[0].permute(1,2,0),npop.adjoint(y),atol=1e-13)
        self.assertAlmostEqual(float((out*tensor(y)).sum()),float((tensor(x)*adj).sum()),delta=1e-12)

    def test_autograd_equals_adjoint(self):
        x=torch.randn(1,3,6,7,dtype=torch.float64,requires_grad=True);y=torch.randn_like(x)
        loss=.5*((self.op(x)-y)**2).sum();loss.backward()
        torch.testing.assert_close(x.grad,self.op.adjoint(self.op(x.detach())-y),rtol=1e-12,atol=1e-12)

    def test_one_stage_equation21_and_gradients(self):
        c=dict(stages=2,features=4,levels=4,epsilon_init=.05,rho_init=.1,threshold_init=.01)
        model=HQSDecoder(self.op,c).double();y=torch.randn(1,3,9,11,dtype=torch.float64)
        out,initial,trace=model(y,True)
        for row in trace:
            expected=(1-row['epsilon']*row['rho'])*row['previous']-row['epsilon']*self.op.adjoint(self.op(row['previous']))+row['epsilon']*initial+row['epsilon']*row['rho']*row['prior']
            torch.testing.assert_close(expected,row['updated'],rtol=1e-12,atol=1e-12)
        out.square().mean().backward()
        for stage in model.stages:
            self.assertGreater(float(stage.raw_epsilon.grad.abs()),0)
            self.assertGreater(float(stage.raw_rho.grad.abs()),0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in stage.prior.parameters()))
        self.assertIsNot(model.stages[0].prior.head.weight,model.stages[1].prior.head.weight)

    def test_prior_odd_size_and_zero_threshold(self):
        prior=SpectralPrior(3,4,4,.01)
        out=prior(torch.randn(2,3,15,17));self.assertEqual(tuple(out.shape),(2,3,15,17))
        self.assertGreater(float(prior.threshold.detach()),0.)


if __name__=='__main__':unittest.main()
