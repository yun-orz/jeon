# -*- coding: utf-8 -*-
"""电子响应及噪声模型的物理、统计和随机流检查。"""
import unittest
import numpy as np
from optics.electron_noise import electron_response,sanitize_poisson_mean,sample_electrons,noise_statistics,random_stream

class ElectronNoiseTests(unittest.TestCase):
    def test_photon_response(self):
        response,energy=electron_response([450e-9,540e-9,630e-9],[.25,.5,0],540e-9)
        np.testing.assert_allclose(response,[.5*450/540,1,0]);np.testing.assert_allclose(energy,6.62607015e-34*299792458/np.array([450e-9,540e-9,630e-9]))
    def test_bad_reference(self):
        for lam,qe,ref in [([1,2],[.5,0],2),([1,1],[.5,.5],1),([1,2],[.5,1.1],1)]:
            with self.assertRaises(ValueError):electron_response(lam,qe,ref)
    def test_negative_roundoff(self):
        clean,record=sanitize_poisson_mean(np.array([1.,-1e-15]),1e-12)
        self.assertEqual(record['negative_count'],1);self.assertEqual(clean[1],0);self.assertEqual(record['added_electrons'],1e-15)
        with self.assertRaises(ValueError):sanitize_poisson_mean([1.,-.01],1e-12)
    def test_roles_and_order(self):
        a=sample_electrons(np.full((9,9),7.),2.,2019,'A');sample_electrons(np.ones((3,3)),2.,2019,'B')
        b=sample_electrons(np.full((9,9),7.),2.,2019,'A')
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertFalse(np.array_equal(random_stream(1,'A','shot').random(10),random_stream(1,'A','read').random(10)))
    def test_negative_readout(self):
        sample=sample_electrons(np.zeros((100,100)),2.,2019,'zero')
        self.assertTrue(np.any(sample['noisy_measurement_e']<0));self.assertTrue(np.all(sample['shot_counts']==0))
        np.testing.assert_array_equal(sample['noisy_measurement_e'],sample['read_noise_e'])
    def test_zero_and_single_noise_controls(self):
        for mean,sigma in [(np.zeros((64,64)),0),(np.zeros((64,64)),2),(np.full((64,64),17.),0)]:
            r=noise_statistics(mean,sigma,2019,'controls',64,8,8)
            self.assertTrue(r['passed']);self.assertEqual(r['sample_count'],64*64*64)
    def test_spatial_moments_independent(self):
        mu=np.arange(256).reshape(16,16)/10;sigma=2.;n=64
        r=noise_statistics(mu,sigma,2019,'spatial',n,8,8);s=r['sufficient_statistics'];var=mu+sigma*sigma
        self.assertAlmostEqual(r['checks']['joint_mean']['z'],s['joint']/np.sqrt(n*var.sum()))
        self.assertAlmostEqual(r['checks']['joint_second_moment']['z'],(s['joint_square']-n*var.sum())/np.sqrt(n*np.sum(mu+2*var*var)))
        self.assertTrue(r['passed'])
    def test_blocks_do_not_change_random_draws(self):
        a=noise_statistics(np.full((10,10),7.),2,2019,'blocks',64,1,8)
        b=noise_statistics(np.full((10,10),7.),2,2019,'blocks',64,13,8)
        for key in a['sufficient_statistics']:self.assertAlmostEqual(a['sufficient_statistics'][key],b['sufficient_statistics'][key],places=8)
    def test_invalid_parameters(self):
        for mean,sigma in [([-1],2),([np.nan],2),([1],-2),([],2)]:
            with self.assertRaises(ValueError):sample_electrons(mean,sigma,2019,'bad')
        with self.assertRaises(ValueError):random_stream(True,'A','B')
        with self.assertRaises(ValueError):noise_statistics([1],2,2019,'A',1,8,8)

if __name__=='__main__':unittest.main()
