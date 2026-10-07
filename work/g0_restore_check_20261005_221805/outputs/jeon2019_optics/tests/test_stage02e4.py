# -*- coding: utf-8 -*-
"""共同α、严格复用前提、实际回退计算和差异理论区间检查。"""
import contextlib,copy,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from optics.stage02e3_source import load_common_alpha_source,reuse_compatible
from optics.comparison_uncertainty import error_difference_interval
from optics.stage02c_source import tree_sha
import main_stage02e4 as m
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads((ROOT/'config_stage02e4.json').read_text(encoding='utf-8'))
        cls.items,cls.evidence=load_common_alpha_source(ROOT/cls.c['source_run'],ROOT)
    def test_real_source_alpha_and_readonly(self):
        self.assertEqual(len(self.items),6);self.assertEqual(len(self.evidence['sources']),8)
        alpha=self.evidence['common_alpha'];self.assertGreater(alpha,0)
        for item in self.items:
            if item['device']=='continuous':self.assertEqual(float(item['values']['alpha']),alpha)
            else:self.assertNotEqual(float(item['values']['alpha']),alpha)
        for s in self.evidence['sources']:self.assertEqual(tree_sha(Path(s['path'])),s['sha256'])
    def test_strict_reuse_conditions(self):
        alpha=self.evidence['common_alpha'];item=next(i for i in self.items if i['device']=='continuous')
        self.assertTrue(reuse_compatible(item,self.c,alpha))
        near=next(i for i in self.items if i['device']=='nearest_depth');self.assertFalse(reuse_compatible(near,self.c,alpha))
        for change in ['iterations','tolerance','disabled']:
            c=copy.deepcopy(self.c)
            if change=='iterations':c['solver']['max_iterations']=1
            elif change=='tolerance':c['solver']['gradient_tolerance']=1e-9
            else:c['reuse_continuous']=False
            self.assertFalse(reuse_compatible(item,c,alpha))
        self.assertFalse(reuse_compatible(item,self.c,alpha*2))
        bad=copy.deepcopy(item);bad['record']['solver_status']='iteration_limit';self.assertFalse(reuse_compatible(bad,self.c,alpha))
    def test_tampered_cached_reconstruction_rejected(self):
        original=np.load
        class View(dict):
            @property
            def files(self):return list(self)
        class Wrapped:
            def __init__(self,p,*a,**kw):self.z=original(p,*a,**kw)
            def __enter__(self):
                v=View({k:self.z[k].copy() for k in self.z.files});self.z.close();v['reconstruction'][0,0,0]+=.01;return v
            def __exit__(self,*args):pass
        def load(p,*a,**kw):return Wrapped(p,*a,**kw) if 'stage02e3' in str(p) and str(p).endswith('.npz') else original(p,*a,**kw)
        with patch('optics.stage02e3_source.np.load',side_effect=load):
            with self.assertRaises(ValueError):load_common_alpha_source(ROOT/self.c['source_run'],ROOT)
    def test_actual_one_iteration_falls_back(self):
        c=copy.deepcopy(self.c);c['solver']['max_iterations']=1
        with contextlib.redirect_stdout(io.StringIO()):r=m.run(c,no_save=True)
        json.dumps(r,allow_nan=False);self.assertEqual(r['computed_count'],6);self.assertEqual(r['reused_count'],0)
        self.assertEqual(r['completed_jobs'],6);self.assertFalse(r['stability_validation_passed'])
        self.assertTrue(all(x['alpha']==r['common_alpha'] for x in r['records']))
        self.assertTrue(all(x['new_iterations']==1 for x in r['records']))
    def test_cached_execution_exact_and_no_solver_call(self):
        c=copy.deepcopy(self.c);c['devices']=['continuous'];c['scenes']=['lines_and_square']
        with patch('main_stage02e4.solve_accelerated_ridge',side_effect=AssertionError('复用时不应重新迭代')),contextlib.redirect_stdout(io.StringIO()):r=m.run(c,no_save=True)
        self.assertEqual(r['reused_count'],1);self.assertEqual(r['computed_count'],0)
        rec=r['records'][0];self.assertEqual(rec['new_iterations'],0);self.assertEqual(rec['elapsed_s'],0.)
        self.assertEqual(rec['baseline_reconstruction_relative_change'],0.)
        self.assertTrue(r['stability_validation_passed'])
    def test_invalid_config(self):
        m.validate_config(self.c)
        for key,value in [('alpha_strategy','wrong'),('reference_alpha_factor',.1),('reuse_continuous',1),('devices',[]),('budget_seconds',0.)]:
            c=copy.deepcopy(self.c);c[key]=value
            with self.assertRaises(ValueError):m.validate_config(c)
class IntervalTests(unittest.TestCase):
    def test_triangle_bounds_and_zero_crossing(self):
        # 两个显式向量的误差排序区间应包住其真实最优解误差差。
        t=np.array([1.,2.]);xc=np.array([.9,2.1]);xn=np.array([1.2,1.9]);sc=np.array([1.,2.]);sn=np.array([1.1,2.])
        norm=np.linalg.norm(t);ec=np.linalg.norm(xc-t)/norm;en=np.linalg.norm(xn-t)/norm
        r=error_difference_interval(ec,en,np.linalg.norm(xc-sc),np.linalg.norm(xn-sn),norm)
        true_delta=(np.linalg.norm(sn-t)-np.linalg.norm(sc-t))/norm
        self.assertLessEqual(r['lower'],true_delta);self.assertGreaterEqual(r['upper'],true_delta)
        self.assertFalse(error_difference_interval(.1,.1001,.1,.1,1.)['ordering_supported_by_theoretical_bounds'])
        self.assertTrue(error_difference_interval(.1,.2,.001,.001,1.)['ordering_supported_by_theoretical_bounds'])
    def test_invalid_interval_inputs(self):
        for values in [(-1.,.1,0.,0.,1.),(.1,.2,0.,0.,0.),(.1,float('nan'),0.,0.,1.)]:
            with self.assertRaises(ValueError):error_difference_interval(*values)
if __name__=='__main__':unittest.main()
