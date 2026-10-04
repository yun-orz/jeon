# -*- coding: utf-8 -*-
"""02E-2来源状态、篡改和实际未收敛报告检查。"""
import contextlib,copy,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from optics.stage02e1_source import load_sensitivity_source
from optics.stage02c_source import tree_sha
import main_stage02e2 as m
class Tests(unittest.TestCase):
    def config(self):return json.loads((ROOT/'config_stage02e2.json').read_text(encoding='utf-8'))
    def test_real_incomplete_optimization_source_allowed(self):
        c=self.config();items,e=load_sensitivity_source(ROOT/c['source_run'],ROOT)
        self.assertEqual(len(items),12);self.assertFalse(e['baseline_all_converged']);self.assertFalse(e['baseline_optimization_validation_passed'])
        self.assertEqual(len(e['sources']),5)
        for s in e['sources']:self.assertEqual(tree_sha(Path(s['path'])),s['sha256'])
    def test_forged_total_optimization_flag_rejected(self):
        source=ROOT/self.config()['source_run'];original=Path.read_text
        def read(p,*a,**kw):
            s=original(p,*a,**kw)
            if p==source/'metrics/validation.json':
                d=json.loads(s);d['optimization_validation_passed']=True;return json.dumps(d)
            return s
        with patch.object(Path,'read_text',read):
            with self.assertRaises(ValueError):load_sensitivity_source(source,ROOT)
    def test_tampered_alpha_rejected(self):
        original=np.load
        class View(dict):
            @property
            def files(self):return list(self)
        class Wrapped:
            def __init__(self,p,*a,**kw):self.z=original(p,*a,**kw)
            def __enter__(self):
                v=View({k:self.z[k].copy() for k in self.z.files});self.z.close();v['alpha']*=2;return v
            def __exit__(self,*args):pass
        def load(p,*a,**kw):return Wrapped(p,*a,**kw) if 'stage02e1' in str(p) and str(p).endswith('.npz') else original(p,*a,**kw)
        with patch('optics.stage02e1_source.np.load',side_effect=load):
            with self.assertRaises(ValueError):load_sensitivity_source(ROOT/self.config()['source_run'],ROOT)
    def test_invalid_configuration(self):
        c=self.config();m.validate_config(c)
        for key,value in [('scenes',['separated_points']),('scenes',['coincident_points','coincident_points']),('alpha_factor',.1),('budget_seconds',0.),('certificate_relative_threshold',float('nan')),('check_seed',True)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):m.validate_config(bad)
    def test_actual_iteration_limit_report_json(self):
        c=self.config();c['solver']['max_iterations']=1
        with contextlib.redirect_stdout(io.StringIO()):r=m.run(c,no_save=True)
        json.dumps(r,allow_nan=False)
        self.assertEqual(r['completed_jobs'],2);self.assertTrue(r['validation_passed'])
        self.assertFalse(r['optimization_validation_passed']);self.assertFalse(r['stability_validation_passed'])
        self.assertTrue(all(x['solver_status']=='iteration_limit' for x in r['records']))
    def test_missing_source(self):
        with self.assertRaises(FileNotFoundError):load_sensitivity_source(ROOT/'results/不存在的02E2源',ROOT)
if __name__=='__main__':unittest.main()
