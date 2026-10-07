# -*- coding: utf-8 -*-
"""02E-3器件身份、参考状态、固定数据及真实上限报告检查。"""
import contextlib,copy,io,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from optics.stage02e2_source import load_comparison_source
from optics.stage02c_source import tree_sha
import main_stage02e3 as m
class Tests(unittest.TestCase):
    def config(self):return json.loads((ROOT/'config_stage02e3.json').read_text(encoding='utf-8'))
    def load(self):
        c=self.config();return load_comparison_source(ROOT/c['source_run'],[ROOT/p for p in c['reference_runs']],ROOT)
    def test_real_pair_source_and_reference_status(self):
        items,e=self.load();self.assertEqual(len(items),6);self.assertEqual(len(e['sources']),7)
        self.assertFalse(e['reference_details'][0]['stability_validation_passed']);self.assertTrue(e['reference_details'][1]['stability_validation_passed'])
        for s in e['sources']:self.assertEqual(tree_sha(Path(s['path'])),s['sha256'])
        for scene in ['coincident_points','separated_points','lines_and_square']:
            group=[i for i in items if i['scene']==scene];np.testing.assert_array_equal(group[0]['values']['truth'],group[1]['values']['truth'])
            self.assertNotEqual(float(group[0]['values']['alpha']),float(group[1]['values']['alpha']))
    def test_forged_reference_certificate_total_rejected(self):
        c=self.config();source=(ROOT/c['reference_runs'][0]).resolve();original=Path.read_text
        def read(p,*a,**kw):
            s=original(p,*a,**kw)
            if p==source/'metrics/validation.json':
                d=json.loads(s);d['certificate_validation_passed']=True;return json.dumps(d)
            return s
        with patch.object(Path,'read_text',read):
            with self.assertRaises(ValueError):self.load()
    def test_tampered_reference_alpha_rejected(self):
        original=np.load
        class View(dict):
            @property
            def files(self):return list(self)
        class Wrapped:
            def __init__(self,p,*a,**kw):self.z=original(p,*a,**kw)
            def __enter__(self):
                v=View({k:self.z[k].copy() for k in self.z.files});self.z.close();v['alpha']*=2;return v
            def __exit__(self,*args):pass
        def load(p,*a,**kw):return Wrapped(p,*a,**kw) if 'stage02e2' in str(p) and str(p).endswith('.npz') else original(p,*a,**kw)
        with patch('optics.stage02e2_source.np.load',side_effect=load):
            with self.assertRaises(ValueError):self.load()
    def test_invalid_configurations(self):
        c=self.config();m.validate_config(c)
        for key,value in [('devices',['continuous','continuous']),('devices',['unknown']),('scenes',['unknown']),('alpha_factor',1.),('reference_runs',[]),('budget_seconds',0.),('certificate_relative_threshold',float('nan'))]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):m.validate_config(bad)
    def test_actual_one_iteration_report_json(self):
        c=self.config();c['solver']['max_iterations']=1
        with contextlib.redirect_stdout(io.StringIO()):r=m.run(c,no_save=True)
        json.dumps(r,allow_nan=False)
        self.assertEqual(r['completed_jobs'],6);self.assertEqual(len(r['pairs']),3)
        self.assertTrue(r['validation_passed']);self.assertFalse(r['stability_validation_passed'])
        self.assertTrue(all(x['solver_status']=='iteration_limit' for x in r['records']))
        self.assertTrue(all(not p['both_stable'] for p in r['pairs']))
    def test_duplicate_references_rejected(self):
        c=self.config();p=ROOT/c['reference_runs'][0]
        with self.assertRaises(ValueError):load_comparison_source(ROOT/c['source_run'],[p,p],ROOT)
if __name__=='__main__':unittest.main()
