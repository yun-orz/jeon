# -*- coding: utf-8 -*-
"""固定实际α源重验及错误源/门槛配置拒绝，无文件删除。"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from optics.stage02d2_source import load_pg_source
from optics.stage02c_source import tree_sha
from main_stage02d3 import validate_config

ROOT=Path(__file__).resolve().parents[1]


class AcceleratedSourceTests(unittest.TestCase):
    def test_real_source_fixed_alpha_and_readonly(self):
        src=ROOT/'results/stage02d2/run_20261002_210155';before=tree_sha(src)
        items,evidence=load_pg_source(src,ROOT)
        self.assertEqual(sum(len(i['baselines']) for i in items),8)
        self.assertFalse(evidence['baseline_all_converged'])
        for i in items:
            for b in i['baselines'].values():self.assertEqual(b['alpha'],b['record']['alpha'])
        self.assertEqual(before,tree_sha(src))

    def test_reject_altered_actual_alpha(self):
        original=np.load
        @contextmanager
        def altered(path,*args,**kwargs):
            with original(path,*args,**kwargs) as z:
                if Path(path).name=='continuous_lines_and_square_crop.npz':
                    data={k:z[k] for k in z.files};data['alpha']=data['alpha']*2;yield data
                else:yield z
        with patch('optics.stage02d2_source.np.load',new=altered):
            with self.assertRaisesRegex(ValueError,'实际α'):load_pg_source(ROOT/'results/stage02d2/run_20261002_210155',ROOT)

    def test_missing_source_and_bad_configuration(self):
        with self.assertRaises(FileNotFoundError):load_pg_source(ROOT/'不存在的PG源run',ROOT)
        cfg=json.loads((ROOT/'config_stage02d3.json').read_text(encoding='utf-8'));validate_config(cfg)
        for key,value in [('check_seed',-1),('full_diagnostic_scene','unknown')]:
            c=copy.deepcopy(cfg);c[key]=value
            with self.assertRaises(ValueError):validate_config(c)

    def test_incomplete_run_report_is_serializable(self):
        # 实际生成包含基线对照的报告，检查未收敛状态也能严格JSON序列化。
        from main_stage02d3 import run
        from contextlib import redirect_stdout
        import io
        cfg=json.loads((ROOT/'config_stage02d3.json').read_text(encoding='utf-8'))
        cfg['solver']['max_iterations']=1
        with redirect_stdout(io.StringIO()):r=run(cfg,no_save=True)
        json.dumps(r,allow_nan=False)
        self.assertFalse(r['optimization_validation_passed']);self.assertFalse(r['all_reconstructions_converged'])
        self.assertTrue(all(type(v['objective_not_worse_than_PG']) is bool for v in r['records']))


if __name__=='__main__':unittest.main()
