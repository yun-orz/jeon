# -*- coding: utf-8 -*-
"""精准反例与本地运行验证；证据保留，不自动删除。"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch, MagicMock
import numpy as np
import main_stage02 as main
from optics.coordinates import make_grid
from optics import stage02_runtime as rt

PROJECT=Path(__file__).resolve().parent.parent


def evidence_dir(name):
    p=PROJECT/'results/test_stage02_r2'/(name+'_'+uuid.uuid4().hex[:10])
    p.mkdir(parents=True,exist_ok=False)
    return p


def _child_env():
    """显式确定子进程的 I/O 编码（R8）。

    读端 `encoding='utf-8'` 只解决解码；中文 Windows 下子进程默认按 GBK 输出，
    因此必须**同时**设置写端编码。这里不传 `errors`，让真实编码问题直接暴露，
    而不是被 `replace` 静默掩盖成乱码。
    """
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    return env


class TestStage02R2(unittest.TestCase):
    def setUp(self):
        self.cfg=main.load_and_validate_config(PROJECT/'config_stage02a.json')

    def test_relative_config_cli_outside_project(self):
        cmd=[sys.executable,'-B',str(PROJECT/'main_stage02.py'),'--config','config_stage02a.json','--only','height','--no-save']
        res=subprocess.run(cmd,cwd=PROJECT.parent.parent,capture_output=True, text=True, encoding='utf-8', env=_child_env())
        self.assertEqual(res.returncode,0,res.stderr)
        self.assertIn(str(PROJECT/'config_stage02a.json'),res.stdout)

    def test_nonempty_directory_preserved_before_logging(self):
        p=evidence_dir('protected')
        for name in ['run.log','config_used.json','old_array.npz']:
            (p/name).write_bytes(('既有文件'+name).encode('utf-8'))
        before={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir()}
        cmd=[sys.executable,'-B',str(PROJECT/'main_stage02.py'),'--only','height','--run-dir',str(p)]
        res=subprocess.run(cmd,cwd=PROJECT,capture_output=True, text=True, encoding='utf-8', env=_child_env())
        self.assertNotEqual(res.returncode,0)
        self.assertEqual(before,{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir()})

    def test_direct_run_rejects_existing_products(self):
        p=evidence_dir('direct');(p/'old.txt').write_text('保留',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'拒绝覆盖'):
            main.run_stage02a(self.cfg,p,only_mode='height')

    def test_fractional_nm_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['optical']['preview_wavelengths_m']=[540e-9,540.05e-9,660e-9]
        p=evidence_dir('near_lambda')/'config.json';p.write_text(json.dumps(cfg),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'整数nm'):main.load_and_validate_config(p)

    def test_same_integer_file_identifier_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['optical']['preview_wavelengths_m']=[540e-9,540e-9+1e-19,660e-9]
        p=evidence_dir('same_filename')/'config.json';p.write_text(json.dumps(cfg),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'文件标识重复'):main.load_and_validate_config(p)

    def test_missing_fields_and_offset_rejected(self):
        for key,val in [('runtime',{}),('acceptance',{}),('runtime',[])]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=val
            p=evidence_dir('missing')/'config.json';p.write_text(json.dumps(cfg),encoding='utf-8')
            with self.assertRaises(ValueError):main.load_and_validate_config(p)
        for val in [float('nan'),float('inf'),True]:
            cfg=copy.deepcopy(self.cfg);cfg['optical']['h_offset_m']=val
            p=evidence_dir('offset')/'config.json';p.write_text(json.dumps(cfg),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'h_offset'):main.load_and_validate_config(p)

    def test_control_absolute_power_excess_fails(self):
        original=main.fresnel_kernel_separable
        with patch.object(main,'fresnel_kernel_separable',side_effect=lambda **kw:2*original(**kw)):
            with self.assertRaisesRegex(RuntimeError,'控制光场物理检查失败'):
                main.run_stage02a(self.cfg,None,only_mode='control')

    def test_control_invalid_fields_write_failed_state(self):
        for name,arr in [('zero',np.zeros((301,301),complex)),('nan',np.full((301,301),np.nan,complex)),('shape',np.ones((2,3),complex))]:
            p=rt.prepare_run_dir(evidence_dir(name)/'run')
            with patch.object(main,'fresnel_kernel_separable',return_value=arr):
                with self.assertRaisesRegex(RuntimeError,'控制光场物理检查失败'):
                    main.run_stage02a(self.cfg,p,only_mode='control')
            state=json.loads((p/'completion_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'],'failed');self.assertEqual(state['error_type'],'RuntimeError')
            self.assertEqual(state['wavelengths_completed'],[])

    def test_required_checks_not_replaced_by_count(self):
        checks={k:{'status':'pass'} for k in rt.required_checks('all')}
        self.assertTrue(rt.checks_pass(checks,'all'))
        del checks['control_550nm'];checks['unrelated']={'status':'pass'}
        self.assertFalse(rt.checks_pass(checks,'all'))

    def test_global_phase_switch_reaches_propagation(self):
        self.cfg['propagation']['include_global_phase']=False
        original=main.fresnel_kernel_separable;flags=[]
        def spy(**kw):
            flags.append(kw['include_global_phase']);return original(**kw)
        with patch.object(main,'fresnel_kernel_separable',side_effect=spy):
            main.run_stage02a(self.cfg,None,only_mode='control')
        self.assertEqual(flags,[False])

    def test_missing_artifact_rejected(self):
        p=evidence_dir('missing_array');(p/'arrays').mkdir()
        with self.assertRaises(FileNotFoundError):
            rt.verify_saved_artifacts(p,'control',self.cfg,'',make_grid(1101,1e-6),make_grid(301,1e-6))

    def test_control_only_report_marks_doe_not_run(self):
        main.plt=None
        result=main.run_stage02a(self.cfg,None,only_mode='control')
        report=main.generate_markdown_report(self.cfg,result['self_check'],None,'',Path('run'),1.0,'control')
        line=next(s for s in report.splitlines() if '**固定器件指纹**' in s)
        self.assertIn('NOT_RUN',line);self.assertNotIn('**PASS**',line)
        self.assertEqual(result['completion_state']['status'],'partial')

    def test_runtime_combinations_and_cli(self):
        for save in [False,True]:
            for show in [False,True]:
                cfg=copy.deepcopy(self.cfg);cfg['runtime']={'save_results':save,'show_plots':show}
                self.assertEqual(rt.effective_runtime(cfg),cfg['runtime'])
                self.assertEqual(rt.effective_runtime(cfg,True,True),{'save_results':False,'show_plots':True})

    def test_report_uses_configured_control_and_threshold(self):
        cfg=copy.deepcopy(self.cfg);cfg['optical']['control_wavelength_m']=600e-9
        cfg['acceptance']['control_dark_ring_rel_err_max']=0.0123
        checks={k:{'status':'not_run'} for k in rt.required_checks('all')}
        report=main.generate_markdown_report(cfg,checks,None,'',Path('run'),0,'height')
        self.assertIn('600nm 控制暗环',report);self.assertIn('0.0123',report)

    def test_backend_probe_before_pyplot_and_fallback(self):
        events=[];fake=MagicMock();fake.rcParams={}
        def imported(name):events.append('import:'+name);return fake
        with patch.object(rt,'_probe_tk',side_effect=lambda:events.append('probe')),patch('matplotlib.use',side_effect=lambda *a,**k:events.append('use:'+a[0])),patch.object(rt.importlib,'import_module',side_effect=imported):
            _,info=rt.configure_plotting(True)
        self.assertEqual(events,['probe','use:TkAgg','import:matplotlib.pyplot']);self.assertTrue(info['available'])
        with patch.object(rt,'_probe_tk',side_effect=RuntimeError('模拟无窗口')),patch('matplotlib.use'),patch.object(rt.importlib,'import_module',return_value=fake):
            _,info=rt.configure_plotting(True)
        self.assertFalse(info['available']);self.assertEqual(info['backend'],'Agg')

    def test_config_and_cli_show_reach_backend(self):
        for config_show,cli_show in [(True,False),(False,True),(False,False)]:
            cfg=copy.deepcopy(self.cfg);cfg['runtime']['show_plots']=config_show
            args=argparse.Namespace(config='config_stage02a.json',only='height',no_save=True,show_plots=cli_show,run_dir='')
            fake=MagicMock()
            with patch.object(main,'parse_args',return_value=args),patch.object(main,'load_and_validate_config',return_value=cfg),patch.object(main,'configure_plotting',return_value=(fake,{'available':config_show or cli_show,'backend':'mock','reason':'mock'})) as select,patch.object(main,'run_stage02a',return_value={'completion_state':{'all_checks_pass':False,'status':'partial'}}) as execute:
                with self.assertRaises(SystemExit) as ctx:main.main()
            self.assertEqual(ctx.exception.code,0);select.assert_called_once_with(config_show or cli_show)
            self.assertEqual(execute.call_args.kwargs['keep_figures_for_show'],config_show or cli_show)
        main.plt=None

    def test_figure_lifecycle_when_show_requested(self):
        main.plt=None
        result=main.run_stage02a(self.cfg,None,only_mode='height',keep_figures_for_show=True)
        self.assertEqual(len(result['active_figures']),1)
        fig=result['active_figures'][0];self.assertTrue(main.plt.fignum_exists(fig.number));main.plt.close(fig)


if __name__=='__main__':unittest.main()
