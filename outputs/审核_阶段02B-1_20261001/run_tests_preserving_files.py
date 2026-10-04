# -*- coding: utf-8 -*-
"""审核测试运行器：拦截新测试的自动删除，保留全部临时证据。"""
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'jeon2019_optics'
TEMP = ROOT / 'retained_test_temp'
TEMP.mkdir(exist_ok=True)
tempfile.tempdir = str(TEMP)
prevented = []
def preserve(path, *args, **kwargs):
    prevented.append(str(path))
shutil.rmtree = preserve
os.chdir(PROJECT)
sys.path.insert(0,str(PROJECT))
sys.stdout.reconfigure(encoding='utf-8')
t = time.perf_counter()
suite = unittest.defaultTestLoader.discover('tests')
with (ROOT/'full_tests_preserved.log').open('w',encoding='utf-8') as f:
    result = unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
summary = {'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
           'elapsed_s':time.perf_counter()-t,'success':result.wasSuccessful(),
           'prevented_deletions':prevented,
           'method':'仅运行器拦截shutil.rmtree并指定保留目录；生产与测试源码未改。'}
(ROOT/'full_tests_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='prevented_deletions'},ensure_ascii=False))
raise SystemExit(0 if result.wasSuccessful() else 1)
