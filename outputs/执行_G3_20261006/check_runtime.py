# -*- coding: utf-8 -*-
"""G3外部目录无保存、无效配置和保存故障验证；不删除证据。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE))
import main_g3

def snapshot():
    result={}
    for path in ENGINE.rglob('*'):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            result[path.relative_to(ENGINE).as_posix()]=h.hexdigest()
    return result

if __name__=='__main__':
    c=json.loads((ENGINE/'config_g3.json').read_text(encoding='utf-8'));folder=ENGINE/'results/g3';before_dirs=set(folder.glob('run_*'))
    bad=json.loads(json.dumps(c));bad['network']['stages']=0
    try:main_g3.run(bad)
    except ValueError:pass
    else:raise AssertionError('无效阶段数未拒绝')
    assert before_dirs==set(folder.glob('run_*'))
    with patch.object(main_g3.torch,'save',side_effect=OSError('审核模拟模型保存失败')):
        try:main_g3.run(c)
        except OSError:pass
        else:raise AssertionError('模型保存故障未传递')
    created=set(folder.glob('run_*'))-before_dirs;assert len(created)==1
    failed=created.pop();assert (failed/'failed.json').is_file() and not (failed/'metrics/validation.json').exists()
    before=snapshot()
    with (HERE/'no_save.log').open('xb') as log:
        process=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g3.py'),'--no-save'],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT,timeout=180)
    after=snapshot();assert process.returncode==0 and before==after
    result=dict(passed=True,invalid_config_before_allocation=True,checkpoint_save_failure_retained=str(failed),no_false_completed=True,
        no_save_returncode=process.returncode,files_checked=len(before),all_file_sha_unchanged=True,external_cwd=str(ROOT))
    with (HERE/'runtime_verification.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
