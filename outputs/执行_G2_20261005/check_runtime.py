# -*- coding: utf-8 -*-
"""运行错误路径与完整无保存核验，所有故障证据保留。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE))
import main_g2

def snapshot():
    result={}
    for path in ENGINE.rglob('*'):
        if path.is_file():
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            result[path.relative_to(ENGINE).as_posix()]=digest.hexdigest()
    return result

def run_dirs():
    return set((ENGINE/'results/g2').glob('run_*'))

if __name__=='__main__':
    c=json.loads((ENGINE/'config_g2.json').read_text(encoding='utf-8'))
    before_dirs=run_dirs();invalid=json.loads(json.dumps(c));invalid['bin_width_nm']=0.
    try:main_g2.run(invalid)
    except ValueError:pass
    else:raise AssertionError('无效带宽未拒绝')
    assert before_dirs==run_dirs()
    # 模拟NPZ保存失败，检查没有completed标记；不删除产生的失败run。
    with patch.object(main_g2.np,'savez_compressed',side_effect=OSError('G2审核模拟保存失败')):
        try:main_g2.run(c)
        except OSError:pass
        else:raise AssertionError('保存故障未向上传递')
    created=run_dirs()-before_dirs;assert len(created)==1
    failed=created.pop();assert (failed/'failed.json').is_file() and not (failed/'metrics/validation.json').exists()
    before=snapshot()
    with (HERE/'no_save.log').open('xb') as log:
        process=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g2.py'),'--no-save'],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT,timeout=180)
    after=snapshot();assert process.returncode==0 and before==after
    result=dict(passed=True,invalid_config_rejected_before_allocation=True,save_failure_retained=str(failed),
                no_false_completed=True,no_save_returncode=process.returncode,files_checked=len(before),all_file_sha_unchanged=True,
                invocation_cwd=str(ROOT))
    with (HERE/'runtime_verification.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
