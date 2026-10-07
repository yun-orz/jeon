# -*- coding: utf-8 -*-
"""从外部启动目录运行无保存入口，前后核对工程全部文件SHA。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ENGINE=ROOT/'outputs/jeon2019_optics'

def snapshot():
    result={}
    for p in ENGINE.rglob('*'):
        if p.is_file():
            digest=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
            result[p.relative_to(ENGINE).as_posix()]=digest.hexdigest()
    return result

if __name__=='__main__':
    start=time.perf_counter();before=snapshot()
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    with (HERE/'no_save.log').open('xb') as log:
        completed=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g1_alignment.py'),'--no-save'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    after=snapshot();assert completed.returncode==0 and before==after
    with (HERE/'no_save_verification.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(passed=True,returncode=completed.returncode,files_checked=len(before),elapsed_s=time.perf_counter()-start,
                       external_cwd=str(ROOT),all_file_sha_unchanged=True),stream,ensure_ascii=False,indent=2)
    print('无保存全工程核对通过，文件数：',len(before))
