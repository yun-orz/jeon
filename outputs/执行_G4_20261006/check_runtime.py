# -*- coding: utf-8 -*-
"""G4故障路径及从工程外部完整无保存验证；故障run保留。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE))
import main_g4

def snapshot():
    values={}
    for p in ENGINE.rglob('*'):
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            values[p.relative_to(ENGINE).as_posix()]=h.hexdigest()
    return values

if __name__=='__main__':
    c=json.loads((ENGINE/'config_g4.json').read_text(encoding='utf-8'));folder=ENGINE/'results/g4';before_dirs=set(folder.glob('run_*'))
    bad=json.loads(json.dumps(c));bad['dataset']['validation_scene']=bad['dataset']['train_scene']
    try:main_g4.run(bad)
    except ValueError:pass
    else:raise AssertionError('重复场景未拒绝')
    assert before_dirs==set(folder.glob('run_*'))
    # 故障路径用同一真实数据和保存函数、较小模型，避免重复正式训练。
    tiny=json.loads(json.dumps(c));tiny['network'].update(stages=1,features=4,levels=2)
    tiny['training']['epochs']=1;tiny['dataset'].update(train_patches=1,validation_patches=1)
    with patch.object(main_g4.torch,'save',side_effect=OSError('G4审核模拟保存失败')):
        try:main_g4.run(tiny)
        except OSError:pass
        else:raise AssertionError('保存故障未传递')
    created=set(folder.glob('run_*'))-before_dirs;assert len(created)==1
    failed=created.pop();assert (failed/'failed.json').is_file() and not (failed/'metrics/validation.json').exists()
    before=snapshot()
    with (HERE/'no_save.log').open('xb') as log:
        proc=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g4.py'),'--no-save'],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT,timeout=180)
    after=snapshot();assert proc.returncode==0 and before==after
    result=dict(passed=True,duplicate_scene_before_allocation=True,save_failure_retained=str(failed),no_false_completed=True,
        failure_config='smaller_model_same_real_data_save_path',no_save_returncode=proc.returncode,files_checked=len(before),all_file_sha_unchanged=True,
        external_cwd=str(ROOT))
    with (HERE/'runtime_verification.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
