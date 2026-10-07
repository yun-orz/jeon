# -*- coding: utf-8 -*-
"""实跑完整无保存入口和两条失败路径，保留所有证据，不删除文件。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[2]
PROJECT=REPO/'outputs/jeon2019_optics'
DEST=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))


def snapshot():
    rows={}
    for path in sorted(PROJECT.rglob('*')):
        if not path.is_file():continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        rows[path.relative_to(PROJECT).as_posix()]=digest.hexdigest()
    return rows


def main():
    print('开始完整工程文件清单与SHA核验',flush=True)
    before=snapshot();start=time.perf_counter()
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    result=subprocess.run([sys.executable,'-B',str(PROJECT/'main_g1.py'),'--no-save'],cwd=DEST,
                          env=env,capture_output=True,text=True,encoding='utf-8')
    with (DEST/'full_no_save.log').open('x',encoding='utf-8') as f:f.write(result.stdout+'\n'+result.stderr)
    after=snapshot()
    if result.returncode or before!=after:raise RuntimeError('完整无保存失败或修改了工程文件')
    no_save=dict(exit_code=result.returncode,files=len(before),sha_unchanged=True,
                 elapsed_s=time.perf_counter()-start,cwd=str(DEST))
    print('完整无保存通过',no_save,flush=True)
    import main_g1
    c=json.loads((PROJECT/'config_g1.json').read_text(encoding='utf-8'))
    failures={}
    # 注入保存异常发生在高度保存阶段，不消耗75组传播。
    for name in ['budget','height_save_fault']:
        old={p.name for p in (PROJECT/'results/g1').iterdir()}
        config=json.loads(json.dumps(c))
        try:
            if name=='budget':
                config['budget_seconds']=1e-6
                main_g1.run(config)
            else:
                with patch('main_g1.np.savez_compressed',side_effect=OSError('G1保存失败注入')):
                    main_g1.run(config)
        except (TimeoutError,OSError) as exc:
            new=[p for p in (PROJECT/'results/g1').iterdir() if p.name not in old]
            if len(new)!=1:raise RuntimeError('失败路径必须有唯一证据目录')
            path=new[0]
            if not (path/'failed.json').is_file() or (path/'metrics/validation.json').exists():
                raise RuntimeError('失败却产生完成标记')
            failures[name]=dict(rejected=True,run=str(path.relative_to(PROJECT)),error=repr(exc),completion_absent=True)
        else:raise RuntimeError('应该拒绝的失败路径未拒绝')
    out=dict(passed=True,no_save=no_save,failures=failures,real_gui_verified=False)
    with (DEST/'runtime_verification.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
