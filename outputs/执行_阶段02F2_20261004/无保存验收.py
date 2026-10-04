# -*- coding: utf-8 -*-
import copy,json,os,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'jeon2019_optics';sys.path.insert(0,str(ROOT))
from optics.stage02c_source import tree_sha
c=json.loads((ROOT/'config_stage02f2.json').read_text(encoding='utf-8'));c['devices']=['continuous'];c['scenes']=['coincident_points'];c['methods']=['unweighted']
p=HERE/'异地无保存配置.json';p.write_text(json.dumps(c),encoding='utf-8')
before=tree_sha(ROOT)
r=subprocess.run([sys.executable,'-B',str(ROOT/'main_stage02f2.py'),'--config',str(p),'--no-save'],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
(HERE/'异地无保存.log').write_bytes(r.stdout)
after=tree_sha(ROOT)
result=dict(passed=r.returncode==0 and after==before,exit_code=r.returncode,hashed_files=len(before),file_set_unchanged=set(before)==set(after),sha_unchanged=after==before,alternate_cwd=str(HERE))
(HERE/'无保存验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False));assert result['passed']
