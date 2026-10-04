# -*- coding: utf-8 -*-
"""从项目外启动并保存完整日志；不删除历史。"""
from pathlib import Path
import sys,subprocess,os
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics')
EV=ROOT.parent/'执行_阶段02E1_20261002'
kind=sys.argv[1]
if kind=='回归':args=['-m','unittest','discover','-s','tests','-v'];cwd=ROOT
else:args=[str(ROOT/'main_stage02e1.py')];cwd=EV
p=subprocess.Popen([sys.executable,'-B']+args,cwd=cwd,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
with (EV/(kind+'.log')).open('w',encoding='utf-8') as f:
    for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
p.wait();print('exit',p.returncode);raise SystemExit(p.returncode)
