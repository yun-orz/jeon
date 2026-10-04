# -*- coding: utf-8 -*-
"""项目外执行入口，逐行记录，无删除操作。"""
from pathlib import Path
import sys,subprocess,os
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(r'D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics');EV=Path(__file__).resolve().parent
kind=sys.argv[1]
if kind=='新增检查':args=['-m','unittest','discover','-s','tests','-p','test_stage02e3.py','-v'];cwd=ROOT
elif kind=='回归':args=['-m','unittest','discover','-s','tests','-v'];cwd=ROOT
else:args=[str(ROOT/'main_stage02e3.py')];cwd=EV
p=subprocess.Popen([sys.executable,'-B']+args,cwd=cwd,env=dict(os.environ,PYTHONIOENCODING='utf-8'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
with (EV/(kind+'.log')).open('w',encoding='utf-8') as f:
    for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
p.wait();print('exit',p.returncode);raise SystemExit(p.returncode)
