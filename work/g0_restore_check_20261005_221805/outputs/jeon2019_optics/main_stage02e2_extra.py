# -*- coding: utf-8 -*-
"""02E-2一次追加诊断：只复核线条弱α，PyCharm无参数CPU入口。"""
import argparse,json
import main_stage02e2 as core
from optics.stage02_runtime import resolve_project_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config_stage02e2_extra.json')
    p.add_argument('--no-save',action='store_true');p.add_argument('--show-plots',action='store_true')
    a=p.parse_args();c=json.loads(resolve_project_path(a.config,core.ROOT).read_text(encoding='utf-8'))
    r=core.run(c,a.no_save,a.show_plots)
    return 0 if r['stability_validation_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
