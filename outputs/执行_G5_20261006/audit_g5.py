# -*- coding: utf-8 -*-
"""独立G5审核；只读原来源，新增审核记录，不覆盖或删除文件。"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'outputs/jeon2019_optics'
sys.path.insert(0,str(ENGINE));sys.path.insert(0,str(ENGINE/'tests'))
from main_g1 import write_json
from optics.stage02c_source import tree_sha
from optics.g4_data import file_sha
from test_g5 import reference_ssim


def main():
    run=ENGINE/'results/g5/run_20261006_080431'
    saved=json.loads((run/'metrics/scores.json').read_text(encoding='utf-8'))
    errors=[];psnr_errors=[];angle_errors=[]
    with np.load(run/'arrays/g5_evaluation.npz',allow_pickle=False) as z:
        targets=z['targets'].copy();predictions=z['predictions'].copy();initial=z['untrained_predictions'].copy()
    for label,values in [('best_epoch',predictions),('untrained',initial)]:
        for i,(a,b) in enumerate(zip(targets,values)):
            a=a.astype(float);b=b.astype(float);row=saved[label][i]
            direct=[reference_ssim(a[:,:,k],b[:,:,k]).mean() for k in range(25)]
            errors.append(float(np.max(np.abs(np.array(direct)-row['ssim_per_band']))))
            mse=float(np.sum((a-b)**2)/a.size);psnr=20*np.log10(1./np.sqrt(mse))
            psnr_errors.append(abs(float(psnr)-row['psnr_cube_db']))
            # 以归一化向量距离的atan2表达光谱角，与arccos实现交叉核对。
            u=a/np.sqrt((a*a).sum(axis=-1,keepdims=True));v=b/np.sqrt((b*b).sum(axis=-1,keepdims=True))
            angles=2*np.arctan2(np.linalg.norm(u-v,axis=-1),np.linalg.norm(u+v,axis=-1))
            angle_errors.append(abs(float(angles.mean())-row['sam_mean_rad']))
    before=tree_sha(ENGINE)
    child=subprocess.run([sys.executable,'-B',str(ENGINE/'main_g5.py'),'--no-save'],cwd=str(ROOT/'work'),capture_output=True,text=True,encoding='utf-8')
    after=tree_sha(ENGINE)
    report=dict(passed=max(errors)<1e-11 and max(psnr_errors)<1e-12 and max(angle_errors)<1e-12 and before==after and child.returncode==0,
        independent_ssim_max_error=max(errors),independent_psnr_max_error_db=max(psnr_errors),independent_sam_max_error_rad=max(angle_errors),
        no_save_returncode=child.returncode,no_save_sha_unchanged=before==after,files_checked=len(before),
        foreign_working_directory=str(ROOT/'work'),stdout=child.stdout,stderr=child.stderr)
    audit_path=Path(__file__).with_name('independent_audit.json');number=1
    while audit_path.exists():
        audit_path=Path(__file__).with_name(f'independent_audit_repeat{number}.json');number+=1
    write_json(audit_path,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:raise RuntimeError('独立审核未通过')
    # 先审核源文件和产物哈希，再记录审核结论；状态索引不作为冻结源。
    names=['main_g5.py','config_g5.json','optics/g5_metrics.py','tests/test_g5.py','G5_README.md']
    write_json(Path(__file__).with_name('最终审核.json'),dict(status='validation_metrics_diagnostic_passed',official_run=str(run),
        independent_audit=report,tests_run=6,paper_alignment_passed=False,independent_test_set=False,
        sha256={str(ENGINE/name):file_sha(ENGINE/name) for name in names},run_sha256=tree_sha(run)))


if __name__=='__main__':main()
