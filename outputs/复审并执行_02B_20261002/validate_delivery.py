# -*- coding: utf-8 -*-
"""冻结版本的独立交付验证；源run只读，所有新增日志永久保留。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent/'jeon2019_optics'
sys.path.insert(0,str(PROJECT))
import main_stage02b as B
from optics.coordinates import make_grid
from optics.doe import DOEHeightProfile


def snapshot(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def child(name,entry,args):
    t=time.perf_counter()
    result=subprocess.run([sys.executable,'-B',str(PROJECT/entry)]+args,
                          cwd=PROJECT.parent.parent,env=dict(os.environ,PYTHONIOENCODING='utf-8'),
                          capture_output=True,text=True,encoding='utf-8')
    (HERE/(name+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(name+' failed: '+result.stderr)
    return {'exit_code':result.returncode,'elapsed_s':time.perf_counter()-t,'stdout':result.stdout}


def validate_stage2(run):
    metrics=json.loads((run/'metrics/convergence.json').read_text('utf-8'))
    cfg=json.loads((run/'config_effective.json').read_text('utf-8'))
    manifest=json.loads((run/'source_manifest.json').read_text('utf-8'))
    assert all(hashlib.sha256((PROJECT/f).read_bytes()).hexdigest()==sha for f,sha in manifest.items())
    profiles={}
    for path in (run/'arrays').glob('height_*.npz'):
        with np.load(path) as a:
            g=make_grid(len(a['x_in_m']),float(a['dx_m']))
            assert np.array_equal(g.x.coords,a['x_in_m']) and np.array_equal(g.y.coords,a['y_in_m'])
            prof=DOEHeightProfile(g,a['delta_h_m'],a['mask'],a['lambda_design_m'],str(a['design_type']),json.loads(str(a['params_json'])))
            assert prof.compute_fingerprint()==str(a['fingerprint'])
            profiles[str(a['fingerprint'])]=prof
    field_ids=[]
    for path in (run/'arrays').glob('*.npz'):
        if path.name.startswith('height_'):continue
        with np.load(path) as a:
            u,I=a['u2_complex'],a['intensity_raw']
            g=make_grid(len(a['x_out_m']),float(a['output_spacing_m']))
            assert np.array_equal(g.x.coords,a['x_out_m']) and np.array_equal(g.y.coords,a['y_out_m'])
            assert u.dtype==np.complex128 and I.dtype==np.float64 and np.all(np.isfinite(u))
            assert np.array_equal(I,abs(u)**2)
            lam=float(a['wavelength_m']);prof=profiles[str(a['device_fingerprint'])]
            u1=B.compute_doe_transmission_field(prof,lam)
            pin=float((abs(u1)**2).sum()*prof.grid.cell_area)
            pw=float(I.sum()*g.cell_area)
            assert pin==float(a['Pin']) and pw==float(a['Pwindow'])
            ident='%s_%s_%dnm'%(str(a['case_name']),str(a['device_key']),round(lam*1e9))
            assert ident==path.stem and ident in metrics['metrics']
            field_ids.append(ident)
    assert len(field_ids)==36 and len(set(field_ids))==36
    # 固定1μm基线应与已审核的02B-1场一致。
    reference=PROJECT/'results/stage02b1/run_20261002_163101'
    exact=True
    for key in ('fresnel','jeon'):
        for nm in (420,540,660):
            with np.load(run/'arrays'/('baseline_%s_%dnm.npz'%(key,nm))) as a,np.load(reference/'arrays'/('psf_%s_%dnm.npz'%(key,nm))) as b:
                exact &= np.array_equal(a['u2_complex'],b['u2_complex'])
    assert exact
    return {'fields':len(field_ids),'height_profiles':len(profiles),'source_manifest_matches':True,
            'independent_Pin_Pwindow_exact':True,'baseline_matches_02b1_exact':True}


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    result={'stage02b2_saved':validate_stage2(PROJECT/'results/stage02b2/run_20261002_164825')}
    before=snapshot(PROJECT)
    no_save=child('02B2_no_save','main_stage02b2.py',['--no-save'])
    after=snapshot(PROJECT)
    assert before==after
    result['stage02b2_no_save']={k:v for k,v in no_save.items() if k!='stdout'}
    result['stage02b2_no_save']['project_snapshot_unchanged']=True
    write=HERE/'delivery_validation.json'
    write.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
