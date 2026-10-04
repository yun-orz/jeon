# -*- coding: utf-8 -*-
"""重读正式恢复结果，直接逐块相关核对目标、约束梯度和状态。"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]/'jeon2019_optics'


def transpose_direct(residual,kernels,response,shape,crop):
    extended=np.zeros((shape[1]+kernels.shape[1]-1,shape[2]+kernels.shape[2]-1))
    y0,y1,x0,x1=crop;extended[y0:y1,x0:x1]=residual
    output=np.empty(shape)
    for b in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                output[b,y,x]=response[b]*np.sum(extended[y:y+kernels.shape[1],x:x+kernels.shape[2]]*kernels[b])
    return output


def audit(run):
    r=json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    m=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==s for f,s in m.items())
    e=json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    for path,key in [(e['source_run'],'sha256'),(e['optical_source_run'],'optical_sha256')]:
        base=Path(path)
        sha={p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file()}
        assert sha==e[key]
    records=[]
    assert {p.stem for p in (run/'arrays').glob('*.npz')}=={rec['identity'] for rec in r['records']}
    for rec in r['records']:
        ident=rec['identity']
        trajectory=json.loads((run/'metrics'/(ident+'_optimization.json')).read_text(encoding='utf-8'))
        h=trajectory['history'];assert len(h)==rec['iterations']
        with (run/'metrics'/(ident+'_optimization.csv')).open(encoding='utf-8-sig',newline='') as stream:assert len(list(csv.DictReader(stream)))==len(h)
        assert all(b['objective']<=a['objective']+1e-13*max(a['objective'],1e-300) for a,b in zip(h,h[1:]))
        with np.load(run/'arrays'/(ident+'.npz'),allow_pickle=False) as z:
            x=z['reconstruction'];truth=z['truth'];assert np.all(x>=0) and np.all(np.isfinite(x))
            # 逐点平移核独立重算拟合；不复用生产FFT前向。
            kernels=z['kernels'];response=z['response'];crop=z['crop_indices']
            full=np.zeros((x.shape[1]+kernels.shape[1]-1,x.shape[2]+kernels.shape[2]-1))
            for b in range(x.shape[0]):
                for iy,ix in np.argwhere(x[b]!=0):
                    full[iy:iy+kernels.shape[1],ix:ix+kernels.shape[2]]+=response[b]*x[b,iy,ix]*kernels[b]
            y0,y1,x0,x1=crop
            direct_prediction=full[y0:y1,x0:x1]
            assert np.linalg.norm(direct_prediction-z['prediction'])/np.linalg.norm(z['prediction'])<1e-12
            residual=z['prediction']-z['measurement'];assert np.array_equal(residual,z['residual'])
            alpha=float(z['alpha']);objective=.5*np.sum(residual**2)+.5*alpha*np.sum(x**2)
            assert np.isclose(objective,h[-1]['objective'],rtol=1e-13,atol=0)
            gradient=transpose_direct(residual,z['kernels'],z['response'],x.shape,z['crop_indices'])+alpha*x
            aty=transpose_direct(z['measurement'],z['kernels'],z['response'],x.shape,z['crop_indices'])
            L=h[-1]['L'];mapping=L*(x-np.maximum(0,x-gradient/L))
            pg=float(np.sqrt(np.sum(mapping**2))/max(np.sqrt(np.sum(aty**2)),1e-300))
            assert np.isclose(pg,h[-1]['projected_gradient_relative'],rtol=1e-10,atol=1e-14)
            cube_error=float(np.linalg.norm(x-truth)/np.linalg.norm(truth))
            assert np.isclose(cube_error,rec['evaluation']['cube_relative_l2'],rtol=1e-13)
            if rec['solver_status']=='converged':
                assert pg<=r['config']['solver']['gradient_tolerance'] and h[-1]['objective_relative_change']<=r['config']['solver']['objective_tolerance']
            else:assert rec['solver_status']=='iteration_limit' and len(h)==r['config']['solver']['max_iterations']
            records.append(dict(identity=ident,objective=objective,projected_gradient_relative=pg,
                                solver_status=rec['solver_status'],cube_relative_l2=cube_error))
    result=dict(implementation_audit_passed=True,all_reconstructions_converged=r['all_reconstructions_converged'],
                run_dir=str(run),source_unchanged=True,records=records)
    (Path(__file__).resolve().parent/'独立重建审核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':audit(Path(sys.argv[1]))
