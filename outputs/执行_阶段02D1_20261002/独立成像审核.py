# -*- coding: utf-8 -*-
"""重读落盘结果，用逐点叠加及逐块内积独立核对前向/伴随。"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]/'jeon2019_optics'
sys.path.insert(0,str(ROOT))
from optics.imaging import SpectralImager


def audit(run):
    report = json.loads((run/'metrics/validation.json').read_text(encoding='utf-8'))
    config = report['config']
    manifest = json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
    source = json.loads((run/'source_evidence.json').read_text(encoding='utf-8'))
    src = Path(source['source_run'])
    actual_src = {p.relative_to(src).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file()}
    assert actual_src == source['sha256']
    expected = {device+'_'+name+'.npz' for device in config['devices'] for name in
                ['coincident_points','separated_points','lines_and_square']}
    assert {p.name for p in (run/'arrays').glob('*.npz')} == expected
    checks = []
    for filename in sorted(expected):
        with np.load(run/'arrays'/filename,allow_pickle=False) as z:
            cube,k = z['cube'],z['kernels']; response = z['response']
            device = filename.split('_coincident')[0].split('_separated')[0].split('_lines')[0]
            assert str(z['fixed_height_fingerprint']) == source['fingerprints'][device]
            for b,lam in enumerate(z['wavelengths_m']):
                with np.load(src/'arrays'/(device+'_'+str(round(float(lam)*1e9))+'nm_q8.npz'),allow_pickle=False) as v:
                    assert np.array_equal(k[b],v['pixel_power']/float(v['Pin']))
            ref = np.zeros_like(z['contributions_full'])
            for b in range(len(cube)):
                for y,x in np.argwhere(cube[b] != 0):
                    ref[b,y:y+k.shape[1],x:x+k.shape[2]] += response[b]*cube[b,y,x]*k[b]
            relative = float(np.linalg.norm(ref-z['contributions_full'])/np.linalg.norm(ref))
            assert relative<1e-12
            assert np.array_equal(z['measurement_full'],z['contributions_full'].sum(axis=0))
            y0,y1,x0,x1 = z['crop_indices']
            assert np.array_equal(z['measurement_crop'],z['measurement_full'][y0:y1,x0:x1])
            pitch = float(z['pitch_m'])
            for axis,start,stop in [('x',x0,x1),('y',y0,y1)]:
                scene_axis=z['scene_'+axis+'_m']; kernel_axis=z['kernel_'+axis+'_m']; full_axis=z['full_'+axis+'_m']
                expected_axis=scene_axis[0]+kernel_axis[0]+np.arange(len(full_axis))*pitch
                assert np.array_equal(full_axis,expected_axis)
                assert np.array_equal(z['output_'+axis+'_m'],full_axis[start:stop])
            rng = np.random.default_rng(7)
            target = rng.standard_normal(z['measurement_crop'].shape)
            extended = np.zeros(z['measurement_full'].shape); extended[y0:y1,x0:x1] = target
            # 每个场景点的贡献核与残差块内积，独立构造数学转置，不调用FFT相关。
            transpose = np.empty(cube.shape)
            for b in range(len(cube)):
                for y in range(cube.shape[1]):
                    for x in range(cube.shape[2]):
                        transpose[b,y,x] = response[b]*np.sum(extended[y:y+k.shape[1],x:x+k.shape[2]]*k[b])
            op=SpectralImager(k,list(cube.shape[1:]),pitch,response,[int(y0),int(y1),int(x0),int(x1)])
            adjoint_relative = float(np.linalg.norm(op.adjoint(target)-transpose)/np.linalg.norm(transpose))
            assert adjoint_relative<1e-12
            flux_expected = float(np.dot(cube.sum(axis=(1,2)),response*k.sum(axis=(1,2))))
            flux_error = abs(float(z['measurement_full'].sum())-flux_expected)/flux_expected
            assert flux_error<1e-12
            checks.append(dict(file=filename,forward_relative_error=relative,
                               direct_transpose_relative_error=adjoint_relative,flux_relative_error=flux_error))
    result=dict(accepted=True,run_dir=str(run),source_run_sha_unchanged=True,
                source_manifest_matches=True,checks=checks)
    (Path(__file__).resolve().parent/'独立成像审核.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': audit(Path(sys.argv[1]))
