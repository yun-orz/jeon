# -*- coding: utf-8 -*-
"""独立审核证据：只读取送审产物；反例写入新目录，不修改生产代码。"""
from pathlib import Path
import hashlib
import json
import logging
import shutil
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'jeon2019_optics'
RUN = PROJECT / 'results/stage02b1/run_20261001_224235'
sys.path.insert(0, str(PROJECT))
import main_stage02b as M
from optics.coordinates import make_grid

cfg = M.load_and_validate_config(PROJECT / 'config_stage02b.json')
gi = make_grid(1101, 1e-6)
go = make_grid(301, 1e-6)
profiles = {
    'jeon': M.design_jeon2019_spiral_height(gi, 1e-3, .05, 3, 420e-9, 660e-9),
    'fresnel': M.design_conventional_fresnel_height(gi, 1e-3, .05, 550e-9),
}
fps = {k: v.compute_fingerprint() for k, v in profiles.items()}
metrics = json.loads((RUN / 'metrics/psf_metrics.json').read_text('utf-8'))
X, Y = go.meshgrid()
r = np.hypot(X, Y)
disk = r <= 150e-6
rows = []
for key in M.DEVICE_KEYS:
    for lam in cfg['optical']['incident_wavelengths_m']:
        nm = round(lam * 1e9)
        ident = f'{key}@{nm}nm'
        with np.load(RUN / 'arrays' / M.psf_array_name(key, lam)) as a:
            I, u = a['intensity_raw'], a['u2_complex']
            u1 = M.compute_doe_transmission_field(profiles[key], lam)
            pin = float(np.sum(abs(u1)**2) * gi.cell_area)
            pw = float(I.sum() * go.cell_area)
            order = np.argsort(r[disk])
            radii = r[disk][order]
            energy = np.cumsum(I[disk][order]) * go.cell_area / pin
            rec = {'identity': ident, 'Pin': pin, 'Pwindow': pw,
                   'eta': pw / pin, 'Eabs_150um': float(I[disk].sum()*go.cell_area/pin),
                   'power_metadata_matches': bool(np.isclose(pin,float(a['Pin']),rtol=1e-13)
                                                 and np.isclose(pw,float(a['Pwindow']),rtol=1e-13)),
                   'I_abs2_exact': bool(np.array_equal(I,abs(u)**2)),
                   'dtype': [str(u.dtype),str(I.dtype)]}
            for q in (.5, .8):
                hits = np.flatnonzero(energy >= q)
                field = 'R%d' % round(q*100)
                rec[field+'_correct_um'] = float(radii[hits[0]]*1e6) if hits.size else None
                rec[field+'_reported_um'] = metrics['records'][ident][field]['radius_um']
            for label, low, high in [('primary',20e-6,100e-6),('sensitivity',30e-6,120e-6)]:
                b = (r>=low)&(r<=high)
                c = np.sum(I[b]*np.exp(3j*np.arctan2(Y[b],X[b]))) * go.cell_area
                rec[label+'_angle_deg'] = float(np.angle(c)*180/np.pi/3)
                rec[label+'_A3'] = float(abs(c)/(I[b].sum()*go.cell_area))
            rows.append(rec)

# 仅在新目录复制产物，注入功率元数据错误，确认重读能否拒绝。
probe = ROOT / ('forged_power_' + uuid.uuid4().hex[:8])
(probe/'arrays').mkdir(parents=True)
for f in (RUN/'arrays').glob('*.npz'):
    shutil.copy2(f, probe/'arrays'/f.name)
f = probe/'arrays/psf_jeon_540nm.npz'
with np.load(f) as a:
    payload = {k:a[k] for k in a.files}
payload['Pin'] = payload['Pin'] * 2
payload['Pwindow'] = payload['Pwindow'] * 2
np.savez_compressed(f, **payload)
forged = M.verify_saved_identity_set(probe,cfg,gi,go,fps,profiles)

# 验证配置z与实际传播调用是否相同，不再进行重传播。
captured = {}
original = M.fresnel_kernel_separable
def capture(**kw):
    captured['distance_m'] = kw['distance']
    return np.ones(go.shape,dtype=np.complex128)
M.fresnel_kernel_separable = capture
M.propagate_device(profiles['jeon'],540e-9,gi,go,True,1.,0.,logging.getLogger('audit'),'audit')
M.fresnel_kernel_separable = original

manifest = json.loads((RUN/'source_manifest.json').read_text('utf-8'))
fingerprint_mismatch = [k for k,v in manifest['files'].items()
                        if hashlib.sha256((PROJECT/k).read_bytes()).hexdigest()!=v['sha256']]
result = {'reviewed_run':str(RUN),'source_mismatch':fingerprint_mismatch,
          'profiles':fps,'rows':rows,'forged_power_verification':forged,
          'forged_power_path':str(probe),'propagation_distance_capture':captured,
          'limitations':'距离捕获为调用检查；默认f=z时不影响原18组结果。'}
(ROOT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print('原始18组功率元数据与模平方:',all(v['power_metadata_matches'] and v['I_abs2_exact'] for v in rows))
print('源码指纹差异:',fingerprint_mismatch)
print('伪造Pin/Pwindow同时乘2后验收:',forged['status'])
for v in rows:
    if v['R80_correct_um'] != v['R80_reported_um']:
        print(v['identity'],'R80 正确/报告:',v['R80_correct_um'],v['R80_reported_um'])
