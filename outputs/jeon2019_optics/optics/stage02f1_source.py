# -*- coding: utf-8 -*-
"""逐数组与统计量核验02F-1来源，不仅相信通过标记。"""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from .stage02c_source import tree_sha
from .stage02_runtime import resolve_project_path
from .stage02e4_source import load_noise_source
from .imaging import SpectralImager
from .electron_noise import electron_response,sanitize_poisson_mean,sample_electrons,noise_statistics

def load_electron_source(source,root):
    source,root=Path(source).resolve(),Path(root).resolve();before=tree_sha(source)
    r=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if r.get('status')!='completed' or r.get('validation_passed') is not True or r.get('budget_exhausted') or (source/'failed.json').exists():raise ValueError('F1来源未完整通过')
    manifest=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02f1.py','optics/electron_noise.py','optics/stage02e4_source.py'}<=set(manifest):raise ValueError('F1源码指纹缺失')
    for name,sha in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('F1源码不相容')
    c=r['config'];d=c['detector'];s=c['statistics'];items,evidence=load_noise_source(resolve_project_path(c['source_run'],root),root)
    if json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))['sources']!=evidence['sources']:raise ValueError('F1递归来源改变')
    controls={}
    for name,mu,sigma in [('zero',np.zeros((32,32)),0.),('read_only',np.zeros((32,32)),d['read_sigma_e']),('shot_only',np.full((32,32),17.),0.)]:
        controls[name]=noise_statistics(mu,sigma,c['seed'],'control_'+name,s['repeats'],s['block_size'],s['z_tolerance'])
    if controls!=r['controls'] or controls!=json.loads((source/'metrics/controls.json').read_text(encoding='utf-8')) or not all(v['passed'] for v in controls.values()):raise ValueError('F1统计控制不一致')
    expected={i['device']+'_'+i['scene']+'_G_'+format(g,'.17g') for i in items for g in d['gains_e_per_relative_power'] if i['device'] in c['devices'] and i['scene'] in c['scenes']}
    records={a['identity']:a for a in r['records']}
    if len(expected)!=18 or set(records)!=expected or len(records)!=len(r['records']) or r['planned_jobs']!=r['completed_jobs'] or r['completed_jobs']!=18 or {p.stem for p in (source/'arrays').glob('*.npz')}!=expected:raise ValueError('需要F1完整18组来源')
    selected=[]
    for item in items:
        old=item['values'];op=SpectralImager(old['kernels'],list(old['truth'].shape[1:]),float(old['pitch_m']),old['response'],old['crop_indices'].tolist())
        bands=op.per_band_full(old['truth'])[:,op.slices[0],op.slices[1]]
        response,energy=electron_response(old['wavelengths_m'],d['quantum_efficiency'],d['reference_wavelength_m'])
        for gain in d['gains_e_per_relative_power']:
            ident=item['device']+'_'+item['scene']+'_G_'+format(gain,'.17g')
            if ident not in expected:continue
            rec=records[ident]
            with np.load(source/'arrays'/(ident+'.npz'),allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
            for key in ['truth','kernels','response','wavelengths_m','pitch_m','fixed_height_fingerprint','crop_indices','output_x_m','output_y_m','scene_x_m','scene_y_m']:
                if key not in v or not np.array_equal(v[key],old[key]):raise ValueError('F1固定物理数据改变')
            signal=gain*np.einsum('b,byx->yx',response,bands);raw=signal+d['background_e'];mean,repair=sanitize_poisson_mean(raw,c['roundoff_relative'])
            sample=sample_electrons(mean,d['read_sigma_e'],c['seed'],ident)
            required=dict(per_band_relative_power=bands,mean_signal_raw_e=signal,mean_raw_e=raw,mean_sampling_e=mean,theoretical_variance_e2=mean+d['read_sigma_e']**2,electron_response=response,quantum_efficiency=np.asarray(d['quantum_efficiency']),photon_energy_J=energy,gain_e_per_relative_power=np.asarray(gain),read_sigma_e=np.asarray(d['read_sigma_e']),background_e=np.asarray(d['background_e']),reference_wavelength_m=np.asarray(d['reference_wavelength_m']),seed=np.asarray(c['seed']),identity=np.asarray(ident),measurement_unit=np.asarray('electron'),**sample)
            if any(k not in v or not np.array_equal(v[k],a) for k,a in required.items()) or rec['roundoff_repair']!=repair or rec['device']!=item['device'] or rec['scene']!=item['scene'] or rec['gain_e_per_relative_power']!=gain:raise ValueError('F1电子数/随机流/单位错误')
            stats=noise_statistics(mean,d['read_sigma_e'],c['seed'],ident,s['repeats'],s['block_size'],s['z_tolerance'])
            if stats!=rec['statistics'] or stats!=json.loads((source/'metrics'/(ident+'_statistics.json')).read_text(encoding='utf-8')) or not stats['passed']:raise ValueError('F1充分统计量不一致')
            selected.append(dict(identity=ident,device=item['device'],scene=item['scene'],values=v))
    with (source/'metrics/statistics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=108 or len({(a['identity'],a['check']) for a in rows})!=108:raise ValueError('F1统计CSV身份错误')
    for row in rows:
        rec=records[row['identity']];check=rec['statistics']['checks'][row['check']]
        if float(row['gain'])!=rec['gain_e_per_relative_power'] or float(row['z'])!=check['z'] or row['passed']!='True':raise ValueError('F1统计CSV不相容')
    if tree_sha(source)!=before:raise RuntimeError('读取期间F1来源改变')
    return selected,dict(sources=[dict(path=str(source),sha256=before)]+evidence['sources'])
