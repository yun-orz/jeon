# -*- coding: utf-8 -*-
"""只读重验已冻结02D-1场景，真实cube仅用于来源验证和后续评价。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .imaging import SpectralImager
from .stage02c_source import load_source,tree_sha
from .stage02_runtime import resolve_project_path


def load_imaging_source(source,root):
    source,root=Path(source),Path(root)
    report=json.loads((source/'metrics/validation.json').read_text(encoding='utf-8'))
    if report.get('status')!='completed' or report.get('validation_passed') is not True:
        raise ValueError('02D-1源run未通过成像/伴随验证')
    frozen=json.loads((source/'source_manifest.json').read_text(encoding='utf-8'))
    if not {'main_stage02d1.py','optics/imaging.py','optics/stage02c_source.py'}<=set(frozen):
        raise ValueError('源成像代码指纹缺失')
    for name,sha in frozen.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError('源成像代码不兼容：'+name)
    before=tree_sha(source);cfg=report['config']
    optical_source=resolve_project_path(cfg['source_run'],root)
    kernels,optical_evidence=load_source(optical_source,root,cfg)
    old_evidence=json.loads((source/'source_evidence.json').read_text(encoding='utf-8'))
    if optical_evidence['sha256']!=old_evidence['sha256']:
        raise ValueError('02C光学源run与成像时的来源不一致')
    from main_stage02d1 import scenes
    patterns=scenes(cfg['scene_shape']);expected={device+'_'+label+'.npz' for device in cfg['devices'] for label in patterns}
    if {p.name for p in (source/'arrays').glob('*.npz')}!=expected: raise ValueError('源场景数组身份集合不一致')
    items=[]
    for device in cfg['devices']:
        op=SpectralImager(kernels[device],cfg['scene_shape'],cfg['pitch_m'],cfg['response'],cfg['crop'])
        for label,truth in patterns.items():
            filename=device+'_'+label+'.npz'
            with np.load(source/'arrays'/filename,allow_pickle=False) as z:
                if not (np.array_equal(z['cube'],truth) and np.array_equal(z['kernels'],kernels[device])
                        and np.array_equal(z['response'],cfg['response'])
                        and np.array_equal(z['wavelengths_m'],cfg['wavelengths_m'])
                        and np.array_equal(z['crop_indices'],cfg['crop']) and float(z['pitch_m'])==cfg['pitch_m']
                        and str(z['fixed_height_fingerprint'])==optical_evidence['fingerprints'][device]):
                    raise ValueError('源场景/核/元数据不一致：'+filename)
                if not all(np.array_equal(z[key],value) for key,value in op.coordinates().items()):
                    raise ValueError('源场景物理坐标不一致')
                bands=op.per_band_full(truth);full=bands.sum(axis=0)
                if not (np.array_equal(z['contributions_full'],bands) and np.array_equal(z['measurement_full'],full)
                        and np.array_equal(z['measurement_crop'],full[op.slices])):
                    raise ValueError('源测量与固定核前向模型不一致')
                items.append(dict(device=device,scene=label,values={key:z[key].copy() for key in z.files}))
    if tree_sha(source)!=before: raise RuntimeError('源场景run在载入期间变化')
    return items,dict(source_run=str(source.resolve()),sha256=before,config=cfg,
        optical_source_run=str(optical_source),optical_sha256=optical_evidence['sha256'],source_unchanged=True)
