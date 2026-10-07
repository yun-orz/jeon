# -*- coding: utf-8 -*-
"""独立测试场景：复用训练尺度，不在测试数据上拟合任何参数。"""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from optics.g4_data import file_sha,photon_relative,valid_tile_origins


def select_test(raw,labels,sensitivity,scale,size,count,seed):
    """先按掩码和预定种子选完整上下文，再执行固定谱换算。"""
    raw=np.asarray(raw);labels=np.asarray(labels);sensitivity=np.asarray(sensitivity)
    if raw.ndim!=3 or raw.shape[-1]!=31 or labels.shape!=raw.shape[:2] or sensitivity.shape!=(31,):
        raise ValueError('Harvard测试场景变量或校正尺寸错误')
    if isinstance(scale,bool) or not np.isfinite(scale) or scale<=0:raise ValueError('必须提供冻结的正训练尺度')
    if type(count) is not int or not 1<=count<=8:raise ValueError('测试块数量须为1至8')
    if type(seed) is not int or seed<0:raise ValueError('测试选择种子非法')
    valid=(labels!=0)&np.isfinite(raw[:,:,:25]).all(axis=-1)&(raw[:,:,:25]>=0).all(axis=-1)
    origins=valid_tile_origins(valid,size)
    if len(origins)<count:raise ValueError('测试完整有效上下文块不足')
    chosen=[origins[i] for i in np.random.default_rng(seed).choice(len(origins),size=count,replace=False)]
    patches=np.stack([photon_relative(raw[y:y+size,x:x+size,:25],sensitivity[:25],np.arange(420,661,10))/scale for y,x in chosen]).astype(np.float32)
    if not np.isfinite(patches).all():raise ValueError('测试归一化结果非有限')
    return patches,chosen,len(origins)


def prepare_test(folder,training,c):
    folder=Path(folder);manifest=json.loads((folder/'download_manifest.json').read_text(encoding='utf-8'))
    records=manifest['files']
    if len(records)!=1:raise ValueError('本批只接受一幅预定独立测试场景')
    record=records[0];name=record['local_name']
    if Path(name).name!=name or not name.endswith('.mat'):raise ValueError('测试文件名非法')
    sha=file_sha(folder/name)
    if sha!=record['sha256']:raise ValueError('测试MAT与下载SHA不一致')
    if name in [training['train']['scene'],training['validation']['scene']] or sha in [training['train']['sha256'],training['validation']['sha256']]:
        raise ValueError('测试与训练/验证文件重复')
    if file_sha(folder/'calib.txt')!=training['sensitivity_sha256']:raise ValueError('测试校正与训练不同')
    sensitivity=np.loadtxt(folder/'calib.txt').reshape(-1)
    if not np.array_equal(sensitivity,np.array(training['sensitivity'])):raise ValueError('校正数值改变')
    data=loadmat(folder/name,variable_names=['ref','lbl']);size=training['context_size']
    patches,origins,eligible=select_test(data['ref'],data['lbl'],sensitivity,training['normalization_scale'],size,c['patches'],c['seed'])
    metadata=dict(scene=name,sha256=sha,shape=list(data['ref'].shape),origins_yx=[list(p) for p in origins],
        eligible_tiles=eligible,patches=len(patches),context_size=size,core_size=training['core_size'],halo=training['halo'],
        normalization_scale=training['normalization_scale'],normalization_fit='G4 train_context_only; no test fitting',
        sensitivity_sha256=training['sensitivity_sha256'],selection_seed=c['seed'],scene_file_split_disjoint=True,
        spatial_scope='four nonoverlapping contexts from ONE held-out capture; not whole-image quality or four independent scenes',
        dataset_url=manifest['url'],license_page=manifest['license_page'],absolute_photon_calibration=False)
    return patches,metadata
