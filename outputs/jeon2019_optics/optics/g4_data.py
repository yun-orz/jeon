# -*- coding: utf-8 -*-
"""Harvard真实HSI读取、灵敏度校正、相对光子换算及场景级划分。"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat


def file_sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def photon_relative(raw,calibration,wavelengths_nm):
    """除以相机相对灵敏度，λ/540将相对能量转成相对光子数。

    只确定相对比例，绝对曝光/标定常数未知；不是绝对光子计数。
    """
    raw=np.asarray(raw,dtype=np.float64);s=np.asarray(calibration,dtype=np.float64);wave=np.asarray(wavelengths_nm,dtype=np.float64)
    if raw.ndim<1 or raw.shape[-1]!=len(s) or s.shape!=wave.shape or not np.all(np.isfinite(s)) or np.any(s<=0):raise ValueError('波段或灵敏度不匹配')
    if not np.all(np.isfinite(raw)) or np.any(raw<0) or not np.all(np.isfinite(wave)) or np.any(wave<=0):raise ValueError('光谱数值非法')
    return raw/s*wave/540.


def valid_tile_origins(mask,size):
    """非重叠完整有效上下文块；不填补运动或无效像元。"""
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or type(size) is not int or size<1:raise ValueError('掩码或块尺寸非法')
    return [(y,x) for y in range(0,mask.shape[0]-size+1,size) for x in range(0,mask.shape[1]-size+1,size) if mask[y:y+size,x:x+size].all()]


def prepare_subset(folder,c,seed):
    folder=Path(folder)
    if c['train_scene']==c['validation_scene']:raise ValueError('训练与验证必须来自不同场景文件')
    manifest=json.loads((folder/'download_manifest.json').read_text(encoding='utf-8'))
    calibration_record=json.loads((folder/'calibration_download_manifest.json').read_text(encoding='utf-8'))
    records={r['local_name']:r for r in manifest['files']}
    calibration_path=folder/'calib.txt'
    if file_sha(calibration_path)!=calibration_record['sha256']:raise ValueError('校正文件SHA改变')
    sensitivity=np.loadtxt(calibration_path).reshape(-1)
    if sensitivity.shape!=(31,) or not np.all(np.isfinite(sensitivity)) or np.any(sensitivity<=0):raise ValueError('官方校正须为31个有限正值')
    size=c['core_size']+2*c['halo'];waves=np.arange(420,661,10)
    rng=np.random.default_rng(seed);splits={};metadata={}
    for split,name,count in [('train',c['train_scene'],c['train_patches']),('validation',c['validation_scene'],c['validation_patches'])]:
        if Path(name).name!=name or name not in records:raise ValueError('场景不在官方下载清单中')
        path=folder/name;sha=file_sha(path)
        if sha!=records[name]['sha256']:raise ValueError('MAT文件SHA改变')
        data=loadmat(path,variable_names=['ref','lbl']);raw=data['ref'];labels=data['lbl']
        if raw.ndim!=3 or raw.shape[2]!=31 or labels.shape!=raw.shape[:2]:raise ValueError('Harvard变量ref/lbl尺寸不符')
        valid=(labels!=0)&np.isfinite(raw[:,:,:25]).all(axis=2)&(raw[:,:,:25]>=0).all(axis=2)
        origins=valid_tile_origins(valid,size)
        if len(origins)<count:raise ValueError('完整有效、非重叠的上下文块不足')
        selected=[origins[i] for i in rng.choice(len(origins),size=count,replace=False)]
        patches=np.stack([photon_relative(raw[y:y+size,x:x+size,:25],sensitivity[:25],waves) for y,x in selected])
        splits[split]=patches
        metadata[split]=dict(scene=name,sha256=sha,shape=list(raw.shape),eligible_tiles=len(origins),origins_yx=[list(t) for t in selected],
            patches=count,all_context_pixels_valid=True,raw_min=float(np.min(raw)),raw_max=float(np.max(raw)))
        del raw,labels,data,valid
    if metadata['train']['sha256']==metadata['validation']['sha256']:raise ValueError('训练与验证文件内容重复')
    # 只用训练块拟合一个共同尺度；不做逐波段/逐场景峰值归一化。
    scale=float(np.percentile(splits['train'],c['train_percentile']))
    if not np.isfinite(scale) or scale<=0:raise ValueError('训练归一化尺度非法')
    result={key:(value/scale).astype(np.float32) for key,value in splits.items()}
    metadata.update(normalization_scale=scale,normalization_fit='train_context_only',normalization_percentile=c['train_percentile'],
        wavelengths_nm=waves.tolist(),sensitivity=sensitivity.tolist(),sensitivity_sha256=calibration_record['sha256'],
        photon_conversion='(ref / relative_sensitivity) * wavelength_nm / 540; absolute factor unknown',
        scene_split_disjoint=True,core_size=c['core_size'],halo=c['halo'],context_size=size,
        license_page=manifest['license_page'],dataset_url=manifest['url'],absolute_photon_calibration=False)
    return result,metadata
