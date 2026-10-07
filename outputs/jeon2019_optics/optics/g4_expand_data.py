# -*- coding: utf-8 -*-
"""扩充原训练/验证场景；保留旧块及冻结尺度，不读取测试数据。"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from optics.g4_data import file_sha,photon_relative,valid_tile_origins


def extend_origins(eligible,original,count,seed):
    """原块在前，新增块不重复；选择与网络输出和目标亮度无关。"""
    original=[tuple(p) for p in original];eligible=[tuple(p) for p in eligible]
    if type(count) is not int or not len(original)<=count<=len(eligible):raise ValueError('扩充块数不足或越界')
    if len(set(original))!=len(original) or any(p not in eligible for p in original):raise ValueError('旧块重复或不在有效网格')
    available=[p for p in eligible if p not in original]
    chosen=np.random.default_rng(seed).choice(len(available),size=count-len(original),replace=False)
    return original+[available[i] for i in chosen]


def prepare_expanded(folder,old,c,seed):
    folder=Path(folder);size=old['context_size'];scale=old['normalization_scale']
    if c['core_size']!=old['core_size'] or c['halo']!=old['halo']:raise ValueError('本批保持旧监督区域与物理边界')
    if not np.isfinite(scale) or scale<=0:raise ValueError('冻结训练尺度非法')
    if file_sha(folder/'calib.txt')!=old['sensitivity_sha256']:raise ValueError('校正来源改变')
    sensitivity=np.loadtxt(folder/'calib.txt').reshape(-1)
    if not np.array_equal(sensitivity,np.array(old['sensitivity'])):raise ValueError('校正数值改变')
    subsets={};metadata={}
    for offset,split in enumerate(['train','validation']):
        row=old[split];name=row['scene']
        if Path(name).name!=name or name not in ['img3.mat','img4.mat']:raise ValueError('本批只读取既有训练/验证场景')
        if file_sha(folder/name)!=row['sha256']:raise ValueError('训练/验证MAT改变')
        data=loadmat(folder/name,variable_names=['ref','lbl']);raw=data['ref'];lbl=data['lbl']
        if raw.ndim!=3 or raw.shape[-1]!=31 or lbl.shape!=raw.shape[:2]:raise ValueError('原始变量尺寸错误')
        valid=(lbl!=0)&np.isfinite(raw[:,:,:25]).all(axis=-1)&(raw[:,:,:25]>=0).all(axis=-1)
        eligible=valid_tile_origins(valid,size)
        selected=extend_origins(eligible,row['origins_yx'],c[split+'_patches'],seed+offset)
        subsets[split]=np.stack([photon_relative(raw[y:y+size,x:x+size,:25],sensitivity[:25],np.arange(420,661,10))/scale for y,x in selected]).astype(np.float32)
        if not np.isfinite(subsets[split]).all():raise ValueError('扩充目标非有限')
        metadata[split]=dict(scene=name,sha256=row['sha256'],shape=list(raw.shape),eligible_tiles=len(eligible),
            origins_yx=[list(p) for p in selected],original_patches=len(row['origins_yx']),patches=len(selected))
    metadata.update(normalization_scale=scale,normalization_fit='frozen original G4 train contexts; no refitting',
        sensitivity_sha256=old['sensitivity_sha256'],sensitivity=old['sensitivity'],core_size=old['core_size'],halo=old['halo'],context_size=size,
        scene_split_disjoint=metadata['train']['sha256']!=metadata['validation']['sha256'],test_data_read=False,
        wavelengths_nm=old['wavelengths_nm'],absolute_photon_calibration=False)
    if not metadata['scene_split_disjoint']:raise ValueError('训练验证内容重复')
    return subsets,metadata
