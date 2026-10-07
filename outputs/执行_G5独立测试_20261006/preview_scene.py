# -*- coding: utf-8 -*-
"""先看完整测试场景，避免仅凭文件名断言场景内容独立。"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'work/datasets/harvard_g5_test_20261006_retry1'

if __name__=='__main__':
    ref=loadmat(FOLDER/'img5.mat',variable_names=['ref'])['ref']
    s=np.loadtxt(FOLDER/'calib.txt').reshape(-1)
    # 这是观察场景身份的伪彩图，显示尺度及gamma不参与测量/指标计算。
    rgb=ref[:,:,[20,12,4]]/s[[20,12,4]]
    scale=np.percentile(rgb,99);rgb=np.clip(rgb/scale,0,1)**(1/2.2)
    image=Image.fromarray(np.round(rgb*255).astype(np.uint8))
    with (Path(__file__).parent/'test_scene_identity.png').open('xb') as stream:image.save(stream,format='PNG')
    print('已保存完整测试场景伪彩预览；仅用于身份检查')
