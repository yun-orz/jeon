# -*- coding: utf-8 -*-
"""测量一致性诊断：先在完整上下文编码，再评价中心区域。"""
import numpy as np


def summary_difference(actual,expected):
    """浮点指标容许1e−12绝对舍入差；字段、整数、布尔与null须严格一致。"""
    if set(actual)!=set(expected):raise ValueError('比较指标字段不一致')
    differences=[]
    for key,value in expected.items():
        other=actual[key]
        if type(value) is float:
            if type(other) is not float or not np.isfinite(value) or not np.isfinite(other):raise ValueError('比较指标非有限')
            difference=abs(other-value)
            if difference>1e-12:raise ValueError('旧指标重算超出容差：'+key)
            differences.append(difference)
        elif type(other) is not type(value) or other!=value:raise ValueError('比较指标计数或状态不一致：'+key)
    return max(differences,default=0.)


def measurement_fit(operator,predictions,measurements,halo,core_size):
    """不裁剪预测光谱；零测量时相对误差无定义，用null及计数表达。"""
    predictions=np.asarray(predictions,dtype=np.float64);measurements=np.asarray(measurements,dtype=np.float64)
    if predictions.ndim!=4 or measurements.shape!=(*predictions.shape[:3],3) or not len(predictions):raise ValueError('预测与RGB测量尺寸不符')
    if type(halo) is not int or halo<0 or type(core_size) is not int or core_size<1 or any(halo+core_size>v for v in predictions.shape[1:3]):raise ValueError('评价区域越界')
    if not np.isfinite(predictions).all() or not np.isfinite(measurements).all():raise ValueError('测量诊断输入非有限')
    recoded=np.stack([operator.forward(p) for p in predictions])
    roi=lambda x:x[:,halo:halo+core_size,halo:halo+core_size,:]
    error=roi(recoded)-roi(measurements);signal=roi(measurements)
    error2=float(np.sum(error*error));signal2=float(np.sum(signal*signal))
    full_error=recoded-measurements
    return dict(center_rgb_mse=float(np.mean(error*error)),center_relative_l2=float(np.sqrt(error2/signal2)) if signal2>0 else None,
        zero_measurement_patches=int(np.sum(np.sum(signal*signal,axis=(1,2,3))==0)),full_context_rgb_mse=float(np.mean(full_error*full_error)),
        convolution_scope='full context FIRST, then center ROI',prediction_clipped=False),recoded
