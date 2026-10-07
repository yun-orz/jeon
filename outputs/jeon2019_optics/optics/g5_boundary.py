# -*- coding: utf-8 -*-
"""固定中心、共享测量的有限域边界诊断工具。"""
import numpy as np


def center_crop(value, size):
    """裁最后三个维度中的空间轴，保留相同物理中心。"""
    value = np.asarray(value)
    if value.ndim < 3 or type(size) is not int or size < 1:
        raise ValueError('数组维度或裁剪尺寸非法')
    h, w = value.shape[-3:-1]
    if size > min(h, w) or (h-size) % 2 or (w-size) % 2:
        raise ValueError('裁剪越界或无法保持同一像素中心')
    y, x = (h-size)//2, (w-size)//2
    return value[..., y:y+size, x:x+size, :]


def support_tiles(raw, labels, origins, old_size, support_size, sensitivity, scale):
    """按原列表顺序检查支持域；无效块明确排除，绝不另选高分块。"""
    from optics.g4_data import photon_relative
    raw = np.asarray(raw); labels = np.asarray(labels)
    if raw.ndim != 3 or raw.shape[-1] != 31 or labels.shape != raw.shape[:2]:
        raise ValueError('场景尺寸错误')
    if type(support_size) is not int or support_size < old_size or (support_size-old_size) % 2:
        raise ValueError('支持域尺寸非法')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('固定尺度非法')
    tiles, records = [], []
    offset = (support_size-old_size)//2
    for index, (y, x) in enumerate(origins):
        sy, sx = y-offset, x-offset
        bounds = sy >= 0 and sx >= 0 and sy+support_size <= raw.shape[0] and sx+support_size <= raw.shape[1]
        record = dict(original_index=index, original_origin_yx=[y, x], support_origin_yx=[sy, sx], in_bounds=bounds)
        if not bounds:
            record.update(accepted=False, reason='支持域越出原始图像', invalid_pixels=None)
        else:
            block = raw[sy:sy+support_size, sx:sx+support_size, :25]
            valid = (labels[sy:sy+support_size, sx:sx+support_size] != 0) & np.isfinite(block).all(-1) & (block >= 0).all(-1)
            invalid = int(np.count_nonzero(~valid))
            record.update(accepted=invalid == 0, invalid_pixels=invalid, reason='完整有效支持域' if invalid == 0 else '官方掩码或光谱无效')
            if invalid == 0:
                tiles.append((photon_relative(block, sensitivity[:25], np.arange(420, 661, 10))/scale).astype(np.float32))
        records.append(record)
    if not tiles:
        raise ValueError('没有完整有效支持域；不填补无效像元')
    return np.stack(tiles), records


def change_metrics(value, reference):
    """对固定中心所有块合并计算；参考范数为零时明确标记无定义。"""
    a = np.asarray(value, dtype=np.float64); b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('差异比较尺寸或数值非法')
    d = a-b; denominator = float(np.linalg.norm(b.ravel()))
    return dict(rmse=float(np.sqrt(np.mean(d*d))), max_absolute=float(np.max(np.abs(d))),
                relative_l2=float(np.linalg.norm(d.ravel())/denominator) if denominator > 0 else None,
                reference_norm_zero=denominator == 0)
