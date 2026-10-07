# -*- coding: utf-8 -*-
"""非负岭目标的强凸误差界；只使用测量与恢复，不使用真值。"""
import numpy as np
from .imaging import real_finite

def ridge_certificate(op,measurement,x,alpha):
    if isinstance(alpha,bool) or not np.isfinite(alpha) or alpha<=0:
        raise ValueError('强凸证书要求有限正α')
    x=real_finite(x,'证书恢复');y=real_finite(measurement,'证书测量')
    if x.shape!=op.cube_shape or y.shape!=op.output_shape or np.any(x<0):
        raise ValueError('证书要求形状正确且恢复非负')
    residual=op.forward(x)-y
    gradient=op.adjoint(residual)+alpha*x
    # 精确零处允许约束法向抵消正梯度；不能把小正值当零。
    subgradient=np.where(x>0,gradient,np.minimum(gradient,0.))
    norm=float(np.linalg.norm(subgradient.ravel()))
    xnorm=float(np.linalg.norm(x.ravel()))
    bound=norm/alpha
    return dict(minimum_subgradient_norm=norm,distance_upper_bound=bound,
        distance_upper_bound_relative=bound/xnorm if xnorm>0 else None,
        objective_gap_upper_bound=norm**2/(2*alpha),alpha=float(alpha),
        positive_count=int(np.count_nonzero(x>0)),zero_count=int(np.count_nonzero(x==0)))
