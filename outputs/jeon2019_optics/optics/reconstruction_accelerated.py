# -*- coding: utf-8 -*-
"""固定实际α的加速投影梯度，带回溯和单调重启；不接收真值。"""
import time
import numpy as np
from .imaging import real_finite
from .reconstruction import squared_norm


def validate_accelerated(c):
    for name in ['gradient_tolerance','objective_tolerance']:
        if not np.isfinite(c[name]) or c[name]<=0:raise ValueError(name+'须有限且为正')
    for name in ['max_iterations','max_backtracks']:
        if isinstance(c[name],bool) or not isinstance(c[name],int) or c[name]<1:raise ValueError(name+'须为正整数')
    if not np.isfinite(c['backtracking_factor']) or c['backtracking_factor']<=1:raise ValueError('回溯倍数须大于1')


def solve_accelerated_ridge(op,measurement,alpha,config,L_initial):
    validate_accelerated(config)
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('实际α须有限且为正')
    if not np.isfinite(L_initial) or L_initial<=0:raise ValueError('初始L须有限且为正')
    y=real_finite(measurement,'加速重建测量')
    if y.shape!=op.output_shape:raise ValueError('测量shape与算子不匹配')
    start=time.perf_counter();L=float(L_initial)
    x=np.zeros(op.cube_shape);z=x.copy();t=1.
    objective=.5*squared_norm(y);ynorm=np.sqrt(squared_norm(y))
    denominator=max(np.sqrt(squared_norm(op.adjoint(y))),1e-300)
    history=[];status='iteration_limit';restarts=0
    for iteration in range(1,config['max_iterations']+1):
        restarted=False;total_backtracks=0
        # 最多一次单调重启；重启后从接受点重新计算梯度和候选，不能跳过更新。
        for restart_attempt in range(2):
            rz=op.forward(z)-y;fz=.5*squared_norm(rz)+.5*alpha*squared_norm(z)
            gz=op.adjoint(rz)+alpha*z
            for attempt in range(config['max_backtracks']):
                candidate=np.maximum(0.,z-gz/L);delta=candidate-z
                residual=op.forward(candidate)-y
                data=.5*squared_norm(residual);reg=.5*alpha*squared_norm(candidate);value=data+reg
                bound=fz+float(np.sum(gz*delta))+.5*L*squared_norm(delta)
                rounding=32*np.finfo(float).eps*max(abs(fz),abs(value),abs(objective),1e-300)
                if np.isfinite(value) and value<=bound+rounding:break
                L*=config['backtracking_factor'];total_backtracks+=1
            else:raise RuntimeError('加速回溯达到上限')
            if value<=objective+rounding:break
            if restart_attempt==1:raise RuntimeError('单调重启后目标仍异常上升')
            z=x.copy();t=1.;restarted=True;restarts+=1
        change=abs(objective-value)/max(abs(objective),1e-300)
        gradient=op.adjoint(residual)+alpha*candidate
        mapping=L*(candidate-np.maximum(0.,candidate-gradient/L))
        pg=float(np.sqrt(squared_norm(mapping))/denominator)
        resnorm=np.sqrt(squared_norm(residual))
        t_new=.5*(1+np.sqrt(1+4*t*t));momentum=(t-1)/t_new
        z_new=candidate+momentum*(candidate-x)
        history.append(dict(iteration=iteration,data_term=data,regularization_term=reg,objective=value,
            residual_norm=resnorm,residual_relative=resnorm/ynorm if ynorm>0 else None,L=L,
            backtracks=total_backtracks,restarted=restarted,restarts_total=restarts,
            momentum=float(momentum),extrapolated_negative_count=int(np.count_nonzero(z_new<0)),
            projected_gradient_relative=pg,objective_relative_change=change,elapsed_s=time.perf_counter()-start))
        x,z,t,objective=candidate,z_new,t_new,value
        if pg<=config['gradient_tolerance'] and change<=config['objective_tolerance']:
            status='converged';break
    return dict(reconstruction=x,status=status,iterations=len(history),alpha=float(alpha),history=history,
        elapsed_s=time.perf_counter()-start,projected_gradient_normalizer=denominator,
        initialization='zeros',restarts=restarts,algorithm='accelerated_projected_gradient_monotone_restart')
