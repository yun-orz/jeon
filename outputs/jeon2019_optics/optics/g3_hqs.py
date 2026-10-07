# -*- coding: utf-8 -*-
"""Jeon式(21)的HQS展开及四层带软阈值的U-net实施解释。"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TorchRGB(nn.Module):
    """可微的G2零延拓线性卷积；内部张量采用[N,C,H,W]。"""
    def __init__(self,kernels,response,spectral_factor=1.):
        super().__init__()
        k=np.array(kernels,dtype=float,copy=True);r=np.array(response,dtype=float,copy=True)
        if k.ndim!=3 or any(s%2!=1 for s in k.shape[1:]) or r.shape!=(3,k.shape[0]):raise ValueError('核或RGB响应形状不匹配')
        if not np.all(np.isfinite(k)) or np.any(k<0) or not np.all(np.isfinite(r)) or np.any(r<0):raise ValueError('物理核和响应须有限非负')
        if not np.isfinite(spectral_factor) or spectral_factor<=0:raise ValueError('带宽因子须正且有限')
        self.register_buffer('kernels',torch.tensor(k))
        self.register_buffer('response',torch.tensor(r))
        self.spectral_factor=float(spectral_factor)

    def _conv(self,value,flip=False):
        h,w=value.shape[-2:];kh,kw=self.kernels.shape[-2:]
        size=(h+kh-1,w+kw-1)
        kernel=self.kernels.flip((-2,-1)) if flip else self.kernels
        # 充分零填充后乘频谱，取full卷积中心；不使用周期边界。
        frequency=torch.fft.rfft2(value,s=size)*torch.fft.rfft2(kernel,s=size)[None]
        full=torch.fft.irfft2(frequency,s=size)
        cy,cx=kh//2,kw//2
        return full[...,cy:cy+h,cx:cx+w]

    def forward(self,cube):
        if cube.ndim!=4 or cube.shape[1]!=len(self.kernels):raise ValueError('光谱输入须为[N,波段,H,W]')
        return torch.einsum('nbhw,cb->nchw',self._conv(cube),self.response)*self.spectral_factor

    def adjoint(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1]!=3:raise ValueError('RGB输入须为[N,3,H,W]')
        mixed=torch.einsum('nchw,cb->nbhw',rgb,self.response)*self.spectral_factor
        return self._conv(mixed,True)


def inverse_softplus(value):
    if not math.isfinite(value) or value<=0:raise ValueError('初值须有限且为正')
    return math.log(math.expm1(value))


class ConvPair(nn.Module):
    def __init__(self,cin,cout,signed_output=False):
        super().__init__()
        parts=[nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(),nn.Conv2d(cout,cout,3,padding=1)]
        if not signed_output:parts.append(nn.ReLU())
        self.layers=nn.Sequential(*parts)

    def forward(self,x):return self.layers(x)


class SpectralPrior(nn.Module):
    """四层U-net、跳连拼接、输出前软阈值；未公开细节用显式假设。"""
    def __init__(self,bands,features,levels,threshold):
        super().__init__()
        if type(features) is not int or features<2 or type(levels) is not int or levels<2:raise ValueError('通道数或层数非法')
        widths=[features*2**k for k in range(levels)]
        self.encoders=nn.ModuleList([ConvPair(bands if k==0 else widths[k-1],v) for k,v in enumerate(widths)])
        self.ups=nn.ModuleList([nn.ConvTranspose2d(widths[k],widths[k-1],2,stride=2) for k in range(levels-1,0,-1)])
        self.decoders=nn.ModuleList([ConvPair(2*widths[k-1],widths[k-1],signed_output=k==1) for k in range(levels-1,0,-1)])
        self.head=nn.Conv2d(features,bands,3,padding=1)
        self.raw_threshold=nn.Parameter(torch.tensor(inverse_softplus(threshold)))
        self.multiple=2**(levels-1)

    @property
    def threshold(self):return F.softplus(self.raw_threshold)

    def forward(self,value):
        h,w=value.shape[-2:]
        # 仅网络特征补零至池化整除尺寸；光学算子仍保留原图像域。
        x=F.pad(value,(0,(-w)%self.multiple,0,(-h)%self.multiple))
        skips=[]
        for k,encode in enumerate(self.encoders):
            if k:x=F.max_pool2d(x,2)
            x=encode(x);skips.append(x)
        for k,(up,decode) in enumerate(zip(self.ups,self.decoders)):
            x=decode(torch.cat((up(x),skips[-2-k]),dim=1))
        x=torch.sign(x)*F.relu(torch.abs(x)-self.threshold)
        return self.head(x)[...,:h,:w]


class HQSStage(nn.Module):
    def __init__(self,bands,config):
        super().__init__()
        self.prior=SpectralPrior(bands,config['features'],config['levels'],config['threshold_init'])
        self.raw_epsilon=nn.Parameter(torch.tensor(inverse_softplus(config['epsilon_init'])))
        self.raw_rho=nn.Parameter(torch.tensor(inverse_softplus(config['rho_init'])))

    @property
    def epsilon(self):return F.softplus(self.raw_epsilon)

    @property
    def rho(self):return F.softplus(self.raw_rho)

    def forward(self,x,y,operator,backprojection):
        # 对应式(21)，V(l)=S_l(I(l))；同一阶段先求先验再更新数据项。
        v=self.prior(x)
        data=operator.adjoint(operator(x))-backprojection
        updated=x-self.epsilon*(data+self.rho*(x-v))
        return updated,v,data


class HQSDecoder(nn.Module):
    def __init__(self,operator,config):
        super().__init__()
        if type(config['stages']) is not int or config['stages']<1:raise ValueError('展开阶段数须正整数')
        self.operator=operator
        # 各阶段的先验、步长、惩罚与软阈值独立，不共享权重。
        self.stages=nn.ModuleList([HQSStage(len(operator.kernels),config) for _ in range(config['stages'])])

    def forward(self,y,return_trace=False):
        initial=self.operator.adjoint(y);x=initial;trace=[]
        for stage in self.stages:
            previous=x;x,v,data=stage(previous,y,self.operator,initial)
            if return_trace:trace.append(dict(previous=previous,prior=v,data_gradient=data,updated=x,
                epsilon=stage.epsilon,rho=stage.rho,threshold=stage.prior.threshold))
        return (x,initial,trace) if return_trace else x
