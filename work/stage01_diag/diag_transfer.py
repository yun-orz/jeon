# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 8（临时）：传递函数（卷积）形式的 Fresnel 传播 vs 轴向解析解。"""
import numpy as np

lam, z = 550e-9, 50e-3
lamz = lam * z
k = 2 * np.pi / lam


def transfer_form(u1, dx, M, pad_value=0.0):
    """u2 = IFFT[ FFT[u1] · H ]，H(f) = exp(ikz)·exp(-iπλz f²)；输出间距与输入相同。"""
    n = u1.shape[0]
    big = np.full((M, M), pad_value, complex)
    o = (M - n) // 2
    big[o:o + n, o:o + n] = u1
    f = np.fft.fftfreq(M, d=dx)                     # cycles/m
    FX, FY = np.meshgrid(f, f, indexing="xy")
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * lamz * (FX ** 2 + FY ** 2))
    out = np.fft.ifft2(np.fft.fft2(big) * H)
    return out


def on_axis_exact(D):
    return np.exp(1j * k * z) * (1 - np.exp(1j * np.pi * D ** 2 / (4 * lamz)))


print("=" * 78)
print("传递函数形式：轴向值 vs 解析值（D=0.5mm 与 D=0.2mm）")
print("=" * 78)
for D in (0.5e-3, 0.2e-3):
    print("--- D = %.3g mm，解析 u2(0,0) = %s" % (D * 1e3, on_axis_exact(D)))
    for N, half in [(256, 2e-3), (512, 2e-3), (512, 4e-3), (1024, 4e-3)]:
        dx = 2 * half / N
        c = (np.arange(N) - N // 2) * dx
        X, Y = np.meshgrid(c, c, indexing="xy")
        u1 = (np.hypot(X, Y) <= D / 2).astype(complex)
        for M in (N, 2 * N, 4 * N):
            out = transfer_form(u1, dx, M)
            v = out[M // 2, M // 2]
            print("   N=%4d M=%5d Δx=%7.3fum Δx²/(λz)=%6.3g  u2(0,0)=%-32s relerr=%.3e"
                  % (N, M, dx * 1e6, dx ** 2 / lamz, v,
                     abs(v - on_axis_exact(D)) / abs(on_axis_exact(D))))
