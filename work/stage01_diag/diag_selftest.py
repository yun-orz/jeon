# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 7（临时）：自带实现的单次 FFT 传播 vs 库函数，逐项打印。"""
import numpy as np

lam, z = 550e-9, 50e-3
lamz = lam * z
k = 2 * np.pi / lam
N = M = 512
half = 2e-3
D = 0.5e-3

dx = 2 * half / N
coords = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(coords, coords, indexing="xy")
ap = (np.hypot(X, Y) <= D / 2).astype(float)
u1 = ap.astype(complex)

r2in = X ** 2 + Y ** 2
phi_in = np.pi * r2in / lamz
chirp = np.exp(1j * phi_in)
A = u1 * chirp
print("sum(A)          =", A.sum())
print("sum(A*e^{-i2pi(m+n)/2}) =", np.sum(A * np.outer((-1.0) ** np.arange(M), (-1.0) ** np.arange(M))))
U = np.fft.fft2(A)
print("numpy U[0,0]    =", U[0, 0])
print("numpy U[256,256]=", U[M // 2, M // 2])
print("|U| max          = %.6g   argmax=%s" % (np.abs(U).max(), np.unravel_index(np.argmax(np.abs(U)), U.shape)))

dxo = lamz / (M * dx)
cout = (np.arange(M) - M // 2) * dxo
Xo, Yo = np.meshgrid(cout, cout, indexing="xy")
qphase = np.exp(1j * np.pi * (Xo ** 2 + Yo ** 2) / lamz)
field = U * qphase * np.exp(1j * k * z) / (1j * lamz)
print("自写 field(0,0) =", field[M // 2, M // 2])
exact = np.exp(1j * k * z) * (1 - np.exp(1j * np.pi * D ** 2 / (4 * lamz)))
print("解析 u2(0,0)    =", exact)
print("U[0,0]*pref     =", U[0, 0] * np.exp(1j * k * z) / (1j * lamz))
print("U[0,0]/sum(A)   =", U[0, 0] / A.sum())
