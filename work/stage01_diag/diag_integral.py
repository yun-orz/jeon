# -*- coding: utf-8 -*-
"""阶段 01 调试脚本 13（临时）：用两种互相独立的数值方法核对同一个复积分。

被积函数 f(c) = exp(-c²/w0²)·exp(i k sinθ c)·exp(iπ c²/(λz))，积分区间 ±6 mm。
"""
import numpy as np
from scipy import integrate, special

lam, z, w0, tilt = 550e-9, 50e-3, 0.39e-3, 1e-3
k = 2 * np.pi / lam
lamz = lam * z
alpha = 1 / w0 ** 2 - 1j * np.pi / lamz
print("alpha = %s ; Re>0? %s" % (alpha, alpha.real > 0))
beta = 1j * k * tilt


def f(c):
    return np.exp(-alpha * c ** 2 + beta * c)


# (1) 密网格梯形积分
for n, half in [(2000000, 6e-3), (4000000, 12e-3), (400000, 6e-3)]:
    c = (np.arange(n) - n // 2) * (2 * half / n)
    tot = np.sum(f(c)) * (2 * half / n)
    print("grid n=%8d half=%5.1fmm  I = %s" % (n, half * 1e3, tot))

# (2) 闭式解（误差函数），带正确的分支处理
sq = np.sqrt(alpha)                       # Re(sq) > 0
limit = sq * 6e-3 - beta / (2 * sq)
val = np.sqrt(np.pi) / sq * np.exp(beta ** 2 / (4 * alpha)) * special.erf(limit)
print("closed form (Re sq>0)      I = %s" % val)
sq2 = -sq
limit2 = sq2 * 6e-3 - beta / (2 * sq2)
val2 = np.sqrt(np.pi) / sq2 * np.exp(beta ** 2 / (4 * alpha)) * special.erf(limit2)
print("closed form (-sqrt)        I = %s" % val2)
print("erf(limit) =", special.erf(limit))
print("exp(beta²/4alpha) =", np.exp(beta ** 2 / (4 * alpha)))
print("beta²/(4alpha) =", beta ** 2 / (4 * alpha))
