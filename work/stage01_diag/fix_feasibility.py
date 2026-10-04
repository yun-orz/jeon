# -*- coding: utf-8 -*-
"""阶段 01 修正：可行性验证（临时脚本）。

验证三件事：
1. 修正后的单次 FFT（补面积权重 + 频率中心化）与原 B 组参数下的完整核求和是否一致；
2. 同一 FFT 的全输出功率是否守恒；
3. D=1 mm、f=50 mm、λ=550 nm 的焦平面 Airy 中心截线在该实现下的精度，
   以及所需的输入采样/零填充规模与内存。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics")
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- 参考实现（独立写出）
def full_kernel_matrix(u1, dx, dy, lam, z, x_out, y_out):
    """完整位移核的矩阵求积（展开核、不预乘输入二次相位）。"""
    k = 2 * np.pi / lam
    lamz = lam * z
    xp = (np.arange(u1.shape[1]) - u1.shape[1] // 2) * dx
    yp = (np.arange(u1.shape[0]) - u1.shape[0] // 2) * dy
    px = np.exp(1j * np.pi * (x_out[:, None] - xp[None, :]) ** 2 / lamz)
    py = np.exp(1j * np.pi * (y_out[:, None] - yp[None, :]) ** 2 / lamz)
    val = py @ u1 @ px.T
    return val * (dx * dy) / (1j * lamz) * np.exp(1j * k * z)


def fft_route(u1, dx, dy, lam, z, Mx, My):
    """修正后的单次 FFT：中心零填充 + 输入二次相位 + 中心化 + 输出二次相位 + 面积权重。"""
    k = 2 * np.pi / lam
    lamz = lam * z
    ny, nx = u1.shape
    oy = (My - ny) // 2
    ox = (Mx - nx) // 2
    big = np.zeros((My, Mx), complex)
    big[oy:oy + ny, ox:ox + nx] = u1
    xp = (np.arange(Mx) - Mx // 2) * dx
    yp = (np.arange(My) - My // 2) * dy
    V = big * np.exp(1j * np.pi * (xp[None, :] ** 2 + yp[:, None] ** 2) / lamz)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(V)))
    x = lamz * np.fft.fftshift(np.fft.fftfreq(Mx, d=dx))
    y = lamz * np.fft.fftshift(np.fft.fftfreq(My, d=dy))
    u2 = F * np.exp(1j * np.pi * (x[None, :] ** 2 + y[:, None] ** 2) / lamz) \
        * (dx * dy) / (1j * lamz) * np.exp(1j * k * z)
    return u2, x, y


def lens_airy_fft(D, f, lam, W, dx, pad_factor, half_out):
    """理想透镜焦平面（z=f）：用修正后的 FFT 求中心截线与 2D 复场。"""
    n = 2 * int(round(W / dx))
    ny = nx = n
    Mx = My = nx * pad_factor
    xp = (np.arange(nx) - nx // 2) * dx
    yp = (np.arange(ny) - ny // 2) * dx
    R2 = xp[None, :] ** 2 + yp[:, None] ** 2
    ap = (R2 <= (D / 2) ** 2).astype(complex)
    u1 = ap * np.exp(-1j * np.pi * R2 / (lam * f))
    u2, x, y = fft_route(u1, dx, dx, lam, f, Mx, My)
    sel_x = np.abs(x) <= half_out
    sel_y = np.abs(y) <= half_out
    return u2[np.ix_(sel_y, sel_x)], x[sel_x], y[sel_y], u1, (ny, nx, My, Mx)


def airy(r, lam, f, D):
    x = np.pi * D * r / (lam * f)
    from scipy import special
    small = np.abs(x) < 1e-9
    xs = np.where(small, 1.0, x)
    out = (2 * special.j1(xs) / xs) ** 2
    return np.where(small, 1.0, out)


if __name__ == "__main__":
    lam, z = 550e-9, 50e-3

    print("=" * 78)
    print("检验 1/2：原 B 组参数（N=48, Δx′=40 μm, M=96）离散恒等与功率守恒")
    print("=" * 78)
    n, dx, M = 48, 40e-6, 96
    xp = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(xp, xp, indexing="xy")
    hw = 0.5 * (n // 2) * dx
    u1 = (np.exp(-((X - 0.30 * hw) / (0.6 * hw)) ** 2 - ((Y + 0.20 * hw) / (0.34 * hw)) ** 2)
          * np.exp(1j * 2 * np.pi / lam * (1.3e-3 * X - 0.9e-3 * Y))
          * np.exp(1j * 2.0e7 * (X ** 3 + 0.5 * Y ** 3))).astype(complex)
    u2f, xf, yf = fft_route(u1, dx, dx, lam, z, M, M)
    u2k = full_kernel_matrix(u1, dx, dx, lam, z, xf, yf)
    rel = np.sqrt(np.sum(np.abs(u2f - u2k) ** 2)) / np.sqrt(np.sum(np.abs(u2k) ** 2))
    p_in = np.sum(np.abs(u1) ** 2) * dx * dx
    p_out = np.sum(np.abs(u2f) ** 2) * (xf[1] - xf[0]) ** 2
    print("  FFT vs 完整核求和 rel L2 = %.4e" % rel)
    print("  FFT 全输出功率 P_out/P_in - 1 = %+.4e   输入面积倒数 = %.6e"
          % (p_out / p_in - 1.0, 1.0 / (dx * dx)))

    print()
    print("=" * 78)
    print("检验 3：理想透镜焦平面 Airy，D=1 mm、f=50 mm、λ=550 nm")
    print("=" * 78)
    from optics import metrics as MM
    D, f = 1e-3, 50e-3
    r1 = MM.airy_first_dark_ring_radius(lam, f, D)
    rings = MM.airy_dark_ring_radii(lam, f, D, 3)
    print("  理论第一暗环：Bessel %.6f μm，1.22λf/D %.6f μm" % (r1 * 1e6, r1 * 1e6))
    for W, dx, pf in [(0.6e-3, 5e-6, 16), (0.6e-3, 2e-6, 8), (0.6e-3, 1e-6, 4)]:
        try:
            u2, x, y, u1, shape = lens_airy_fft(D, f, lam, W, dx, pf, 120e-6)
            line = np.abs(u2[len(y) // 2, :]) ** 2
            line = line / line.max()
            a = airy(np.abs(x), lam, f, D)
            l1 = float(np.sum(np.abs(line - a)) / np.sum(a))
            i0 = int(np.argmin(np.abs(x - rings[0])))
            print("  W=%.2fmm dx=%.3gum pad=%d  形状 %s  L1=%.4e  峰值=%.4f  "
                  "r1 附近实测/理论=%.4f/%.4f"
                  % (W * 1e3, dx * 1e6, pf, shape, l1, line.max(),
                     np.abs(x[i0]) * 1e6, rings[0] * 1e6))
            print("     M=%d，单个 complex128 数组 %.0f MB"
                  % (shape[3], shape[2] * shape[3] * 16 / 2 ** 20))
        except MemoryError:
            print("  W=%.2fmm dx=%.3gum pad=%d  内存不足" % (W * 1e3, dx * 1e6, pf))
