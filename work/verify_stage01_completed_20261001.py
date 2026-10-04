"""最终独立复场核验及PyCharm默认入口等价运行，证据不覆盖。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import numpy as np
from scipy.special import j0, j1

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "outputs/jeon2019_optics"
OUT = Path((ROOT / "work/接手阶段01当前目录.txt").read_text(encoding="utf-8"))
sys.path.insert(0, str(PROJECT))
from optics.coordinates import Axis1D, Grid2D
from optics.propagation import (FresnelConfig, fresnel_fft, fresnel_kernel_matrix,
                               fresnel_angular_spectrum)

lam, z = 550e-9, .05
g = Grid2D(Axis1D(45, 8e-6, "x"), Axis1D(33, 11e-6, "y"))
x, y = g.meshgrid()
u = np.exp(-((x - 15e-6) ** 2 + 1.3 * (y + 20e-6) ** 2) / (100e-6) ** 2)
u = u * np.exp(1j * (2e4 * x - 3e4 * y))
f = fresnel_fft(u, g, FresnelConfig(lam, z, (64, 96)))
xo = lam * z * np.fft.fftshift(np.fft.fftfreq(96, g.dx))
yo = lam * z * np.fft.fftshift(np.fft.fftfreq(64, g.dy))
kx = np.exp(1j * np.pi / (lam * z) * (xo[:, None] - g.x.coords[None, :]) ** 2)
ky = np.exp(1j * np.pi / (lam * z) * (yo[:, None] - g.y.coords[None, :]) ** 2)
reference = np.exp(2j * np.pi / lam * z) * g.cell_area / (1j * lam * z) * (ky @ u @ kx.T)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
numbers = {"矩形奇数输入FFT对独立式4相对L2": rel(f.field, reference)}
reduced = fresnel_fft(u, g, FresnelConfig(lam, z, (64, 96), omit_output_quadratic_phase=True))
q = np.pi * (xo[None, :] ** 2 + yo[:, None] ** 2) / (lam * z)
numbers["补回一次输出相位相对L2"] = rel(reduced.field * np.exp(1j * q), f.field)
gp = Grid2D(Axis1D(32, 20e-6, "x"), Axis1D(32, 20e-6, "y"))
ones = np.ones(gp.shape, dtype=complex)
numbers["精确角谱去掉全局相位的单位平面波误差"] = rel(
    fresnel_angular_spectrum(ones, gp, lam, z, include_global_phase=False).field, ones)

# 倾斜高斯直接对独立闭式表达式，不调用测试辅助函数。
gg = Grid2D(Axis1D(240, 5e-6, "x"), Axis1D(240, 5e-6, "y"))
X, Y = gg.meshgrid()
w0, tx, ty = 100e-6, .001, -.0004
k = 2 * np.pi / lam
u = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2) * np.exp(1j * k * (tx * X + ty * Y))
xo, yo = np.linspace(-200e-6, 200e-6, 61), np.linspace(-170e-6, 170e-6, 53)
X, Y = np.meshgrid(xo, yo)
den = 1 + 1j * z * lam / (np.pi * w0 ** 2)
analytic = (np.exp(1j * k * z) / den
            * np.exp(-((X - z * tx) ** 2 + (Y - z * ty) ** 2) / (w0 ** 2 * den))
            * np.exp(1j * k * (tx * X + ty * Y - z * (tx * tx + ty * ty) / 2)))
numbers["双方向倾斜高斯独立解析复场相对L2"] = rel(
    fresnel_kernel_matrix(u, gg, lam, z, xo, yo), analytic)

# 按保存的坐标、复场和输入孔径，重新积分，不使用汇总数值作为计算输入。
r = OUT / "完整运行"
with np.load(r / "arrays/airy_D1mm_f50mm_field.npz", allow_pickle=False) as data:
    pin = float(np.sum(data["aperture"] ** 2) * (1e-6) ** 2)
with np.load(r / "arrays/energy_absolute_direct40_R4r1.npz", allow_pickle=False) as data:
    dx, dy = np.diff(data["x_m"])[0], np.diff(data["y_m"])[0]
    rr = np.hypot(data["x_m"][None, :], data["y_m"][:, None])
    R = float(data["R_m"])
    pdisk = float(np.sum(abs(data["u2"][rr <= R]) ** 2) * dx * dy)
    argument = np.pi * 1e-3 * R / (lam * z)
    analytic_energy = float(1 - j0(argument) ** 2 - j1(argument) ** 2)
    numbers["保存数组独立能量积分"] = {"P_in": pin, "P_disk": pdisk,
                                      "E_numeric": pdisk / pin,
                                      "E_analytic": analytic_energy,
                                      "absolute_difference": abs(pdisk / pin - analytic_energy)}

with (OUT / "最终独立物理核验.json").open("x", encoding="utf-8") as fp:
    json.dump(numbers, fp, ensure_ascii=False, indent=2, allow_nan=False)
assert numbers["矩形奇数输入FFT对独立式4相对L2"] < 1e-6
assert numbers["补回一次输出相位相对L2"] < 1e-10
assert numbers["双方向倾斜高斯独立解析复场相对L2"] < 1e-6
assert numbers["保存数组独立能量积分"]["absolute_difference"] < .005
print(json.dumps(numbers, ensure_ascii=False, indent=2), flush=True)

# 从项目外启动，不传配置或输出路径，等价于PyCharm默认入口。
result_root = PROJECT / "results/stage01"
before = set(result_root.iterdir())
env = os.environ.copy()
env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
start = time.perf_counter()
proc = subprocess.run([sys.executable, "-B", str(PROJECT / "main.py")],
                      cwd=ROOT, env=env, capture_output=True)
with (OUT / "最终默认入口.log").open("xb") as fp:
    fp.write(proc.stdout + proc.stderr)
added = set(result_root.iterdir()) - before
record = {"command": [sys.executable, "-B", str(PROJECT / "main.py")],
          "cwd": str(ROOT), "exit_code": proc.returncode,
          "elapsed_s": time.perf_counter() - start,
          "new_runs": [str(p) for p in sorted(added)]}
with (OUT / "最终默认入口记录.json").open("x", encoding="utf-8") as fp:
    json.dump(record, fp, ensure_ascii=False, indent=2)
print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
if proc.returncode:
    print((proc.stdout + proc.stderr).decode("utf-8", errors="replace")[-3000:])
raise SystemExit(proc.returncode)
