"""阶段 01 第二轮独立审核：只新增证据，不修改 DSH 项目。"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "outputs" / "jeon2019_optics"
OUT = ROOT / "outputs" / "复审_阶段01_20261001"
OUT.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(PROJECT))
import numpy as np
from scipy.special import j0, j1, jn_zeros
from optics.coordinates import Axis1D, Grid2D
from optics.propagation import (FresnelConfig, fresnel_fft,
                               fresnel_kernel_matrix, fresnel_kernel_separable,
                               fresnel_angular_spectrum)


def write_json(name, value):
    """已有证据不覆盖；重复审核应选择新的审核目录。"""
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def source_hashes():
    paths = list(PROJECT.glob("*.py")) + list(PROJECT.glob("optics/*.py"))
    paths += list(PROJECT.glob("tests/*.py")) + [PROJECT / "config.json"]
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def independent_kernel(u, grid, wavelength, distance, xo, yo):
    """按论文式(4) 完整位移核独立构造，不调用被审函数。"""
    coefficient = np.pi / (wavelength * distance)
    kx = np.exp(1j * coefficient * (xo[:, None] - grid.x.coords[None, :]) ** 2)
    ky = np.exp(1j * coefficient * (yo[:, None] - grid.y.coords[None, :]) ** 2)
    pref = np.exp(2j * np.pi / wavelength * distance) / (1j * wavelength * distance)
    return pref * grid.dx * grid.dy * (ky @ u @ kx.T)


def run(cmd, name, cwd):
    print("执行", name, flush=True)
    t0 = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(cmd, cwd=cwd, env=environment, capture_output=True)
    with (OUT / name).open("xb") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    result = {"command": cmd, "cwd": str(cwd), "exit_code": proc.returncode,
              "elapsed_s": time.perf_counter() - t0, "log": name}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


before = source_hashes()
write_json("源码审核前SHA256.json", before)
lam, z = 550e-9, 50e-3
numbers = {}
for label, nx, ny, dx, dy, mx, my in [
        ("原粗采样", 48, 48, 40e-6, 40e-6, 96, 96),
        ("原细采样", 24, 24, 10e-6, 10e-6, 48, 48),
        ("非方形奇数输入", 45, 33, 8e-6, 11e-6, 96, 64)]:
    g = Grid2D(Axis1D(nx, dx, "x"), Axis1D(ny, dy, "y"))
    x, y = np.meshgrid(g.x.coords, g.y.coords)
    u = np.exp(-((x - 15e-6) ** 2 + 1.3 * (y + 20e-6) ** 2) / (100e-6) ** 2)
    u = u * np.exp(1j * (2e4 * x - 3e4 * y + 1e7 * x * y))
    r = fresnel_fft(u, g, FresnelConfig(lam, z, (my, mx)))
    xo = lam * z * np.fft.fftshift(np.fft.fftfreq(mx, dx))
    yo = lam * z * np.fft.fftshift(np.fft.fftfreq(my, dy))
    reference = independent_kernel(u, g, lam, z, xo, yo)
    direct = fresnel_kernel_matrix(u, g, lam, z, xo, yo)
    separated = fresnel_kernel_separable(u, g, lam, z, xo, yo)
    power_in = np.sum(np.abs(u) ** 2) * dx * dy
    power_out = np.sum(np.abs(r.field) ** 2) * (lam * z / (mx * dx)) * (lam * z / (my * dy))
    numbers[label] = {"FFT对独立位移核relL2": rel(r.field, reference),
                      "直接核对独立核relL2": rel(direct, reference),
                      "分离核对独立核relL2": rel(separated, reference),
                      "原生全输出功率相对误差": float(abs(power_out / power_in - 1))}

# 去除输出相位的定义要求补回一次 exp(+iq) 后等于完整场。
g = Grid2D(Axis1D(24, 10e-6, "x"), Axis1D(24, 10e-6, "y"))
x, y = np.meshgrid(g.x.coords, g.y.coords)
u = np.exp(-((x - 15e-6) ** 2 + 1.3 * (y + 20e-6) ** 2) / (70e-6) ** 2)
u = u * np.exp(1j * (2e4 * x - 3e4 * y))
full = fresnel_fft(u, g, FresnelConfig(lam, z, 48))
part = fresnel_fft(u, g, FresnelConfig(lam, z, 48, omit_output_quadratic_phase=True))
xo = lam * z * np.fft.fftshift(np.fft.fftfreq(48, g.dx))
yo = lam * z * np.fft.fftshift(np.fft.fftfreq(48, g.dy))
q = np.pi * (xo[None, :] ** 2 + yo[:, None] ** 2) / (lam * z)
numbers["输出二次相位选项"] = {
    "补回一次二次相位relL2_应接近零": rel(part.field * np.exp(1j * q), full.field),
    "补回两次二次相位relL2_定位错误": rel(part.field * np.exp(2j * q), full.field),
    "相位因子浮点逆运算relL2": rel(full.field * np.exp(-1j * q) * np.exp(1j * q), full.field),
    "最大相位幅角_rad": float(q.max())}
np.savez_compressed(OUT / "输出相位分支独立核查.npz", full=full.field,
                    omitted=part.field, x_m=xo, y_m=yo, q_rad=q)

# 不填充的均匀平面波只有零频率，最直接地检验全局相位开关。
g = Grid2D(Axis1D(32, 20e-6, "x"), Axis1D(32, 20e-6, "y"))
u = np.ones((32, 32), dtype=complex)
yes = fresnel_angular_spectrum(u, g, lam, z, include_global_phase=True)
no = fresnel_angular_spectrum(u, g, lam, z, include_global_phase=False)
numbers["精确角谱全局相位开关"] = {
    "关闭时对单位平面波relL2_应接近零": rel(no.field, u),
    "开关两个输出的relL2_不应恒为零": rel(yes.field, no.field)}

# 连续解析高斯场对照，同时检验绝对复场而非仅形状。
g = Grid2D(Axis1D(240, 5e-6, "x"), Axis1D(240, 5e-6, "y"))
x, y = np.meshgrid(g.x.coords, g.y.coords)
w0 = 100e-6
u = np.exp(-(x * x + y * y) / w0 ** 2).astype(complex)
xo = np.linspace(-200e-6, 200e-6, 65)
yo = np.linspace(-170e-6, 170e-6, 57)
numeric = fresnel_kernel_matrix(u, g, lam, z, xo, yo)
den = 1 + 1j * z * lam / (np.pi * w0 ** 2)
analytic = np.exp(2j * np.pi / lam * z) / den * np.exp(
    -(xo[None, :] ** 2 + yo[:, None] ** 2) / (w0 ** 2 * den))
numbers["连续高斯复场"] = {"relL2": rel(numeric, analytic)}

# 使用 DSH 保存的未归一化场，独立重算绝对能量口径。
latest = PROJECT / "results/stage01/run_20261001_142041"
with np.load(latest / "arrays/energy_window_airy.npz") as a:
    intensity = a["intensity"]
    xo, yo = a["x_m"], a["y_m"]
rr = np.hypot(xo[None, :], yo[:, None])
r1 = float(jn_zeros(1, 1)[0] * lam * z / (np.pi * 1e-3))
radius = 5 * r1
input_n = 2401
xp = (np.arange(input_n) - input_n // 2) * 1e-6
input_count = np.count_nonzero(xp[None, :] ** 2 + xp[:, None] ** 2 <= (0.5e-3) ** 2)
p_in = input_count * (1e-6) ** 2
p_disk = float(np.sum(intensity[rr <= radius]) * (xo[1] - xo[0]) * (yo[1] - yo[0]))
t = np.pi * 1e-3 * radius / (lam * z)
theory = float(1 - j0(t) ** 2 - j1(t) ** 2)
numbers["包围能量定义"] = {
    "实际输入功率_振幅单位平方乘m2": p_in,
    "圆盘功率_振幅单位平方乘m2": p_disk,
    "正确口径Pdisk除Pin": p_disk / p_in,
    "DSH口径Pdisk除有限方窗功率": float(np.sum(intensity[rr <= radius]) / np.sum(intensity)),
    "无限平面解析包围能量": theory,
    "正确口径绝对误差": abs(p_disk / p_in - theory)}
with np.load(latest / "arrays/airy_D1mm_f50mm_field.npz") as a:
    ra = np.hypot(a["x_m"][None, :], a["y_m"][:, None])
    numbers["AiryA组能量空检查"] = {
        "最大已保存半径除第一暗环": float(ra.max() / r1),
        "小于5r1的像素占比": float(np.mean(ra <= radius))}
write_json("独立复审数值.json", numbers)
print(json.dumps(numbers, ensure_ascii=False, indent=2), flush=True)

commands = []
commands.append(run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                    "DSH测试独立复跑.log", PROJECT))
commands.append(run([sys.executable, "-B", str(PROJECT / "main.py"), "--run-dir",
                     str(OUT / "默认入口独立复跑")], "默认入口独立复跑.log", ROOT))

# 无保存模式只运行轻量的 B 组；结果文件集合必须保持不变。
file_before = set(p.relative_to(PROJECT).as_posix() for p in PROJECT.rglob("*") if p.is_file())
commands.append(run([sys.executable, "-B", str(PROJECT / "main.py"), "--no-save", "--only", "direct"],
                    "无保存模式复跑.log", ROOT))
file_after = set(p.relative_to(PROJECT).as_posix() for p in PROJECT.rglob("*") if p.is_file())
write_json("无保存模式文件集合检查.json", {"新增": sorted(file_after - file_before),
                                       "消失": sorted(file_before - file_after)})

# 强制 B 组验收失败，验证真实非零返回码而非仅显示告警。
cfg = json.loads((PROJECT / "config.json").read_text(encoding="utf-8"))
cfg["acceptance"]["discrete_identity_rel_l2_max"] = 0.0
write_json("审核用失败配置.json", cfg)
commands.append(run([sys.executable, "-B", str(PROJECT / "main.py"), "--no-save", "--only", "direct",
                     "--config", str(OUT / "审核用失败配置.json")], "失败返回码检查.log", ROOT))

# 内存中注入三种 Airy 算例，不改变生产源码；第二、三项失败必须影响总验收。
injection = '''import runpy,sys
sys.argv=[sys.argv[1],"--only","airy","--no-save"]
d=runpy.run_path(sys.argv[0],run_name="audit_import")
fn=d["main"]
fn.__globals__["study_airy"]=lambda *a,**k:{"cases":[
 {"first_ring_rel_err":0.001,"line_L1_vs_airy":0.001},
 {"first_ring_rel_err":0.5,"line_L1_vs_airy":0.5},
 {"first_ring_rel_err":0.7,"line_L1_vs_airy":0.7}]}
raise SystemExit(fn())
'''
commands.append(run([sys.executable, "-B", "-c", injection, str(PROJECT / "main.py")],
                    "Airy非首项失败注入检查.log", ROOT))
guard_dir = OUT / "非空目录保护核查"
guard_dir.mkdir()
sentinel = guard_dir / "保留证据.txt"
sentinel.write_text("此文件不得删除或覆盖。", encoding="utf-8")
sentinel_before = sentinel.read_bytes()
commands.append(run([sys.executable, "-B", str(PROJECT / "main.py"), "--run-dir", str(guard_dir)],
                    "已有结果保护检查.log", ROOT))
write_json("独立执行记录.json", commands)
after = source_hashes()
write_json("源码审核后SHA256.json", after)
write_json("审核完整性.json", {"生产源码未变": before == after,
                               "非空目录证据未变": sentinel.read_bytes() == sentinel_before})
print("复审完成，证据目录：", OUT, flush=True)
