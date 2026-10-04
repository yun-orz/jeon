"""在相同物理输出坐标上，独立检查 D 组输入采样是否收敛。"""
from pathlib import Path
import json
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "复审_阶段01_20261001"
lam, z, half = 550e-9, 50e-3, 2e-3
xo = np.linspace(-200e-6, 200e-6, 65)
yo = xo.copy()
fields = {}
for n in (32, 64, 128, 512, 1024, 2048):
    dx = 2 * half / n
    xp = (np.arange(n) - n // 2) * dx
    x, y = np.meshgrid(xp, xp)
    # 按生产代码中的同一连续输入函数取样；物理半宽与参数全部固定。
    u = np.exp(-((x - 0.3 * half) / (0.55 * half)) ** 2
               - ((y + 0.22 * half) / (0.3 * half)) ** 2)
    u = u * np.exp(2j * np.pi / lam * (1.3e-3 * x - 0.9e-3 * y))
    u = u * np.exp(1j * 2e7 * (x ** 3 + 0.5 * y ** 3))
    kx = np.exp(1j * np.pi / (lam * z) * (xo[:, None] - xp[None, :]) ** 2)
    ky = np.exp(1j * np.pi / (lam * z) * (yo[:, None] - xp[None, :]) ** 2)
    fields[n] = np.exp(2j * np.pi / lam * z) / (1j * lam * z) * dx ** 2 * (ky @ u @ kx.T)
    print("完成输入等级", n, flush=True)

reference = fields[2048]
iref = np.abs(reference) ** 2
rows = []
for n, field in fields.items():
    rows.append({"N": n, "输入间距_um": 2 * half / n * 1e6,
                 "复场相对N2048的relL2": float(np.linalg.norm(field - reference) / np.linalg.norm(reference)),
                 "未归一化强度相对N2048的L1": float(np.sum(np.abs(np.abs(field) ** 2 - iref)) / np.sum(iref))})
result = {"说明": "所有等级采用同一[-200,200]μm输出网格和完整位移核；N2048是数值参考，不声称无限精度解析解。",
          "输出间距_um": float((xo[1] - xo[0]) * 1e6), "比较": rows}
with (OUT / "D组真实输入采样比较.json").open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
np.savez_compressed(OUT / "D组真实输入采样比较.npz", x_m=xo, y_m=yo,
                    **{f"u_N{n}": u for n, u in fields.items()})
print(json.dumps(result, ensure_ascii=False, indent=2))
