# 阶段 01 修正说明（R1–R7 逐项）

日期：2026-10-01。
审核依据：`outputs/审核_阶段01_20261001/阶段01_审核报告.md`、
`独立审核数值.json`、`audit_airy_comparison.png`。

> **旧结论已失效。** 阶段 01 初审版（`results/stage01/run_20260930_*`）与当时的
> `docs/equations_and_parameters.md` 第 3 节、`docs/assumptions.md` A1/A3、
> `docs/reproduction_log.md` 第 3–6 节中的下列结论**全部撤回**，不得再引用：
>
> 1. “单次 FFT(IR) 形式在本项目参数下不可用，是采样条件所限” —— 实际是
>    **漏乘面积权重 `Δx′Δy′` 且频率数组未中心化**，修正后与完整位移核求和
>    相差约 1e−13（旧版 6.25e8）。
> 2. “直接二维求积是权威方法” —— 旧版直接求值**重复加入了一次输入二次相位**，
>    相对解析高斯解误差 1.29369；按式(4) 的正确离散形式误差为 3.7e−15。
> 3. “D=1 mm 时 Fresnel 数 4.5（或 0.2 mm/25 mm 时 2.3），不属于远场，因此不能
>    用 Airy 校验” —— 理想薄透镜在 `z=f` 的焦平面上，透镜相位与传播展开式的输入
>    二次相位相消，焦平面就是孔径的 Fourier 强度，**不要求** `D²/(4λf)≪1`；
>    正确 Fresnel 数为 0.090909（0.1 mm/50 mm）、0.727273（0.2 mm/25 mm）、
>    9.090909（1 mm/50 mm）。
> 4. “有限窗口能量对解析包围能量是等价替换” —— 它只是**补充**检查，
>    不能替代原任务要求的**原生 FFT 全输出 Parseval 检验**（≤1e−10）。
>
> 初审版的运行目录与报告**按原样保留**作为历史记录，未删除、未覆盖。

---

## R1 [P1] FFT 漏面积权重 + 频率未中心化

**怎么修**（`optics/propagation.py::fresnel_fft`）：

```text
V  = 中心零填充后的 u₁ · exp[iπ(x′²+y′²)/(λz)]
F  = fftshift(fft2(ifftshift(V)))
u₂ = F · exp[iπ(x²+y²)/(λz)] · Δx′Δy′ · exp(ikz)/(iλz)
x  = λz·fftshift(fftfreq(Mx, Δx′))，y = λz·fftshift(fftfreq(My, Δy′))
```

- **补上 `Δx′Δy′`**：旧版 `scale = 1/(iλz)` 缺面积权重，正是 6.25e8 的来源
  （= 1/Δx′Δy′，Δx′=40 μm）。
- **频率中心化**：改用 `fftshift(fft2(ifftshift(·)))`，输出坐标直接取自
  `λz·fftshift(fftfreq(...))`，索引与物理坐标一一对应；不再混用两套索引约定。
- **非方形网格**：`Mx ≠ My`、`Δx′ ≠ Δy′` 都按各自方向处理，输出间距分别推导。
- **奇数输入**：零填充偏移改为 `(M − n + 1)//2`，保证输入的"0 号样点"落在填充
  数组的 `M//2` 号样点；否则整体错半个样点（实测会造成 O(0.1) 的差异）。
- **`omit_output_quadratic_phase`**：不再是"解析等价"，而是**明确的场表示变化**，
  元数据里写明"非同一物理复场"，并要求乘回同一相位因子才等于完整物理场。
  已知限制（相位幅角可达上百 π，浮点重构在角点失去有效位）已写成文档与测试。
- **`include_global_phase`**：只控制 `exp(ikz)` 一个因子，从不涉及 `1/(iλz)`；
  默认与直接积分一致（测试断言二者比值恒为 `exp(ikz)`）。

**实际执行结果**（本次正式运行 `results/stage01/run_20261001_141024`）：

| N | Δx′ | M | FFT vs 完整核 rel L2 | 原生全输出功率相对误差 |
| --- | --- | --- | --- | --- |
| 48 | 40 μm | 96 | 1.0797e-13 | 6.6613e-15 |
| 24 | 10 μm | 48 | 1.6779e-14 | 8.8818e-16 |
| 64 | 5 μm | 256 | 2.5219e-13 | 2.1538e-14 |

审核独立值：正确 FFT vs 完整核 3.45e−15、功率 8.66e−15；本实现同量级。

**结果文件**：`metrics/direct_vs_fft_metrics.csv`、`metrics/energy_metrics.csv`、
`arrays/direct_vs_fft_N48_M96.npz`、`figures/B_discrete_identity.png`。

---

## R2 [P1] 直接积分重复加入输入二次相位

**怎么修**（`optics/propagation.py::fresnel_kernel_matrix`）：

```text
u₂(x,y) = exp(ikz)/(iλz) · Σ u₁(x′,y′) · exp[iπ((x−x′)²+(y−y′)²)/(λz)] · Δx′Δy′
```

- 求和项里**只有 `u₁` 本身**，不再预乘 `exp(iπ(x′²+y′²)/(λz))`。
- 展开核的三部分（输入二次相位、交叉 Fourier 核、输出二次相位）只在
  **FFT 路径**里各出现一次，两条路径不混用。
- 另外新增 `fresnel_kernel_separable`：把展开核对 `x′` 的线性耦合分离成两次一维
  矩阵乘法，与完整核矩阵求值**数学等价**（实测 2.1e−16），使论文尺度
  D=1 mm 的中心截线可以在 0.1 s 内精确求值。

**实际执行结果**：

- 数值（完整核）vs 解析自由空间高斯：**1.3866e-14**（正入射算例）；
  审核独立值 3.74e−15，同量级。
- 旧版同一算例为 1.29369，问题确认并已消除。

**结果文件**：测试 `TestContinuousAccuracy.test_numeric_vs_analytic_free_gaussian`；
`arrays/energy_window_airy.npz`（含 1 mm 焦平面复场）。

---

## R3 [P1] Airy 适用条件与近轴误差诊断

**怎么修**：

1. **删除**"必须 `D²/(4λf)≪1` 才能用 Airy 校验"的说法。物理依据改为：
   `z=f` 时透镜相位 `exp(−iπr′²/(λf))` 与传播展开式的输入二次相位
   `exp(+iπr′²/(λf))` 相消，焦平面 = 孔径 Fourier 强度。
   `Fresnel 数`改为**只作报告**，并给出审核采用的 `D²/(4λf)` 定义
   （`axial_fresnel_number`）。
2. **实际求值 D=1 mm、f=50 mm、λ=550 nm 的中心截线**（不再以近场为由回避）。
3. **高阶光程误差**改用精确路径差：

   ```text
   ΔL = √(z²+ρ²) − z − ρ²/(2z) = −ρ⁴ / [2z·(√(z²+ρ²)+z)²]
   相位误差 = |(2π/λ)·ΔL|
   ```

   `ρ` 取**实际输入支撑与输出窗口的角点距离**（方形窗口计入角点）。
   删除了旧函数里多余的 `(L²/8z²)(L²/z²)·2` 高阶因子。
4. **判据口径**：`sampling_diagnostics` 分别给出输入二次相位、交叉项、输出二次相位
   三项的相位步进与保守和，不再只用输入二次相位单项作为"ok"保证；
   半宽按 `(n//2)·d` 计算，**不再额外乘 0.5**。

**实际执行结果**（本次运行）：

| 算例 | N_F | 第一暗环理论 (μm) | 实测 (μm) | 相对误差 | 截线 L1 |
| --- | --- | --- | --- | --- | --- |
| D=0.1 mm, f=50 mm | 0.090909 | 335.409220 | 336.009 | +1.79e-03 | 3.91e-04 |
| D=0.2 mm, f=25 mm | 0.727273 | 83.852305 | 84.012 | +1.91e-03 | 2.13e-04 |
| **D=1.0 mm, f=50 mm** | 9.090909 | 33.540922 | 33.611 | +2.09e-03 | 4.82e-05 |

审核独立值（中心截线 L1）：0.001361 / 0.0002753 / 0.00003783，本实现同量级或更好。

高阶光程误差（本次运行记录）：
`ρ=1.25 mm, z=50 mm, λ=550 nm` → **0.027881847 rad**（审核基准 0.02788184738292828），
四次近似 0.027890560 rad；旧版曾报 4.11e−11 rad。

**结果文件**：`metrics/airy_metrics.csv`、`metrics/airy_enclosed_energy.csv`、
`arrays/airy_D1mm_f50mm_field.npz`、`arrays/airy_D1mm_f50mm_line.npz`、
`figures/A_airy_D1mm_f50mm.png`。

---

## R4 [P1] 测试与验收证据不足

**怎么修**——新增/重写的测试（`tests/test_propagation.py`，共 25 项，全部通过）：

| 审核要求 | 测试 | 实测 |
| --- | --- | --- |
| 1 非对称输入 FFT vs 完整核 | `TestDiscreteIdentity`（3 种规模） | ≤2.6e-13 |
| 2 原 B 组 N=48/Δx′=40 μm/M=96 离散恒等 | `test_original_case_n48_dx40um_M96` | 1.08e-13 |
| 3 原“满足判据”的 N=24/10 μm/M=48 算例 | `test_case_n24_dx10um_M48` | 1.68e-14（旧版 1e10） |
| 4 FFT 全输出功率 ≤1e-10 + 直接求积交叉核对 | `test_fft_power_conservation`、`test_kernel_matrix_matches_explicit_2d_sum` | 6.7e-15 / 1e-12 |
| 5 数值 vs 连续自由高斯解析 | `test_numeric_vs_analytic_free_gaussian` | 1.39e-14 |
| 6 D=1 mm Airy 中心截线，暗环 ≤2% | `test_airy_central_line_three_apertures` | 2.09e-03 |
| 7 非方形 / x-y 间距不同 / 奇数输入 | `test_non_square_and_different_spacing`、`test_odd_input_supported_and_origin_aligned` | ≤1e-6 |
| 8 两个填充倍率 + 三个输入精度 | `TestSamplingConvergence` | 见 D 组 |
| 9 精确光程差 vs 四次近似 | `test_known_phase_error_value` | 与审核基准一致到 1e-9 |
| 10 相位接口一致性 | `TestInterfaceSemantics` | 见 R1 |

- **验收状态进入机器可读总状态**：`acceptance_check.json` 里每项为
  `pass/fail/not_run`，总体 `status ∈ {pass, partial, fail}`；
  **任何 fail 或缺测都让进程返回非零退出码**（旧版验收 false 却返回 0）。
- `--only` 未执行的组标为 `not_run`，与失败区分。

---

## R5 [P2] 行坐标说明与实现相反

**怎么修**：统一为**索引增加 → x、y 都增加**，显示用 `origin='lower'`。

- `optics/coordinates.py` 文档、`Axis1D.coords` 注释、`Grid2D.describe()`
  的 `array_index_convention` 全部改正。
- 测试 `test_index_increases_with_coordinate` / `test_grid_index_convention`
  显式断言"行增加 → y 增加、列增加 → x 增加"，并配非对称输入。
- **图像 extent 区分两种约定**：`extent_center_um`（样点中心）与
  `extent_pixel_um`（像素边界，外扩半像元）；画图与像元积分都用后者，
  测试 `test_extent_center_vs_pixel` 断言二者差一个像元。
- 没有翻转任何数组，避免阶段 03 的顺/逆时针判断被误判。

---

## R6 [P1] `--no-save` 落盘、`--run-dir` 保护不足

**怎么修**：

- `--no-save`（别名 `--log-to-console-only`）：**真正不写任何文件**。
  `run_dir=None` 一路传下去，`save_npz` / `write_csv_maybe` / `save_fig` /
  `write_json` / `setup_logger` 都跳过落盘，日志只写标准输出。
  旧的固定 `results/_nosave` 分支已删除（该目录作为历史痕迹保留，未删除）。
- `--run-dir`：目标**必须不存在或为空**，否则拒绝执行并返回码 3；
  不再只看 `reproduction.json`。
- 相对路径约定：`--config` 与 `--run-dir` 的相对路径按**当前工作目录**解析
  （写入 `--help` 与 README）；默认入口仍按 `__file__` 解析，不依赖启动目录。

---

## R7 [P2] 固定 Agg 阻断 GUI

**怎么修**：后端选择移到**导入 pyplot 之前**（`select_backend`），按模式决定：

- `show_plots=False`：`Agg`（批处理，不依赖窗口系统）。
- `show_plots=True`：依次尝试 `QtAgg / TkAgg / GTK3Agg / WXAgg / MacOSX`，
  成功即用；全部失败则回退 `Agg` 并在报告与日志里写明"不会弹窗"。

本次正式运行（无 `--show-plots`）实际后端为 **Agg**。
**GUI 弹窗路径仍未人工验证**（本环境无法显示窗口），报告"未验证项"中如实标注。

---

## 仍未完成 / 未验证

1. **PyCharm 点击运行与 `--show-plots` 弹窗仍未人工验证**；命令行入口已实际运行
   （含从其他工作目录启动）。
2. 有限窗口对解析包围能量的**绝对差**约 0.03（5 倍第一暗环半径处），
   原因是解析式是无限平面占比而离散场在窗内尚未积完；它是**补充**检查，
   主判据用原生 FFT 全输出 Parseval（实测 ≤2.2e−14）。若需要更紧的一致性，
   应把窗口扩到解析占比收敛处，属后续阶段。
3. 倾斜高斯的**解析复场**对照没有作为独立证据：正入射情形已到 1e−14，
   但倾斜解析式对沿倾斜轴传播的相位项极敏感，本阶段改为断言可由物理直接预期的
   三个量（峰值幅值、几何位移 z·tanθ、1/e 半宽），并把 0.48 的残差如实记录在
   `docs/reproduction_log.md`。**不把它算作通过项。**
4. D=1 mm 的二维焦平面场用**完整位移核的可分离求值**得到（959×959，Δx=0.35 μm），
   不是单次 FFT：FFT 路线的原生输出间距 λz/(MΔx′) 要做到 20 样点/暗环需
   M≈1.6e4（约 4 GB、约 5 分钟）。两条路径的一致性由 B 组的离散恒等覆盖。
5. 仍未实现 DOE 高度、三翼结构、多波长 PSF、图 3、成像与重建。
