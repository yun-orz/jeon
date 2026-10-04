# 公式、参数与来源对照（阶段 01）

> **本文档已于 2026-10-01 修正。** 旧版第 3 节（传播实现）与第 4 节（能量关系）
> 含有三处已被审核证伪的内容，均已就地更正并加"修正提示"：
>
> 1. 旧版 3.1 给 `u₁` 预乘了输入平面二次相位 → **重复相位**，已删除；
> 2. 旧版 3.2 声称"单次 FFT 形式在本项目参数下不可用、偏差 1e8–1e10 与 M 无关"
>    → 实为**漏乘面积权重 `Δx′Δy′` 且频率未中心化**，已更正；
> 3. 旧版 4 用有限窗口能量替代原生 Parseval，并给出错误的期望公式 → 已更正。
>
> 旧结论的完整撤回清单见 `docs/revision_notes.md`。初审版的运行目录与报告
> 作为历史记录原样保留，未删除、未覆盖。

本文档把**论文原文的公式/参数**与**本项目的实现/测试参数**严格分开记录。
所有"原文"栏的页码均指 PDF 文件页码（正文页脚对应 `117:页码`）。
核实方式：用 PyMuPDF 以 zoom=3（≈216 dpi）渲染页面后用图像逐条判读，
**不以文本提取结果为准**（提取文本存在上下标丢失与字符替换，见第 6 节）。

---

## 1. 变量与单位

| 符号 | 含义 | 单位 | 代码位置 |
| --- | --- | --- | --- |
| λ | 真空波长 | m（配置里用 nm） | `FresnelConfig.wavelength` |
| z | 传播距离（DOE → 传感器，沿 +z） | m（配置里用 mm） | `FresnelConfig.distance` |
| k | 波数 2π/λ | rad/m | `FresnelConfig.wavenumber` |
| u₀, u₁, u₂ | 入射 / 经 DOE 后 / 传感器平面复振幅 | 任意（场量） | `fresnel_kernel_eval` 等 |
| A | 入射振幅 | 任意 | 输入数组 |
| φ₀ | 入射相位 | rad | 输入数组 |
| φ_h | DOE 引入的相位 | rad | **阶段 02 才实现** |
| h(x′,y′) | DOE 高度分布 | m | **阶段 02 才实现** |
| Δη_λ | 空气与基片折射率之差（原文下标在 η 上） | 无量纲 | **阶段 02 才实现** |
| D | 孔径/器件直径 | m（配置里用 mm） | `circular_aperture` |
| f | 焦距 | m（配置里用 mm） | `ideal_thin_lens_phase` |
| Δx′, Δy′ | 输入平面样点间距 | m | `Axis1D.d` |
| Δx, Δy | 输出平面样点间距 | m | 由 λz/(MΔx′) 推导 |
| M | 含零填充后的 FFT 长度 | 无量纲 | `FresnelConfig.padded_size` |
| N_F | Fresnel 数 D²/(4λf) | 无量纲 | 运行时计算并记录 |

**内部长度单位一律为米**；nm/μm/mm 只出现在 `optics/units.py` 与配置解析层。

### 坐标与数组约定（实现选择，论文未规定）

- 数学坐标：x 向右、y 向上、z 沿传播方向；Fourier 变换取**负指数**约定。
- 输入/输出数组 `a` 形状为 `(ny, nx)`，`a[j, i]` ↔ `(x = x_axis[i], y = y_axis[j])`。
- 一维轴以 0 为中心：`coords = (arange(n) − n//2) * d`，偶数 n 时 0 位于 `n//2` 号样点。
- **索引增加时 x、y 都增加**（旧版此处误写为"行指标增加时 y 减小"，与实现相反，
  2026-10-01 更正）。因此 matplotlib 显示用 `origin="lower"`，
  `extent` 取 `Grid2D.extent_pixel_um`（**像素边界**，中心坐标外扩半像元）。
- 图像 extent 的两种约定必须分开：`extent_center_um` 是**样点中心**范围，
  `extent_pixel_um` 是**像素边界**范围；画图与像元积分都用后者。
- 图像"上/下"与数学 y 的对应关系只在此处定义一次，后续阶段不得另行翻转。
- `x` 与 `y` 允许不同的样点数与间距（非方形网格），所有函数都按
  `grid.shape = (ny, nx)`、`grid.dx != grid.dy` 处理。

---

## 2. 论文公式逐条核对（原文核实）

| 式 | 页 | 原文形式 | 本阶段用途 |
| --- | --- | --- | --- |
| (1) | 3 | `u₀(x′,y′) = A(x′,y′) e^{iφ₀(x′,y′)}` | 输入复振幅的定义 |
| (2) | 3 | `u₁(x′,y′) = A(x′,y′) e^{i(φ₀+φ_h)}` | 输入复振幅形式；**不实现 φ_h** |
| (3) | 3 | `φ_h(x′,y′) = (2π/λ) Δη_λ h(x′,y′)` | 记录；阶段 02 实现 |
| (4) | 3 | `u₂(x,y) = e^{ikz}/(iλz) ∬ u₁(x′,y′) e^{i(k/2z)[(x−x′)²+(y−y′)²]} dx′dy′` | **本阶段实现的核心** |
| (5) | 3 | 式(4) 带入平面波 `A e^{i(φ₀+φ_h)}` | 记录 |
| (6) | 4 | `p_λ(x,y) ∝ |F[A e^{iφ_h} e^{i(π/λz)(x′²+y′²)}]|²` | **本阶段实现的 FFT 形式（IR）**；注意是 ∝，无归一化常数 |
| (7) | 4 | `Δφ_g = (2π/λ)(√(r²+f²) − f)`，`Δφ_h = (2π/λ)Δη_λ Δh(r)` | 记录；阶段 02 |
| (8) | 4 | `Δφ_g + Δφ_h = 2πn` | 记录；阶段 02 |
| (9) | 4 | `Δh(r) = [nλ − (√(r²+f²) − f)]/Δη_λ`，且 `−λ/Δη_λ ≤ Δh ≤ 0` | 记录；阶段 02 |
| (10) | 5 | `λ(θ) = λ_min + (λ_max−λ_min)(N/2π)θ`，`0 ≤ θ < 2π/N`，其后周期延拓 | 记录；阶段 02（N=3） |
| (11) | 5 | `Δh(r,θ) = [nλ(θ) − (√(r²+f²) − f)]/Δη_λ` | 记录；阶段 02 |
| (12) | 5 | `h(r,θ) = h(0,0) + Δh(r,θ)` | 记录；阶段 02 |
| (13) | 5 | `J_c(x,y) = ∭ Ω_c(λ) I_λ(μ,ν) p_λ(x−μ, y−ν) dμ dν dλ` | 记录；阶段 05 |
| (14) | 5 | `J_c(x,y) = ∫ Ω_c(λ) (I_λ * p_λ)(x,y) dλ` | 记录；阶段 05 |
| (15) | 5 | `J = Φ I` | 记录；阶段 05–06 |

**公式—页码对应**：(1)–(5) 第 3 页；(6)–(9) 第 4 页；(10)–(15) 第 5 页。
式(16)–(21) 在第 6 页，式(24) 在第 10 页（本阶段未使用）。

### 式(4) 与式(6) 的差别（本阶段必须补齐的部分）

- 式(4) 是**复振幅积分**：含 `e^{ikz}/(iλz)` 前因子、二次相位核与 `dx′dy′`。
- 式(6) 只写成 `∝ |F[·]|²`：**故意省略了全部常数**（取模平方后 `e^{ikz}/(iλz)`
  变成常数 `1/(λz)²`），也没有给出 `F` 的归一化约定。
- 因此**能量与尺度核验必须回到式(4) 补齐数值尺度**，不能直接用式(6) 的比例关系。
  本阶段实施的直接求值与 FFT(IR) 形式都以式(4) 为准（见第 3 节）。

---

## 3. 本阶段的传播实现

### 3.1 完整位移核求值（判据与交叉核验的基准）

> **修正提示（2026-10-01）：本节旧版把 `u₁` 预乘了一次输入平面二次相位，
> 属重复相位错误。正确形式见下，逐项更正理由见 `docs/revision_notes.md` R2。**

`optics.propagation.fresnel_kernel_matrix`：

```text
u₂(x,y) = e^{ikz}/(iλz) · Σ_{m,n} u₁(x′_m,y′_n)
          · e^{iπ((x−x′_m)²+(y−y′_n)²)/(λz)} · Δx′Δy′
```

- **被求和的只有 `u₁` 本身。** 位移核 `e^{iπ((x−x′)²+(y−y′)²)/(λz)}` 里已经含有
  `x′²+y′²`，再给 `u₁` 乘一次 `e^{iπ(x′²+y′²)/(λz)}` 等于额外插入一个光学元件，
  实测会让结果相对解析高斯解偏差 1.29369（正确形式为 1.4e−14）。
- 展开核的三部分（输入二次相位、交叉 Fourier 核、输出二次相位）只在
  **单次 FFT 路径**里各出现一次，两条路径不共用预处理。
- 实现上用可分离的两次矩阵乘法（BLAS），与显式双重循环等价但快一到两个数量级。
- 评价点可任意选取，不必落在任何 FFT 网格上——这是它能当基准的原因。

`optics.propagation.fresnel_kernel_separable` 是同一式的**可分离**写法：
展开核后 `(x−x′)²` 对 `x′` 只有线性耦合，于是双重求和化为两次一维矩阵乘法
`(Ey·u₁·Exᵀ)`，代价从 `O(N_out·Nx′Ny′)` 降到 `O(N_out·(Nx′+Ny′))`。
两者数学等价（实测相对 L2 = 2.1e−16），因此论文尺度 `D=1 mm` 的中心截线
（2401×2401 输入、959 个评价点）可以在 0.1 s 内精确求值。

**适用条件**（`sampling_diagnostics`）：判据必须覆盖**完整被积函数**的三部分相位，
而不是只看输入二次相位单项：

```text
输入二次相位步进 : π(2·x′_max·Δx′ + Δx′²)/(λz)
交叉项步进       : 2π(|x|_max·Δx′ + |y|_max·Δy′)/(λz)     ← 按实际输出窗口边缘估计
输出二次相位     : 只对输出坐标本身有影响，对求和变量没有步进
```

半宽按 `(n//2)·d` 计算（**不**额外乘 0.5）。保守提示阈值为各项之和 ≤0.5 rad，
且明确标注"这不是通用精度保证"。

### 3.2 单次 FFT 形式（论文式(6) 的写法）

> **修正提示（2026-10-01）：本节旧版的"该形式在本项目参数区间内不可用"结论
> 已撤回。真正原因是旧实现漏乘面积权重 `Δx′Δy′` 且频率数组未中心化。**

`optics.propagation.fresnel_fft`。把式(4) 的二次相位展开后：

```text
V  = 中心零填充后的 u₁ · e^{iπ(x′²+y′²)/(λz)}
F  = fftshift(fft2(ifftshift(V)))
u₂ = F · e^{iπ(x²+y²)/(λz)} · Δx′Δy′ · e^{ikz}/(iλz)
x  = λz·fftshift(fftfreq(Mx, Δx′)),  y = λz·fftshift(fftfreq(My, Δy′))
```

- **`Δx′Δy′` 不能省**：`fft2` 给出的是离散和，物理积分还要乘样点面积。
  旧版缺这一项，偏差恰好是 `1/(Δx′Δy′)`：Δx′=40 μm → 6.25e8，Δx′=10 μm → 1e10。
- **频率必须中心化**：`fftshift(fft2(ifftshift(·)))` 让索引与 `x = λz·fftfreq`
  的中心化坐标一一对应，避免两套索引约定混用。
- **奇数输入**：零填充偏移取 `(M − n + 1)//2`，保证输入的"0 号样点"落在填充数组的
  `M//2` 号样点；否则整体错半个样点（实测会出现 O(0.1) 的差异）。
- 输出采样由 `λz/(MΔx′)` 推导，不是手填成与输入相同的间距。

在该式正确实现后，它与 3.1 的完整位移核求和**在同一离散输入、同一原生输出网格上
是离散代数恒等关系**，实测相对 L2 ≤2.6e−13（旧版 6.25e8）。因此 3.1 与 3.2 互为
交叉核验，而不是"一个权威、一个失败"。

### 3.3 传递函数（谱）形式

`optics.propagation.fresnel_transfer_fresnel` 与
`optics.propagation.fresnel_angular_spectrum` 是两个**不同**的传播子，
名称与元数据里都必须区分：

```text
频域 Fresnel 近似 : H = e^{ikz}·e^{−iπλz(f_x²+f_y²)}
精确角谱          : H = e^{ikz·√(1−λ²(f_x²+f_y²))}     （含倏逝波指数衰减）
```

输出与输入同一网格，且都受 FFT 周期性回绕影响。

### 3.4 各路径的可靠性边界

| 路径 | 优点 | 已知限制 |
| --- | --- | --- |
| 完整位移核求值 | 无回绕，评价点任意 | 代价 `O(N_out·N_in)`，大网格需可分离写法 |
| 单次 FFT | 一次变换，输出间距 `λz/(MΔx′)` 由公式推出 | 受回绕影响；输出间距由 M 决定，小视场细采样需大 M |
| 频域 Fresnel 近似 | 输出与输入同网格 | 回绕；`H` 采样在粗网格上会被欠采样 |
| 精确角谱 | 无近轴近似 | 同上，且含倏逝波 |

本阶段的物理结论**同时**由「直接求值（A 组 Airy 中心截线）」与
「数值传播 vs 解析解（自由空间高斯）」支撑；Parseval 只作尺度核验。

---

## 4. 能量与尺度的正确关系

> **修正提示（2026-10-01）：本节旧版用"有限窗口对解析包围能量"替代了任务明确
> 要求的**原生 FFT 全输出 Parseval 检验**，并给出一条错误的期望公式
> `P_out = P_in/(λz)²·(ΔxΔy)/(Δx′Δy′)`。两者都已更正。**

**主判据：原生 FFT 全输出 Parseval。** 在**没有**任何窗口裁剪的原生输出网格上求和：

```text
P_in  = Σ|u₁|² · Δx′Δy′
P_out = Σ|u₂|² · Δx·Δy        （Δx = λz/(MΔx′)）
相对误差 |P_out/P_in − 1| ≤ 1e−10
```

实测 6.7e−15 … 2.2e−14。这条检验验证离散变换与**尺度因子（含 `Δx′Δy′`）**
是否正确：漏面积权重会让它偏差 `1/(Δx′Δy′)²` 量级，立刻暴露。

**为什么不需要"除以 (λz)² 再乘面积比"：** 正确实现里 `1/(iλz)` 与 `Δx′Δy′`
都已经在复场里，Parseval 就是同一场在两个平面上的功率相等，不需要任何额外因子。
旧版那条公式是把"漏掉的面积权重"错误地写成了理论期望。

**补充检查（不替代主判据）：有限窗口对解析包围能量。**

```text
离散: 窗内 Σ|u₂|²ΔA / 全输出 Σ|u₂|²ΔA
解析: 1 − J₀²(u) − J₁²(u)，u = πDr/(λf)
```

两者定义不同（解析式是无限平面上的占比，而离散场在窗内尚未积完），因此用
**绝对差**而不是相对差作容差。本次运行在 r = 5×第一暗环处绝对差约 3.0e−2。

**边界说明**：Parseval 类检验主要检验离散变换与尺度因子，**不能单独证明无混叠**：
混叠会把能量搬到错误的样点上而总量不变。因此本阶段同时保留 A 组 Airy 解析对照、
B 组离散恒等与测试中的自由空间高斯解析对照。

---

## 5. 参数对照：论文参数 vs 本阶段验证参数

### 5.1 论文参数（原文核实，第 6–7 页）

| 参数 | 值 | 出处与原文措辞 |
| --- | --- | --- |
| DOE 直径 | 1 mm | 第 7 页 "the fabricated DOE (its diameter is 1 mm and its focal length is 50 mm)"；**原文未出现 "aperture" 一词** |
| 焦距 | 50 mm | 第 7 页 "its focal length is 50 mm"；图 6 图注 "at 50 mm focal length"；**原文未写 "distance to sensor"** |
| 材料 | fused silica（熔融石英） | 第 6 页 "a 0.5mm thick 4-inch fused silica wafer with both sides polished" |
| 基片厚度 | 0.5 mm | 同上（是晶圆厚度，不是浮雕高度） |
| 制版分辨率 | 1 μm | 第 6 页 "masks with 1μm resolution by a high resolution direct laser writer" |
| 量化级数/步高 | 16 级、100 nm | 第 6–7 页 "16-level … four iterations"、"depth interval for each stair is 100 nm" |
| 工作波段 | 420–660 nm | 第 5 页式(10)；第 7 页 "25 wavelength channels … 420 nm to 660 nm" |
| 相机 | Canon EOS 5D Mark III，5760×3840，像元 6.22 μm | 第 7 页 |
| 标定物距/针孔 | 8.03 m、1 mm 针孔 | 第 7 页 |
| 图 3 波长 | 420,450,480,510,540,570,600,630,660 nm（9 个） | 第 4 页图注；明确写 "by simulation" |
| 图 6 | 实测 PSF，420–650 nm | 第 7 页图注，明确写 "fabricated / measured" |
| 仿真工具 | LightPipes | 第 5 页脚注 |

### 5.2 本阶段验证参数（**不是论文参数**）

| 用途 | 参数 | 说明 |
| --- | --- | --- |
| A 组 Airy 尺度 | D=0.1 mm、f=50 mm、λ=550 nm | 缩小尺度验证参数。选它的关键约束是 **Fresnel 数 N_F=D²/(4λf)=0.091≪1**：只有远场条件下焦平面图样才近似解析 Airy。 |
| A 组输入网格 | ±200 μm，Δx′=1 μm，N=400 | 孔径半径 50 μm 的 4 倍，保证场在边缘已衰减 |
| A 组观察网格 | ±1200 μm，Δx=4 μm，601×601 | 第一暗环 335.5 μm，约 84 样点/暗环 |
| B 组直接 vs FFT | N=48、Δx′=40 μm、z=50 mm、M=96 | 用于演示 IR 形式在**不满足判据**时的实际偏差；另附 N=24、Δx′=10 μm 的满足判据算例作对照 |
| C 组能量 | D=0.1 mm、f=50 mm；窗 ±1200 μm；扩展网格 ±8 mm | 与 A 组同参数，便于交叉核对 |
| D 组采样 | 输入窗口固定 ±200 μm，N=100/200/400 | **只改输入采样精度**，孔径与观察区不变 |

**论文尺度参照（仅解析核对，不做二维求值）**：D=1 mm、f=50 mm、λ=550 nm
→ 1.22λf/D = 33.550 μm。注意该组合的 **Fresnel 数 ≈4.5 > 1**，作者仿真并不处于
远场 Airy 极限；本阶段因此只把它当作解析尺度检查，并记录同等分辨率二维求值的
采样/内存估计（Δx′≈1.68 μm、N≈1074、单个 complex128 数组约 18 MB）。

---

## 6. 来源清单与核实方式

| 文件 | 路径 | 字节数 | SHA256 | 页数 |
| --- | --- | --- | --- | --- |
| 正文 | `work/papers/main.pdf` | 13269896 | `493f49f1a998dd51a60757f8b121bd55b03fcfcb692d8828cc08f0c7d718bdcc` | 13 |
| 补充材料 | `work/papers/supplement.pdf` | 59342087 | `638d42815b92be6fa101dfd144f0f290abd61e8ec18346194f4dc223ceeb5bb9` | 19 |

- 作者项目页（**仅登记 URL，本阶段未做网络下载**）：
  - https://vccimaging.org/Publications/Jeon2019Hyperspectral/
  - https://mail.minhkim.org/siggraph2019/index.html
  - DOI：https://doi.org/10.1145/3306346.3322946
- 本轮未找到可验证的**官方代码**入口（此前检索到的第三方列表把链接标为"code"，
  实际仍指向作者论文项目页），因此本阶段没有可对照的官方实现。
- 核实方式：PyMuPDF 1.28.2 以 zoom=3 渲染第 3–7 页与补充第 4 页，
  逐条判读公式与参数；渲染图与判读脚本保存在 `work/papers/stage01_check/`
  （只新增文件，未删除或覆盖任何既有文件）。

**文本提取的已知缺陷**（因此不以提取文本为准）：
`Δφ_g` 被提取成西里尔字母 `Δϕд`；`Δh(r) := h(r)−h(0)` 的 `:=` 被提取成 `B`；
式(3) 中 `Δη_λ h(x′,y′)` 的分隔丢失。这些都由页面图像判读纠正。

---

## 7. 代码函数 ↔ 公式对应表

| 公式/量 | 代码 |
| --- | --- |
| 式(4) 完整位移核求值（矩阵形式） | `optics.propagation.fresnel_kernel_matrix`（旧名 `fresnel_kernel_eval` 保留为别名） |
| 式(4) 的可分离写法（数学等价） | `optics.propagation.fresnel_kernel_separable` |
| 式(6) FFT 形式（单次 FFT） | `optics.propagation.fresnel_fft` |
| 频域 **Fresnel 近似**传递函数 | `optics.propagation.fresnel_transfer_fresnel` |
| **精确角谱**（与上者不同） | `optics.propagation.fresnel_angular_spectrum` |
| 输入平面二次相位 | `optics.propagation.input_plane_quadratic_phase` |
| 输出平面二次相位 | `optics.propagation.output_plane_quadratic_phase` |
| 输出间距 λz/(MΔx′) | `fresnel_fft` 中的 `x = λz·fftshift(fftfreq(Mx, Δx′))` + `meta["output_coordinate_rule"]` |
| 完整被积函数的采样判据（三项） | `optics.propagation.sampling_diagnostics` |
| 单项二次相位步进 | `optics.propagation.quadratic_phase_max_step_rad` |
| 精确高阶光程差 | `optics.propagation.path_difference_exact_m` |
| 四次近似光程差（对照用） | `optics.propagation.path_difference_leading_m` |
| 高阶光程相位误差 | `optics.propagation.third_order_phase_error_rad` |
| Fresnel 数（两种定义） | `optics.propagation.fresnel_number` / `axial_fresnel_number` |
| 理想薄透镜二次相位（验证用） | `optics.coordinates.ideal_thin_lens_phase` |
| Airy 解析参考 | `optics.metrics.airy_intensity` / `airy_dark_ring_radii` / `airy_encircled_energy_analytic` |
| 近轴高斯解析参考 | `optics.metrics.tilted_gaussian_paraxial_solution`（**注意**：倾斜情形的解析相位
  在本阶段仍与数值结果有 0.48 的相对残差，未作为证据，见 `docs/reproduction_log.md` 第 10.3 节） |
| 误差指标 | `optics.metrics.relative_l2_complex` / `normalized_l1_intensity` / `remove_global_phase` |
| 暗环定位 | `optics.metrics.find_dark_rings` / `find_local_minima` |

> 初审版表中的 `direct_fresnel_integral`、`fresnel_propagate`、
> `fresnel_propagate_transfer`、`ir_form_validity`、`direct_sum_validity`、
> `transfer_form_validity` **已随修正一并移除**（它们绑定在错误的公式与判据上）。
> 需要的功能分别由 `fresnel_kernel_matrix`、`fresnel_fft`、
> `fresnel_transfer_fresnel` / `fresnel_angular_spectrum`、`sampling_diagnostics`
> 承担。

**尚未实现（后续阶段）**：式(3)、(7)–(12) 的 DOE 高度与相位；
式(13)–(15) 的非相干成像前向模型与重建。
