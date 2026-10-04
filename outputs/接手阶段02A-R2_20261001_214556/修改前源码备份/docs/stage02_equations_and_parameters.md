# Jeon2019 阶段 02A：物理公式、参数出处与文献追溯

## 1. 论文核心公式与文献页码定位

主文献：Jeon et al., "Compact Snapshot Hyperspectral Imaging with Diffracted Rotation", ACM Trans. Graph. (SIGGRAPH 2019), Vol. 38, No. 4, Article 117. 本地有效文件：`work/papers/main.pdf`。

### 1.1 高度差引起的相位延迟：式 (3)（PDF 第 3 页）
$$\phi_h(x', y') = \frac{2\pi}{\lambda} \Delta\eta_\lambda h(x', y')$$
- **说明**：$\Delta\eta_\lambda = n_{\text{substrate}}(\lambda) - n_{\text{air}}$ 为基底材料与空气在波长 $\lambda$ 下的折射率差。
- **实现约定**：内部长度一律为米（m）；空气折射率 $n_{\text{air}} = 1.0$。

### 1.2 Fresnel 衍射传播积分：式 (4)（PDF 第 3 页）
$$u_2(x, y) = \frac{e^{ikz}}{i\lambda z} \iint u_1(x', y') \exp\left[ \frac{ik}{2z} \left((x - x')^2 + (y - y')^2\right) \right] dx' dy'$$
- **说明**：$k = 2\pi / \lambda$ 为波数。
- **离散实现**：采用阶段 01 验证通过的完整位移核可分离矩阵求积 `fresnel_kernel_separable`，保留严谨面积权重 $\Delta x'\Delta y'$、振幅因子 $1/(i\lambda z)$ 与全局相位 $e^{ikz}$。

### 1.3 几何光程差与相长干涉聚焦条件：式 (7)–(8)（PDF 第 4 页）
$$\Delta\phi_g = \frac{2\pi}{\lambda} \left(\sqrt{r^2 + f^2} - f\right), \quad \Delta\phi_h = \frac{2\pi}{\lambda} \Delta\eta_\lambda \Delta h(r)$$
$$\Delta\phi_g + \Delta\phi_h = 2\pi n \quad (n \in \mathbb{Z})$$
- **说明**：式 (8) 中的 $n$ 为相长干涉聚焦时的绕回整数阶数（代码中统一命名为 `wrap_order`），与材料折射率严格区分。
- **数值稳定形式**：$\delta(r) = \sqrt{r^2 + f^2} - f = \frac{r^2}{\sqrt{r^2 + f^2} + f}$，消除小半径下的浮点相消灾难。

### 1.4 高度轮廓与 $2\pi$ 相位包裹：式 (9)（PDF 第 4–5 页）
$$\Delta h(r) = \frac{n\lambda - \left(\sqrt{r^2 + f^2} - f\right)}{\Delta\eta_\lambda}$$
$$-\frac{\lambda}{\Delta\eta_\lambda} \le \Delta h(r) \le 0$$
- **说明**：通过令 $n = \lfloor \delta / \lambda \rfloor$，将高度严格限制在 $[-\lambda / \Delta\eta_\lambda, 0]$ 区间内。中心 $r=0$ 处 $\delta=0, n=0, \Delta h = 0$。

### 1.5 各向异性螺旋设计波长匹配：式 (10)（PDF 第 5 页）
$$\lambda(\theta) = \begin{cases} \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \frac{N}{2\pi} \theta, & 0 \le \theta < \frac{2\pi}{N} \\ \lambda\left(\theta - \frac{2\pi}{N}\right), & \theta \ge \frac{2\pi}{N} \end{cases}$$
- **说明**：极角 $\theta = \text{mod}(\text{atan2}(y', x'), 2\pi) \in [0, 2\pi)$。$N$ 为翼数/周期数。每个扇区内波长从 $\lambda_{\min}$ 线性增加至 $\lambda_{\max}$。

### 1.6 连续 DOE 全局高度：式 (11)–(12)（PDF 第 5 页）
$$\Delta h(r, \theta) = \frac{n\lambda(\theta) - \left(\sqrt{r^2 + f^2} - f\right)}{\Delta\eta_\lambda}$$
$$h(r, \theta) = h(0, 0) + \Delta h(r, \theta)$$
- **说明**：固定器件使用相对高度 $\Delta h$ 计算相位调制。
- **折射率分工**：
  1. 高度设计阶段：式(11)中分母的折射率差采用对应局部设计波长的折射率：$n_{\text{design}}(\theta) = n(\lambda_{\text{design}}(\theta))$；
  2. 光场传播阶段：式(2)–(3)中相位调制采用实际入射单色光波长的折射率：$n_{\text{in}} = n(\lambda_{\text{in}})$。

---

## 2. 材料色散模型与来源

- **模型**：Malitson (1965) 熔融石英（Fused Silica）三项 Sellmeier 公式
- **文献来源**：I. H. Malitson, "Interspecimen Comparison of the Refractive Index of Fused Silica," J. Opt. Soc. Am. 55(10), 1205-1209 (1965). DOI: `10.1364/JOSA.55.001205`.
- **公式**（$\lambda$ 以 $\mu\text{m}$ 计）：
  $$n^2 - 1 = \frac{0.6961663 L^2}{L^2 - 0.0684043^2} + \frac{0.4079426 L^2}{L^2 - 0.1162414^2} + \frac{0.8974794 L^2}{L^2 - 9.896161^2}$$
- **模型理论核算数值**（由上述公式直接求得，非实测批次标定）：
  - 420nm: $n = 1.468093690040$
  - 540nm: $n = 1.460343603077$
  - 550nm: $n = 1.459910886469$
  - 660nm: $n = 1.456268423490$

---

## 3. 阶段 02A 参数配置表

| 参数名 | 符号 / 键名 | 阶段 02A 默认值 | 物理单位 | 来源与性质 |
| :--- | :--- | :--- | :--- | :--- |
| 孔径直径 | $D$ / `diameter_m` | 1.0e-3 (1.0 mm) | m | 正文第 7 页实验与标定描述；仿真采用值 |
| 焦距 | $f$ / `focal_length_m` | 5.0e-2 (50.0 mm) | m | 正文第 7 页光学系统配置 |
| 传播距离 | $z$ / `distance_m` | 5.0e-2 (50.0 mm) | m | 焦平面成像设定（$z = f$） |
| 螺旋翼数 | $N$ / `wings_N` | 3 | 无量纲 | 正文第 5 页设计推荐值（$N=3$ 重建效果最优） |
| 设计波长下限 | $\lambda_{\min}$ / `design_wavelength_min_m` | 4.2e-7 (420 nm) | m | 正文第 5 页式 (10) 说明 |
| 设计波长上限 | $\lambda_{\max}$ / `design_wavelength_max_m` | 6.6e-7 (660 nm) | m | 正文第 5 页式 (10) 说明 |
| 控制波长 | $\lambda_0$ / `control_wavelength_m` | 5.5e-7 (550 nm) | m | 正文第 4 页传统单色透镜设计例 |
| 预览波长 | - / `preview_wavelengths_m` | 420, 540, 660 nm | m | 图 3 典型波长代表采样点 |
| 输入网格 | - / `grid.input` | $1101 \times 1101$, 半宽 $550\,\mu\text{m}$, 间距 $1\,\mu\text{m}$ | m | 完整覆盖 $1\,\text{mm}$ 孔径的离散求积网格 |
| 输出网格 | - / `grid.output` | $301 \times 301$, 半宽 $150\,\mu\text{m}$, 间距 $1\,\mu\text{m}$ | m | 三波长共同探测器物理坐标窗口 |
