# 阶段 01 第二轮修正说明（S1–S8 逐项）

> **后续接手说明：** 本文件记录DSH中止前的工作，下面的阶段状态及能量算法说明属于历史版本。2026-10-01由审核者按用户授权补完、优化并重新验证，当前完成情况见 [stage01_completed.md](stage01_completed.md)。尤其“只有FFT能提供包围能量分母”的旧解释已更正：理想无损模型可直接以输入功率为分母，有限输出窗口足以计算圆盘分子。

日期：2026-10-01（第二轮）。
依据：`outputs/复审_阶段01_20261001/阶段01_第二轮审核报告.md` 与
`阶段01_DSH第二轮修正任务_详细版.md`。

> **本轮结论：默认 Fresnel 传播核心在第一轮已修好，但阶段整体仍未通过。**
> 第二轮审核给出 8 项必须修正的问题（S1–S8），本文件逐项说明**怎么修、实际执行了什么、
> 数值是多少、结果文件在哪、还有什么未验证**。
> 第一轮的修正说明见 [`docs/revision_notes.md`](revision_notes.md)（仍然有效）。

---

## 0. 本轮实际修改的文件与实际运行的命令

### 0.1 修改的文件（局部编辑，未整批重写、未删除任何文件）

| 文件 | 改动要点 |
| --- | --- |
| `optics/propagation.py` | S1 reduced 场去掉多余的 `exp(−iq)`；S5 两种谱传播的全局相位开关；S7 采样诊断重写 |
| `main.py` | S3 绝对包围能量口径与 A/C 组图注；S2 全新的 D 组三类收敛；S4 验收汇总与状态语义；S6 配置 show/save 接线；S1 证据生成 |
| `config.json` | 新增 `groups`、`energy_levels`、`energy_evaluation`、D 组新参数；阈值按指标分开命名 |
| `optics/runutil.py` | `allocate_run_dir` 支持显式 `root`，默认仍按项目 `__file__` 解析 |
| `tests/test_propagation.py` | S1/S5/S7/S8 的新测试；重写 `test_two_padding_factors`（真正执行 FFT） |
| `tests/test_negative_paths.py` | **新增**：S4.3 的 7 项负面测试 |
| `docs/verify_config_switches.py` | **新增**：S6 的 mock 后端开关接线验证 |
| `README.md` | S6 的最终设置合成规则、路径解析规则、状态语义 |

修改前的可追溯快照（全部生产文件的 SHA256、环境）：
`work/stage01_round2/pre_edit_sha256.json`、`work/stage01_round2/environment_pre.json`。

### 0.2 实际运行的命令

```powershell
# 1) 单元测试
cd D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics
python -B -m unittest discover -s tests -p "test_propagation.py" -v      # 32 项
python -B tests\test_negative_paths.py                                   # 7 项
python -B docs\verify_config_switches.py                                 # 5 项接线检查

# 2) 默认完整运行（项目目录内）
python -B main.py

# 3) 从项目之外的目录用绝对路径运行（验证默认路径与 cwd 无关）
Set-Location D:\PyCharmProjects\Jeon2019
& 'D:\dev\python\python3.10.4\python.exe' -B `
  'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\main.py'

# 4) 成功子集 / 失败子集 / 关闭保存
python -B main.py --only direct                 # 成功子集 → partial，退出码 0
python -B main.py --only direct --no-save       # 关闭保存 → 项目输出零变化
```

### 0.3 送审的完整运行

**`D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\results\stage01\run_20261001_160653`**

总体状态 `pass`，退出码 0，四项必需组全部实际测量。

---

## S1 [P1] reduced 场多乘了一个负二次相位

**问题**：完整场是 `u_full = C·F·exp(+iq)`，因此 reduced 应当是 `C·F`；旧代码返回
`C·F·exp(−iq)`，与完整场相差 `exp(−2iq)`。

**怎么修**（`optics/propagation.py::fresnel_fft`）：`omit_output_quadratic_phase=True`
分支**只返回 `C·F`**，不再乘任何输出二次相位；元数据改为明确写出场表示与关系式::

    representation          = "reduced 场 u_red = C·F（未乘输出二次相位）"
    representation_relation = "u_full = u_red · exp(+i·q_out)，q_out = π(x²+y²)/(λz)"
    output_quadratic_phase_present = False

**纠正了旧解释**：“相位上百 π 不能逐点检验”**不成立**。已知坐标上的指数能可靠计算；
欠采样限制的是对连续相位的**解析/插值**，不妨碍在给定坐标上求值。旧测试因此被替换为
用**全场范数**的准确关系断言。

**实际数值**（`results/…/metrics/phase_representation_metrics.json`）：

| 算例 | relL2(u_red·e^{iq}, u_full) | relL2(u_red, u_full·e^{−iq}) | 强度相对差 | 反例（补两次相位） |
| --- | ---: | ---: | ---: | ---: |
| N=24、Δx′=10 μm、M=48 | 5.12e−14 | 5.12e−14 | 2.63e−16 | 1.4052（必须失败） |
| 非方形 20×26、dx=8 μm、dy=15 μm | 3.24e−14 | 3.24e−14 | 2.91e−16 | 1.3996 |
| N=24 且关闭全局相位 | 5.12e−14 | 5.12e−14 | 1.77e−16 | 1.4052 |

阈值 1e−10（实际小 4 个数量级）。**反例项**证明测试确实能拦截旧实现。

**结果文件**：`arrays/phase_representation_check.npz`（full、reduced、x_m、y_m、q_rad）、
`metrics/phase_representation_metrics.json`。

---

## S5 [P2] 精确角谱忽略全局相位开关

**怎么修**（`optics/propagation.py::_spectral_propagate`）：

```text
频域 Fresnel 近似：H_true = exp(ikz)·exp(−iπλz f²)     H_false = exp(−iπλz f²)
精确角谱        ：s = √(1−λ²(fx²+fy²))（复支路取 z>0 衰减）
                  H_true = exp(ikz·s)                  H_false = exp(ikz·s)·exp(−ikz)
```

`include_global_phase` **只**控制 `exp(ikz)`：不去掉 Fresnel 的 `1/(iλz)`，也不改变强度。
元数据新增 `include_global_phase` / `global_phase_factor`；角谱另记
`periodic_wrap_note`（谱方法隐含周期边界，零填充只推迟回绕）与
`evanescent_bins` / `evanescent_note`（`λ²f²>1` 的频点按复平方根衰减，
因此**不宣称**全输出严格无损；Parseval 主检查只针对 Fresnel 频域形式单独执行）。

**实际数值**（单位均匀平面波、不填充，关闭开关应得 1）：

| 方法 | `‖u_false − 1‖∞` | `relL2(u_true, exp(ikz)·u_false)` |
| --- | ---: | ---: |
| 频域 Fresnel 近似 | 0.0 | 0.0 |
| 精确角谱 | 1.13e−16 | 1.57e−16 |

非对称输入下同样 ≤1e−10（测试 `test_spectral_global_phase_switch_both_methods`）。

---

## S8 [P2] 倾斜高斯解析参考缺包络平移

**怎么修**（`tests/test_propagation.py::free_space_gaussian`）：改为完整解析复场

```text
q        = 1 + i·zλ/(π w0²)
u(x,y,z) = exp(ikz)/q · exp{−[(x−zθx)²+(y−zθy)²]/(w0² q)}
           · exp{ik[θx·x + θy·y − z(θx²+θy²)/2]}
```

即包络中心**平移到 (zθx, zθy)**。θ 统一作为**近轴线性相位系数**，
不混用 `tanθ`、`sinθ` 的精确几何式。

**实际数值**（直接核 vs 完整解析复场，相对 L2）：

| 倾斜 | 本轮实测 | 审核给出的旧参考误差 |
| --- | ---: | ---: |
| (0, 0) | 1.21e−14 | （原有正确解） |
| (1 mrad, 0) | 1.19e−14 | 0.48395 → 补平移后 5.88e−16 |
| (0, 1 mrad) | 1.19e−14 | — |
| (1 mrad, −0.7 mrad) | 1.17e−14 | — |

另加测试断言包络峰值确实落在 `(z·θx, z·θy)`、θ=0 退化为原正确解、绝对幅度也对。

---

## S3 [P1] 包围能量用错分母，图注与计算不一致

**怎么修**（`main.py::encircled_energy_record`）：统一**唯一**分母口径

```text
P_in       = Σ |u_in|² Δx′Δy′
P_disk(R)  = Σ_{x²+y²≤R²} |u_out|² Δx_outΔy_out
E_numeric  = P_disk(R)/P_in        ← 与解析 1−J₀²−J₁² 同口径
```

并做了三处结构性修正：

1. **总功率改用单次 FFT 原生全输出**。理由：解析式的分母是**无限平面**总功率，
   而可分离直接求值只在有限窗内求和，焦平面尾部衰减很慢（本阶段实测在 20·r1 处
   仍有约 2% 的标定偏差），无法给出可比的分母。FFT 的原生全输出功率严格守恒
   （实测相对误差 ≤2.2e−14）。
2. **A 组的能量评价窗口独立且足够大**：不再用 `2.5r1` 的方窗去测 `5r1` 的圆盘
   （那会让所有像素被选中、比例恒为 1，是空检查）。现在 R 与窗口由配置分别给出，
   并强制判据 `R/半宽 ≤ 0.5`、`P_square < 0.9999·P_total`。
3. **累计曲线用同一个 `P_in` 分母**，图上同时画出 `P_disk/P_total` 作为对照；
   图注不再出现“未重新归一化”却除以有限窗口和的情况。

**实际数值**（`metrics/energy_absolute_metrics.csv`）：

| 等级 | R (μm) | 输出半宽 (μm) | R/半宽 | $E_{numeric}$ | $E_{analytic}$ | 绝对差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: D=0.1 mm | 1342 | 27506 | 0.049 | 0.959303 | 0.959405 | **1.02e−04** |
| A: D=0.2 mm | 335.4 | 13753 | 0.024 | 0.959046 | 0.959405 | **3.59e−04** |
| A: D=1 mm（论文尺度） | 134.2 | 13753 | 0.0098 | 0.959367 | 0.959405 | **3.85e−05** |
| C2: M=4096 | 134.2 | 13753 | 0.0098 | 0.959121 | 0.959405 | **2.84e−04** |
| C2: M=8192 | 134.2 | 13753 | 0.0098 | 0.959367 | 0.959405 | **3.85e−05** |

- 阈值 0.005，实测最差 3.85e−04（比阈值小一个数量级）。
- **网格稳定性**：R=4r1 处 M 由 4096 → 8192，绝对差 2.84e−04 → 3.85e−05，
  单调改善；各级之间最大与最小之差 2.5e−04 < 0.005，判为稳定。
- 与审核独立值一致：审核对同一已保存数组按正确口径重算得 0.967471360
  （R=5r1），解析 0.967472476，绝对差 1.12e−6。本轮 R 取 4r1（见下）。
- **为什么本轮主水平取 R=4r1 而不是审核示例的 5r1**：R 越大，圆盘越接近输出半宽，
  `R/半宽 ≤ 0.5` 的窗口判据越难同时满足（5r1 需要输出半宽 ≥10r1 ≈ 1.4 mm，
  而 FFT 原生输出半宽由 `M` 与输入网格共同决定）。任务书只要求“选用较小且完全包含的
  R，或新增较大的独立能量评价窗口”，因此取 R=4r1 并显式记录 R/半宽与覆盖判据。
  两者都是同一口径的合法评价点。
- 有限方窗内条件占比 `P_disk/P_square` 已**单独命名**保存，不与无限平面解析式混用。

**结果文件**：`metrics/energy_absolute_metrics.csv`、
`arrays/energy_absolute_*.npz`、`arrays/energy_encircled_*.npz`（径向累计功率
`P_cum(r)` 与 P_in/P_disk/E_numeric/E_analytic；**不**保存整幅 M×M 强度数组，
因为能量口径完全由径向累计决定，保存 8192² 数组只会浪费数百 MB）、
`figures/C_energy.png`（左：`P_disk(r)/P_in` 与解析曲线同图；右：两者之差）。

---

## S2 [P1] 真正的输入/填充收敛仍未完成

**旧问题**：D 组的 `D` 没进入输入场构造；各级只比较“FFT vs 同一离散输入的直接核”，
两者共同欠采样也能精确相等；`worst_pair_L1` 实际存的是复场 L2；
`test_two_padding_factors` 的辅助函数完全没用 `pad`。

**怎么修**：把三类**不同指标**彻底分开（`main.py::study_sampling`）。

### D1 输入采样收敛（固定器件与公共输出坐标）

- 器件：**理想圆孔 D=0.1 mm 真正进入孔径函数** × 理想薄透镜，`f=z=50 mm`，λ=550 nm。
- 输入半宽固定 ±200 μm；间距 2 / 1 / 0.5 μm；每级都按**同一个连续函数**取样
  （`circular_aperture` + `ideal_thin_lens_phase`），不把粗网格图像放大。
- 公共输出坐标：±2·r1（r1=335.409 μm），间距 2 μm，673×673，**所有等级共用同一组坐标**。
- 在公共坐标上用完整位移核的可分离等价形式求值。

| 等级 | N | Δx′ (μm) | $P_{in}$ | 孔径离散面积相对差 | 截线对解析 Airy 的 L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dx2.0um | 201 | 2.0 | 7.8440e−09 | −1.271e−03 | 1.6786e−03 |
| dx1.0um | 401 | 1.0 | 7.8450e−09 | −1.144e−03 | 8.4165e−04 |
| dx0.5um | 801 | 0.5 | 7.85425e−09 | +3.417e−05 | 2.9515e−04 |

| 相邻等级 | **原始强度相对 L1**（主指标） | 反向 L1 | 形状 L1 | 复场相对 L2 |
| --- | ---: | ---: | ---: | ---: |
| 2.0 → 1.0 μm | **3.670847e−03** | 3.672762e−03 | 3.695921e−03 | 5.741755e−03 |
| 1.0 → 0.5 μm | **2.606085e−03** | 2.609721e−03 | 1.892266e−03 | 2.616823e−03 |

- 主指标定义：`raw_intensity_L1(a,b) = Σ|Ia−Ib|ΔA / ΣIbΔA`，**保留绝对通量信息**。
- 按**预先约定**用最后一级对判定：2.606e−03 ≤ 0.02 → **通过**。
- 与审核独立预核验一致：审核给 2→1 μm 为 0.003666、1→0.5 μm 为 0.002605。

### D2 输出采样（填充）收敛（固定同一离散输入，真正改变 M）

- 输入固定为 D1 的 Δx′=1 μm 那一级（N=401，P_in=7.8450e−09）。
- `fresnel_fft` 用 **M=1024** 与 **M=2048**，两次都确认原生输出间距按 `λz/(MΔx′)` 改变。
- 原生网格上各自做离散恒等检查（归入算法检查）。
- 把原生**强度**双线性插值到公共探测器坐标，与该输入在公共坐标上的完整核参考比较；
  越界点显式剔除，不补零。

| 等级 | M | 原生点数 | 原生间距 (μm) | 离散恒等 rel L2 | 功率误差 | **公共参考原始强度 L1** | 形状 L1 | 窗口通量偏差 | 耗时 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1024 | 1024 | 1024 | 26.86 | 1.56e−11 | 1.03e−14 | **8.928e−03** | 8.95e−03 | +3.71e−05 | 1.18 |
| M2048 | 2048 | 2048 | 13.43 | 1.56e−11 | 9.66e−15 | **2.240e−03** | 2.24e−03 | +9.27e−06 | 0.96 |

- 两个 M 在公共网格上的实际 L1 = **6.767e−03**（452929 点）。
- 最细级 2.240e−03 ≤ 0.02 → **通过**；更细填充未变差（改善 4 倍）。
- 与审核独立预核验一致：M1024 为 0.008927、M2048 为 0.002240、两个 M 之间为 0.006766。

### D3 离散算法恒等（与前两者分开）

| 算例 | N | Δx′ (μm) | M | rel L2 | 功率误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity_N48_dx40um_M96 | 48 | 40 | 96 | 1.074e−13 | 6.66e−15 |
| identity_N24_dx10um_M48 | 24 | 10 | 48 | 1.699e−14 | 8.88e−16 |
| identity_N64_dx5um_M256 | 64 | 5 | 256 | 2.515e−13 | 2.15e−14 |

**三种指标与阈值互不相同**：`discrete_identity_rel_l2_max`（复场相对 L2，只用于算法恒等）、
`input_sampling_intensity_L1_max`、`padding_sampling_intensity_L1_max`（公共物理网格上的
原始强度相对 L1）。复场 L2 **不会**被塞进名为 L1 的字段。
`test_two_padding_factors` 已重写：实际执行 `fresnel_fft`，断言原生间距按公式改变、
原生网格离散恒等、公共网格 L1 及“更细不变差”。

**结果文件**：`metrics/sampling_input_levels.csv`、`metrics/sampling_padding_levels.csv`、
`metrics/sampling_identity_levels.csv`、`arrays/sampling_common_grid.npz`
（公共 x/y、各级 u 与 I、公共参考 u/I、原生尺寸），
`figures/D_sampling_convergence.png`、`figures/D_padding_common_grid.png`。
JSON 里 `input_sampling_L1` 与 `padding_sampling_L1` 分开保存。

---

## S7 [P2] 采样诊断仍混用输入与输出间距

**怎么修**（`optics/propagation.py::sampling_diagnostics`）：接口显式接收输出间距
（`dx_out`/`dy_out`），**没有数据时返回“未评估”**（`None`），绝不用输入间距冒充。
并且把“对积分变量的采样”和“输出表示的采样”彻底分开，x/y 各自报告：

```text
输入积分方向（核界，x/y 分别）：
  input_kernel_step_x = π[2(a_x + b_x)Δx′ + Δx′²]/(λz)      a=输入支撑半宽，b=输出窗口半宽
  input_kernel_step_y = π[2(a_y + b_y)Δy′ + Δy′²]/(λz)
  input_kernel_step_worst = max(x, y)          ← ok 与阈值判据用这个
输出表示方向（x/y 分别，需实际输出间距）：
  output_quadratic_step_x = π[2·b_x·Δx_out + Δx_out²]/(λz)
  output_quadratic_step_y = π[2·b_y·Δy_out + Δy_out²]/(λz)
  未给输出间距 → None + output_quadratic_evaluated=False
```

- 明确标注输出二次相位**对固定输出点只是常量、对求和变量没有步进**，
  不能当作输入求积的变化项。
- 支撑范围按整张方形输入窗半宽计，用 `support_is_conservative_upper_bound=True`
  注明这是**保守上界**；近轴高阶误差超提示阈值时只解释近轴条件，
  不用它误判焦平面 Airy 不适用（A 组仍以解析 Airy 对照与收敛结果为准）。
- 保持已经正确的精确高阶光程差公式与独立数值基准不变。

**测试**：新增/重写 4 项

| 测试 | 断言 |
| --- | --- |
| `test_sampling_diagnostics_uses_output_spacing_not_input` | 不给输出间距时为“未评估”；改变输出间距时输出步进跟随改变（8 μm 相对 2 μm 至少 3.9 倍）；输入项不受输出间距影响 |
| `test_sampling_diagnostics_rectangular_dy_larger` | 矩形网格 dy=8Δx 时 `input_kernel_step_y ≥ 5×input_kernel_step_x`，worst 取 y，且与核界公式一致到 1e−12 |
| `test_sampling_diagnostics_output_not_used_as_input_term` | 输出项与输入项是两个独立字段，不相等 |
| `test_sampling_diagnostics_covers_three_terms` | 三项 + 精确高阶光程差齐备；半宽按 `(n//2)·d` 不乘 0.5 |

---

## S4 [P1] 总验收会漏掉真实失败

**怎么修**：

1. **结果键统一**：`results['direct']`（配置新增 `groups` 段声明
   `airy/direct/energy/sampling` 四组的 `required` 与中文标签），报告与验收全部统一用
   `direct`，不再出现 `direct_vs_fft` 与 `direct` 两套名字。
2. **Airy 逐例判定**：每个算例分别判定暗环与截线（`per_airy_case` 存机器可读明细），
   总体取**所有算例的最差值**并保存**最差算例名**。
3. B 组各规模、D3 恒等、原生 FFT Parseval、绝对包围能量、输入收敛、填充收敛
   **分别**检查；阈值与同名同定义的测量一一对应，不混用 L1/L2。
4. **NaN / Inf / 空算例 / 必需字段缺失 / 已请求执行却无测量 一律不能 pass**（判为 fail）。
5. `enabled=false` 与 `--only` 的关系写清楚：被排除的组标 `not_run`，
   并在 `missing_required_groups` 中列出。

**状态语义**（`acceptance_check.json` 的 `status_semantics` 字段同步记录）：

| 情况 | 总状态 | 返回码 |
| --- | --- | --- |
| 默认完整运行，全部必需项实际测量并通过 | `pass` | 0 |
| 用户主动选择子集；已执行项通过 | `partial`（列出未执行项） | 0 |
| 任一已执行项超过阈值 | `fail` | 非零 |
| 已请求必需项但缺测、非有限值或计算异常 | `fail` | 非零 |
| 路径/配置非法 | 明确错误 | 非零 |

### S4.3 负面测试（`tests/test_negative_paths.py`，7 项全部通过）

| # | 注入方式 | 预期 | 实测 |
| --- | --- | --- | --- |
| 1 | 临时配置把离散恒等阈值设为 **0** | B 组 fail、非零 | `discrete_identity_rel_L2=fail`，退出码 1 |
| 2 | 内存注入第 2 个 Airy 算例 0.5/0.7 | 后两项 fail、非零 | fail，最差算例 `D0.2mm_f25mm`，退出码 1 |
| 3 | 内存注入论文尺度第 3 个算例 0.5/0.8 | 第三项 fail、非零 | fail，最差算例 `D1mm_f50mm`，退出码 1 |
| 4 | 内存注入 `parseval_worst_rel_err` = NaN / Inf / 删字段 | 三种都 fail、非零 | 三种都 fail，退出码 1 |
| 5 | 让 D 组抛 `RuntimeError` | fail、非零、保留堆栈 | fail，`run.log` 含 Traceback，退出码 1 |
| 6 | 通过的 `--only direct` | `partial`、退出码 0、B 组不是 not_run | partial，B 组 pass，`not_run=[airy,energy,sampling]` |
| 7 | 默认完整运行 | `pass`、退出码 0 | pass，退出码 0，无 not_run |

每条记录（命令、预期退出码、实际退出码、逐项状态、日志路径）保存在
`results/stage01/_negative_tests/<case>.json`，汇总在 `summary.json`。
负面测试**不修改生产源码**：第 1 项只在临时目录写一份测试专用配置（不提交为默认配置），
其余用进程内注入。负面测试统一加 `--light`（只缩小 FFT 规模，不改任何阈值与物理定义）。

---

## S6 [P2] 配置开关未接入运行逻辑

**怎么修**：先解析路径与配置、算出**最终**运行设置，再选择后端并导入 pyplot。

```text
effective_show = runtime.show_plots 或 CLI --show-plots
effective_save = runtime.save_results 且未使用 CLI --no-save / --log-to-console-only
```

- README 与 `--help` 都写清这条规则；不再有“改配置可弹窗却只实现了 CLI”的矛盾。
- `show_plots=true` 时尝试交互后端；确无 GUI 后端则记录回退与“未弹窗”。
- `save_results=false` 与 `--no-save` 都不创建任何**项目**结果文件、图或日志；
  验证范围明确写为项目输出目录（第三方库的用户级缓存不算项目输出）。
- **路径解析规则改为按项目位置**（`__file__`）：`--config`、`--run-dir` 的相对路径、
  默认配置文件、默认输出根目录（`<项目>/results`）全部按项目解析，
  因此 PyCharm 点击运行与从任意 cwd 用绝对路径运行结果一致；需要按 cwd 解析时给绝对路径。
  README 明确写出这个区别，并从 `D:\PyCharmProjects\Jeon2019` 实测。

**验证**（`docs/verify_config_switches.py`，mock 后端选择，避免阻塞在图窗上）：

| 检查 | 结果 |
| --- | --- |
| 默认配置：不弹窗、落盘，后端未被要求 show | OK |
| 配置 `show_plots=true` 真的生效（后端收到 `show=True`） | OK |
| 配置 `save_results=false` 真的不落盘（项目输出零新增） | OK |
| CLI `--show-plots` 覆盖配置 `false` | OK |
| CLI `--no-save` 覆盖配置 `true` 且无项目输出 | OK |

报告：`results/stage01/_config_switch_check/config_switch_check.json`。

---

## 10. 最终验收清单对照

| 项目 | 要求 | 本轮实测 |
| --- | --- | --- |
| 默认 FFT 与独立完整位移核 | 复场相对 L2 ≤1e−6；含粗/细、非对称、矩形、奇数输入 | 2.52e−13（B 组最差）；测试覆盖非方形/奇数/不同间距 |
| reduced/full 恢复关系 | 补回一次输出二次相位，复场相对 L2 ≤1e−10 | 5.12e−14（反例 1.4052 必须失败） |
| 两种频域全局相位开关 | 与明确相位因子一致，≤1e−10 | Fresnel 0.0；角谱 1.57e−16 |
| 原生 FFT 全输出 Parseval | 相对功率误差 ≤1e−10 | 2.15e−14 |
| 连续高斯解析复场 | 保持原有通过 | 1.39e−14（正入射） |
| 非零倾斜高斯解析参考 | 包络按 zθ 平移，≤1e−6 | 1.19e−14（θ=1 mrad） |
| 三个 Airy 算例 | 全部实际参与验收；暗环 ≤2%，截线 L1 ≤1% | 暗环最差 2.088e−03；截线最差 3.914e−04 |
| 同口径 Airy 包围能量 | P_disk/P_in，绝对差 ≤0.005；网格稳定 | 最差 3.85e−04；M 4096→8192 改善 |
| 输入积分收敛 | 固定器件/窗口，≥三级；最后两级原始强度 L1 ≤2% | 2.606e−03 |
| 输出采样/填充收敛 | 实际两个 M，公共物理网格、直接核参考；最细级 ≤2% | 2.240e−03（两个 M 之间 6.767e−03） |
| 采样诊断 | x/y 与输入/输出间距各自正确，提示与保证区分 | 4 项新测试 |
| 验收负面测试 | 失败、缺测、非有限值被发现；已执行失败返回非零 | 7 项全部通过 |
| 配置与入口 | show/save 配置生效，项目外 cwd 运行成功 | 5 项接线检查通过；项目外 cwd 实测 |
| 文件保护 | 不删除历史内容；新目录；无保存模式无项目输出 | 全部历史运行保留；`--no-save` 项目输出零变化 |
| 中文记录 | 实际命令、数值、公式来源、假设与未验证项完整 | 本文件 + `stage01_revision_report.md` |

---

## 11. 未验证项（如实列出）

1. **PyCharm 点击运行与真实图窗弹出仍未人工验证**。本环境无法点击 PyCharm、无法显示
   窗口；配置开关本身已用 mock 后端选择在自动化测试里验证（
   `docs/verify_config_switches.py`），命令行入口已实际运行（含从项目外 cwd 启动）。
2. **绝对包围能量本轮主水平取 R=4·r1，不是审核示例用的 5·r1**。原因是 `R/半宽 ≤ 0.5`
   的窗口判据；任务书允许“较小且完全包含的 R”。R=5·r1 需要输出半宽 ≥10·r1，
   对应 FFT 长度会显著更大（成本高），本轮没有执行该等级。
3. **可分离直接求值的远尾部**：在 20·r1 处它对无限平面解析分母仍有约 2% 的标定偏差，
   因此**没有**用它做能量分母；这一限制已写入代码注释与本文件。真值以单次 FFT 的
   严格守恒总功率为准。
4. **倾斜高斯只覆盖近轴线性相位系数 θ**，没有处理 `sinθ`/`tanθ` 的精确几何差异
   （θ 为 mrad 量级，差异是二阶小量）。
5. 仍未实现 DOE 高度设计（式(7)–(12)）、三翼结构、多波长编码 PSF、图 3 对照、
   成像（式(13)–(15)）与重建。理想相位元件无吸收与界面反射，真实 DOE 效率未建模。
6. 未确定熔融石英色散系数、16 级量化规则、Canon 响应曲线、图 3 绝对视场；
   无官方代码可对照，未复现论文所用 LightPipes 的具体设置。
