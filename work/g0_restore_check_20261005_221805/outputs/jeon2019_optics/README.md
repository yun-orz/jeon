# Jeon2019 光学复现 —— 阶段 01 与阶段 02A (含 02A-R 修正)

> **当前状态索引（2026-10-05，G0）**：本项目已推进到02F-3.2；下面按日期保留的“未执行/等待复核”等文字描述当时状态。
> 当前范围为“Jeon光学编码复现＋基础重建验证”，未实现25波段RGB前向模型、原论文HQS网络或双孔径。
> 统一入口索引、已知限制、冻结证据与G0–G6路线见 [G0冻结说明](../../baseline/g0_20261005/README.md)。
> 原光学/重建源码和配置保持不变；Git标签不包含被忽略的NPZ数组，迁移需保留冻结清单中登记的本地来源。

> **2026-10-01 阶段 02A-R 修正完成版。**
> 已针对原审核意见完成全部 R1–R7 修正：
> 1. **消除自动删除与报告覆盖 (R1)**：测试彻底移除了 `unlink`，每次为非法测试用例分配独立唯一的持久化证据目录；移除了主入口覆盖 `docs/stage02a_report.md` 的代码，每次仅向对应唯一的 `results/stage02a/run_<时间戳>/` 输出报告。
> 2. **真实失败路径与验收完整性 (R2)**：增加了全零场（`peak(I)=0`、`eta<=0`）、NaN 场、shape 错误、指纹篡改及波长列表重复/为空的严格防御；增加了失败路径 mock 测试。
> 3. **配置参数全量校验 (R3)**：严格检查参数类型与有限正数，排斥 float 截断整数、NaN 距离；增加实际网格与声明半宽的一致性校验；限制材料模型为 `fused_silica_malitson1965`、传播方法为 `separable_kernel`；`include_global_phase` 配置真实生效。
> 4. **显示/保存开关与后端生命周期 (R4)**：在导入 `pyplot` 之前完成命令行与配置解析，优先尝试交互 GUI 后端，无可用 GUI 时安全回退至 `Agg` 并如实记录“未弹窗”；保存关闭模式（`--no-save`）下严禁写入磁盘文件。
> 5. **状态、检查点与异常处理 (R5)**：运行目录创建后立即写入 `in_progress` 初始状态，每步骤完成后逐步记录 checkpoint；子任务模式（`--only height`）在自检中将未执行项标为 `not_run`，总体状态标为 `partial`，严禁虚报整阶段 pass；异常时写入 `status=failed` 及完整堆栈。
> 6. **真实报告与文档数值修正 (R6)**：修正了 `stage02_equations_and_parameters.md` 中的理论折射率数值（420nm: 1.468094, 540nm: 1.460344, 550nm: 1.459911, 660nm: 1.456268）；正文式(11)定位更正为第 5 页；澄清三波长 $R_{50}$ 包围能量半径存在差异，未证明物理尺寸恒定。
> 7. **测试集扩充与全绿 (R7)**：测试套件包含 66 项测试（原 50 项传播基线测试 + 阶段 02A 新增 16 项物理/入口/失败路径测试），全部通过。
> **本批仅完成阶段 02A-R 修正，停机等待原审核者复核。02B/02C 及后续工作未执行。**

> **2026-10-01 阶段 02A 初版记录（已被 02A-R 修正取代）。**

> **2026-10-01 接手完成版（阶段 01）。** DSH中止后已补完并实际验证阶段01：50项测试、5项配置检查和默认完整运行通过。两次CPU完整运行实测约16秒、85秒。最新结果与物理解释见 [阶段01接手完成记录](docs/stage01_completed.md)。历史报告继续保留，但当前能量评价以该记录为准；本阶段尚未实现DOE或图3。

默认能量评价为 `method="direct_window"`：用完整位移核在覆盖圆盘的有限窗口求值，以输入功率作分母，按20/40样点每第一暗环半径检查收敛。`energy_evaluation.fft_size` 和 `energy_levels[].fft_size` 仅在显式选择 `method="fft"` 时生效；普通运行不会分配8192×8192能量数组。原生FFT全输出Parseval仍独立运行。

测试与配置核验均保留证据，不自动删除目录。`python -B -m unittest discover -s tests -v` 运行全部测试；`show_plots=true` 的GUI显示与PyCharm人工点击仍需本机手动核验。

> **2026-10-01 修正版。** 初审版有三处物理/实现错误（直接求值重复相位、
> 单次 FFT 漏面积权重与频率未中心化、Airy 适用条件判断错误），已全部修正并
> 重新实际运行。旧结论的撤回清单与逐项修法见
> [`docs/revision_notes.md`](docs/revision_notes.md)；
> 初审版的运行目录与报告作为历史记录**原样保留，未删除、未覆盖**。

本目录是一个**普通本地 Python 项目**：用 PyCharm 打开后点击 `main.py` 的运行按钮即可执行。
项目不依赖 Docker、Notebook、GPU、Agent 状态或任何在线服务；所有路径都按
`Path(__file__).resolve()` 解析，**默认入口与启动时的工作目录无关**。

阶段 01 的范围：**只建立并验证传播基线**。
不实现 DOE 高度设计（论文式(7)–(12)）、三翼结构、多波长编码 PSF、图 3 对照、
成像或重建。

---

## 1. 解释器与依赖

本项目在下列环境中实际运行验证（不是"候选版本"）：

| 项 | 值 |
| --- | --- |
| Python | 3.10.4（64 位，`D:/dev/python/python3.10.4/python.exe`） |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.9 |
| 操作系统 | Windows 11（13th Gen Intel Core i7-13650HX，20 逻辑核，约 15.7 GB 内存） |

`requirements.txt` 固定的就是这些版本。若你的环境不同：

```powershell
# 建议使用项目虚拟环境（可选，不会改动全局 Python）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 本阶段**不需要** pytest：测试用标准库 `unittest` 编写。

---

## 2. 在 PyCharm 里怎么跑

1. `File → Open` 选择本目录（`outputs/jeon2019_optics`）。
2. 解释器选择 `D:/dev/python/python3.10.4/python.exe`（或你自己的 3.10+ 环境）。
3. 打开 `main.py`，点击绿色三角运行。**无需任何参数**：
   - 默认 `stage = stage01`、CPU、保存结果、`show_plots = false`；
   - 结果写入 `results/stage01/run_<时间戳>/`，每次运行新建编号，**不覆盖旧结果**。
4. 想看窗口：把 `config.json` 里的 `runtime.show_plots` 改成 `true`，**或**在
   `Run → Edit Configurations` 的 `Parameters` 里填 `--show-plots`。两者**都真实生效**
   （S6 修正：配置与命令行在**选择后端之前**就合并成最终设置）。

### 最终运行设置的合成规则（S6）

```text
effective_show = runtime.show_plots 或 CLI --show-plots
effective_save = runtime.save_results 且未使用 CLI --no-save / --log-to-console-only
```

* `show_plots=true`：在**导入 pyplot 之前**依次尝试 `QtAgg/TkAgg/GTK3Agg/WXAgg/MacOSX`；
  确无可用 GUI 后端时回退 `Agg`，并在日志与报告里明确写“未弹窗”。
* `save_results=false` 或 `--no-save`：**不创建任何项目结果文件、图或日志**，
  只在标准输出打印。验证范围是**项目输出目录**；第三方库（如 matplotlib）
  可能自行维护用户级字体缓存，那不属于项目输出。

> **未验证项**：本项目的开发与验证都在命令行完成。当前执行环境无法点击 PyCharm、
> 也无法显示窗口，因此“PyCharm 点击运行”与真实的**图窗弹出**路径
> **尚未人工验证**；配置开关本身已用 mock 后端选择在自动化测试中验证。
> 命令行入口已实际运行（包括从其他工作目录用绝对路径启动）。

### 命令行等价写法

```powershell
cd outputs/jeon2019_optics
python main.py                        # 默认配置（四组全部执行）
python main.py --show-plots           # 选择交互后端并弹出图窗（需有可用窗口系统）
python main.py --no-save              # **真正不写任何文件**，只打印（调试）
python main.py --only airy            # 只跑某一组（airy/direct/energy/sampling）；
                                      # 未执行的组在总状态里标为 not_run
python main.py --config other.json    # 指定配置
python main.py --run-dir results/stage01/run_x   # 目标必须**不存在或为空**
python main.py --light                # 只缩小 A/C/D 的 FFT 规模加快回归；
                                      # 不改任何物理定义与阈值，正式运行不使用
python -B -m unittest discover -s tests -v            # 全部单元测试
python -B tests/test_negative_paths.py                # 负面测试（7 项，含退出码与状态）
python -B docs/verify_config_switches.py              # 配置 show/save 接线的 mock 验证
```

**路径与退出码约定**：`--config` 与 `--run-dir` 的**相对路径按项目位置**（`__file__`）
解析，因此 PyCharm 点击运行与从任意工作目录用绝对路径运行的结果一致；
需要按当前工作目录解析时请直接给绝对路径。默认配置文件与默认输出根目录
（`<项目>/results`）同样按 `__file__` 解析，与启动目录无关。

**状态语义**：`pass`（默认完整运行、全部必需项实测通过，退出码 0）、
`partial`（用户主动选择子集且已执行项全部通过，退出码 0，并列出未执行项）、
`fail`（任一已执行项超阈值 / 缺测 / NaN·Inf / 计算异常，退出码非零）。
逐项状态写入 `acceptance_check.json`（`pass` / `fail` / `not_run`）。

---

## 3. 目录与文件职责

```text
jeon2019_optics/
├── main.py                     阶段 01 的入口：四组验证 + 自动生成中文报告
├── config.json                 全部参数（阶段、波长、采样、阈值、输出开关）
├── requirements.txt            实际运行验证过的依赖版本
├── README.md                   本文件
├── optics/
│   ├── __init__.py
│   ├── units.py                nm/μm/mm ↔ m 的集中换算
│   ├── coordinates.py          坐标与采样约定、网格、圆孔、理想透镜
│   ├── propagation.py          式(4) 的四条路径 + 采样/近轴判据
│   ├── metrics.py              误差指标、暗环、包围能量、解析参考
│   └── runutil.py              运行目录/日志/环境记录（不含光学计算）
├── tests/
│   └── test_propagation.py     25 项 unittest（含离散恒等、解析解、D=1 mm Airy）
├── docs/
│   ├── revision_notes.md            2026-10-01 修正说明：R1–R7 逐项怎么修
│   ├── equations_and_parameters.md  公式/单位/页码/代码对应 + 参数对照 + 来源清单
│   ├── assumptions.md               逐条假设：出处、采用值、影响、替代值
│   ├── reproduction_log.md          实际命令、失败与修正过程（含旧结论失效横幅）
│   └── source_manifest.json         PDF 的 SHA256/页数/核实内容（机器可读）
└── results/
    └── stage01/
        └── run_<时间戳>/
            ├── config_used.json          本次实际生效配置
            ├── environment.json          解释器/依赖/平台/工作目录
            ├── run.log                   逐行日志（含所有判据与偏差数值）
            ├── results_summary_auto.json 机器可读汇总
            ├── acceptance_check.json     逐项验收判定（pass/fail/not_run + 总状态）
            ├── stage01_revision_report.md 中文阶段报告（自动生成）
            ├── arrays/*.npz              复场、坐标、强度、剖面（坐标单位 m）
            ├── metrics/*.csv             报告表格的原始数值
            └── figures/*.png             A/B/C/D 各图
```

> `results/stage01/run_20260930_*` 是**初审版**的历史运行，只读保留。
> 其 `reproduction.json` 中的 `6.25e8` 与"N_F=2.3/4.5"等数值**均已作废**。

---

## 4. 单位与坐标约定（重要）

- **内部长度单位一律为米（m）**；nm/μm/mm 只出现在 `config.json`、
  `optics/units.py` 与报告文字里。
- 数学坐标：**x 向右、y 向上、z 沿传播方向**。
- 数组 `a` 形状 `(ny, nx)`，`a[j, i]` ↔ `(x = x_axis[i], y = y_axis[j])`；
  **索引增加时 x、y 都增加**。因此显示用 `origin="lower"`。
  （初审版此处误写为"行指标增加时 y 减小"，与实现相反，2026-10-01 更正。）
- 图像 extent 必须区分两种约定：`extent_center_um` 是**样点中心**范围，
  `extent_pixel_um` 是**像素边界**范围（中心坐标外扩半像元）；画图与像元积分
  都用后者。旧名 `extent_um` 指向像素边界。
- 一维轴以 0 为中心：`coords = (arange(n) − n//2) × d`；半宽 = `(n//2)·d`
  （**不**额外乘 0.5）。
- `x` 与 `y` 允许不同的样点数与间距（非方形网格）。
- 复振幅 `u = A e^{iφ}`，强度 `I = |u|²`；Fourier 变换取**负指数**约定。

---

## 5. 本阶段用什么算法，为什么

代码里保留四条路径，各自附适用条件。**它们互为交叉核验，没有"权威方法"之分**
（初审版称直接求值为"权威方法"的结论已撤回）：

| 路径 | 函数 | 状态 |
| --- | --- | --- |
| 完整位移核求值（式(4) 的离散形式） | `fresnel_kernel_matrix` / `fresnel_kernel_eval` | 与单次 FFT 相差 ≤2.6e−13；评价点可任意选 |
| 同一式的可分离写法 | `fresnel_kernel_separable` | 与上者数学等价（2.1e−16），代价 O(N_out·(Nx′+Ny′))；D=1 mm 焦平面用它求值 |
| 单次 FFT（论文式(6) 的 FFT 写法） | `fresnel_fft` | 补面积权重 + 频率中心化后与完整核**离散恒等**（≤2.6e−13）；原生 Parseval ≤2.2e−14 |
| 频域 Fresnel 近似 / 精确角谱 | `fresnel_transfer_fresnel` / `fresnel_angular_spectrum` | 两个**不同**的传播子，输出与输入同网格，受回绕影响 |

理想透镜焦平面的 Airy 校验依据：**`z=f` 时透镜相位与传播展开式的输入二次相位相消，
焦平面就是孔径的 Fourier 强度**，因此**不要求** `D²/(4λf)≪1`。
论文尺度 `D=1 mm、f=50 mm` 是正式算例（第一暗环 33.541 μm，实测误差 2.1e−3）。

原因与完整诊断过程见 `docs/reproduction_log.md`、
`docs/equations_and_parameters.md` 第 3 节与 [`docs/revision_notes.md`](docs/revision_notes.md)。

---

## 6. 结果怎么看

打开 `results/stage01/<最新 run>/stage01_revision_report.md`，它包含：运行环境与实际命令、
四组验证的完整数值表、逐项验收结论、未验证项清单、文件对应关系。

`figures/` 里每张图的标题都写明参数与网格；物理坐标轴单位为 μm。
`metrics/*.csv` 用 UTF-8-SIG 编码（Excel 可直接打开中文表头）。
`arrays/*.npz` 中的坐标数组单位为米，`index [j, i] = (x_i, y_j)`。

### 验收阈值（`config.json → acceptance`）

| 项 | 阈值 | 说明 |
| --- | --- | --- |
| Airy 第一暗环半径误差 | ≤ 2% | 本项目建议值，**非论文数值** |
| Airy 中心截线相对解析 L1 | ≤ 1e-2 | 同上 |
| 离散恒等：FFT vs 完整位移核 rel L2 | ≤ 1e-6 | 同一离散输入、同一原生输出网格；实测 2.6e−13 |
| 原生 FFT 全输出 Parseval 相对误差 | ≤ 1e-10 | **主能量判据**；实测 2.2e−14 |
| 有限窗口对解析包围能量（绝对差） | ≤ 0.05 | **补充**检查，不替代 Parseval |
| 采样/填充收敛 | ≤ 2e-2 | 各等级与填充倍率的离散恒等 |

阈值在运行前写入配置，未在看结果后放宽；`acceptance_check.json` 记录
`pass`/`fail`/`not_run` 与总体状态，失败或缺测返回非零退出码。

---

## 7. 无窗口批处理 vs 手动查看

- **无窗口批处理（默认）**：在导入 pyplot **之前**选择 `Agg` + `show_plots=false`，
  只写 PNG，不弹窗；适合远程/服务器。
- **手动查看**：`show_plots=true`（或 `--show-plots`）会依次尝试
  `QtAgg / TkAgg / GTK3Agg / WXAgg / MacOSX`，成功即用并在报告里记录实际后端；
  全部失败则回退 `Agg` 并明确标注"不会弹窗"。
  **该弹窗路径尚未人工验证**（见第 2 节说明）。

---

## 8. 下一步（不在本阶段）

阶段 01 完成后停止，提交审核。只有收到审核通过后的阶段 02 任务，
才继续实现 DOE 高度设计（式(7)–(12)）、材料色散与量化分支。
本阶段**没有**生成任何三翼相位图、多波长 PSF 或重建结果。


## 2026-10-01：阶段02A-R2接手完成

用户授权后已直接完成C1–C7修正。冻结版本87项测试及默认完整运行通过；最新有效记录见 [stage02a_r2_completed.md](docs/stage02a_r2_completed.md)。

本阶段运行main_stage02.py，配置config_stage02a.json；正式结果为results/stage02a/run_20261001_215848_02。相对配置和输出路径按项目根解析，非空run目录拒绝覆盖。预览波长目前仅接受整数nm，不支持540.05nm等非整数值。显示配置与CLI在导入pyplot前合成，采用TkAgg或Agg回退；真实GUI及PyCharm点击尚未人工验证。

历史报告保留；图3九波长、采样收敛、加工量化和重建尚未执行。本记录更新旧章节的阶段02A状态，未修改阶段01物理模型。


## 2026-10-01：阶段02B-1 九波长对照

本批（**Jeon光学编码复现：02B-1九波长对照**）在**同一个探测器物理网格**上，
为**两张固定 DOE 高度**各算九个入射波长，共 **18 组**真实衍射复场与原始强度：

1. 固定 550 nm 设计的**传统 Fresnel DOE** —— 色散对照；
2. **Jeon 连续高度 N=3 DOE** —— 正文图 3 的三翼旋转编码对象。

完整记录见 [stage02b1_completed.md](docs/stage02b1_completed.md)，
自动生成的中文报告见本次 run 目录下的 `report_stage02b1.md`。

### 在 PyCharm 里怎么跑

1. 打开项目根 `outputs/jeon2019_optics`，选择解释器 `D:\dev\python\python3.10.4\python.exe`。
2. 打开 `main_stage02b.py`，点击运行。**无需任何参数**：默认执行全部 02B-1、
   保存 PNG 与关键数值、不弹窗；结果写入 `results/stage02b1/run_<唯一编号>/`，
   每次新建编号，**不覆盖也不清理旧 run**。
3. 想看窗口：把 `config_stage02b.json` 的 `runtime.show_plots` 改成 `true`，
   或传 `--show-plots`。程序会在**导入 pyplot 之前**探测 Tk 依赖与窗口系统，
   成功用 `TkAgg`，失败回退 `Agg` 并在日志与报告里写明“未弹窗”。

### 命令行

```powershell
# 在项目目录内
python -B main_stage02b.py
python -B main_stage02b.py --no-save            # 不写任何项目文件
python -B main_stage02b.py --only height        # 只做高度设计（partial）
python -B main_stage02b.py --only control       # 单器件单波长试算（partial）
python -B main_stage02b.py --only preview       # Jeon 九波长（partial）
python -B main_stage02b.py --only analyze --from-run <run_dir>
python -B main_stage02b.py --run-dir results/stage02b1/my_run   # 目标必须为空

# 从项目之外的任意目录（路径按 __file__ 解析，与 cwd 无关）
& 'D:\dev\python\python3.10.4\python.exe' -B `
  'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\main_stage02b.py'

# 专项测试与全量测试
python -B -m unittest discover -s tests -p "test_stage02b1.py" -v
python -B -m unittest discover -s tests
```

### 参数与输出位置

参数集中在 `config_stage02b.json`（内部长度一律 m）：D=1 mm、f=z=50 mm、N=3、
设计范围 420–660 nm、传统设计 λ0=550 nm、入射 9 个波长
（420/450/480/510/540/570/600/630/660 nm）、输入 1101×1101 @1 μm、
输出 301×301 @1 μm。输出目录结构：

```text
results/stage02b1/run_<唯一编号>/
  config_effective.json / environment.json / source_manifest.json
  run.log / checkpoint.json / completion.json / report_stage02b1.md
  arrays/height_fresnel.npz, arrays/height_jeon.npz
  arrays/psf_fresnel_<nm>nm.npz ×9, arrays/psf_jeon_<nm>nm.npz ×9
  figures/figure3_comparison_peak_normalized.png
          figure3_comparison_raw_shared_scale.png
          figure3_comparison_raw_per_device_scale.png
          rotation_angles.png / rotation_reliability.png
          size_metrics.png / energy_metrics.png
  metrics/psf_metrics.csv, metrics/psf_metrics.json
```

三张图3对照图的色标含义**不同，不可混用**：峰值归一化图只观察形状；
`raw_shared_scale` 用 18 幅共同的绝对色标（比较绝对亮度，但两种器件峰值量级差约
7 倍，Jeon 行会偏暗）；`raw_per_device_scale` 是 raw 强度、同一行共用色标
（观察色散趋势与三翼形状）。raw 强度与 `Pin` 均**未**被归一化修改。

### 状态语义与保护

- 默认完整运行全部必需项通过 → `completed`，退出码 0；子模式成功 → `partial`；
  传播/保存完整性异常 → `failed` 且非零退出。
- 角度低可靠与 R80 未达到是 **limitations**，如实记录，不算计算失败。
- 非空 `--run-dir` 在写日志之前拒绝；相对路径按项目根解析。
- `--no-save` 不创建 run、日志或报告；此时只有“落盘才能验证”的检查标为
  `not_applicable`，物理与设计检查照旧必须通过。
- 完成后重读全部 18 组数组，核对身份集合严格等于 {两器件}×{九波长}、
  坐标、精确 λ、高度指纹、`Iraw == |u2|^2` 与功率范围。

### 未人工验证项

**PyCharm 人工点击与真实图窗弹出未人工确认**（本执行环境无法点击、无法显示窗口）。
配置 `show_plots` / `save_results` 与 CLI 覆盖由自动化测试用 mock 后端验证，
不等同于人工验证。本批**没有**执行采样加密（02B-2）、加工量化、相机像元积分、
场景成像、重建或双孔径。


## 2026-10-02：阶段02B-1 修正批（R1–R8）

审核结论为“暂不通过最终审核”，要求逐项关闭 R1–R8。修正记录见
[stage02b1_revision_report.md](docs/stage02b1_revision_report.md)；
正式 run 为 `results/stage02b1/run_20261002_100625`（状态 `completed`，退出码 0）。
历史 run 与历史报告全部保留，未删除、未覆写。

### 本批修正要点

| 项 | 修正 |
| --- | --- |
| R1 | 圆盘累计先用 `Rlimit=min(xmax,−xmin,ymax,−ymin)` 截断，`Eabs_at_Rlimit` 对应真实圆盘；超过 Rlimit 的评价半径直接拒绝 |
| R2 | 保存后重读会**独立重算** `Pin`（用配置+固定高度+λ 重建 u1）与 `Pwindow`（重积分保存强度），并核对 η/峰值/振幅/n(λ)/z/f/dtype/坐标/高度内容指纹 |
| R3 | 传播**显式使用配置 z**；PSF 同时记录 `propagation_distance_m` 与 `design_focal_length_m` |
| R4 | 报告在状态确定后按**实际模式**生成；**只有完整 all** 才是 `completed`，子模式为 `partial`；报告/图缺失即失败且非零退出 |
| R5 | `--only analyze --from-run <run_dir>` 已真正实现：只读源 run、兼容性检查、输出到新的唯一 run，并核对源 run 文件 SHA256 不变 |
| R6 | 显示与保存解耦：`--no-save --show-plots` 也会创建图（只在内存，不落盘） |
| R7 | `A3` 未达阈值时 `alpha` 为 `null`；原始相位另名 `alpha_raw_*` 并标注**不是**有效方向 |
| R8 | 测试**不再删除**任何目录（证据永久保留在 `results/_test_evidence/`）；子进程编码双向显式设为 UTF-8，并移除 `errors='replace'` |

### 新增运行方式

```powershell
# 用已有 run 重新分析（源 run 只读，输出到新的唯一 run）
python -B main_stage02b.py --only analyze --from-run results/stage02b1/run_20261002_100625

# 全量回归（134 项）
python -B -m unittest discover -s tests
```

测试证据目录：`results/_test_evidence/stage02b1/<用例_时间戳>/`，内含
`EVIDENCE_RETAINED.txt` 说明来源；**如需清理请由用户明确指示后再删除**。

### 仍未解决

- **旋转方向与正文图 3 的对应**：本项目坐标（x 右 / y 上、逆时针为正、未做镜像）
  实测为**逆时针**；正文图 3 文字记为顺时针，**符号相反且未解决**。
  统一角零点偏置只能平移角度、无法解释反号；需由原文观察面/角向符号约定确认，
  本批不擅自改高度、相位或镜像数据。
- 未证明采样收敛（属 02B-2）；PyCharm 人工点击与真实弹窗未人工验证。
# 2026-10-02：审核者接手与02B-2采样/视场验证

保留原入口。新增`main_stage02b2.py`，在PyCharm中打开本项目并直接点击运行，无参数即在CPU完成两种DOE、三个波长及六组采样/视场实验；默认保存、不弹窗。参数集中在`config_stage02b2.json`，内部长度仍为m，路径按入口文件位置解析。

解释器与依赖实际验证：Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9（Windows/CPU）。依赖没有新增。可在终端使用：

```powershell
& 'D:\dev\python\python3.10.4\python.exe' -B 'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\main_stage02b2.py'
& 'D:\dev\python\python3.10.4\python.exe' -B 'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\main_stage02b2.py' --no-save
```

结果在`results/stage02b2/run_...`，包含原始场/强度/物理坐标、采样高度、指标CSV/JSON、两张结果图和`report_stage02b2.md`。可用Windows图片查看器手动打开PNG。设置runtime.show_plots=true或传--show-plots会尝试显示；真实PyCharm点击和弹窗仍需人工确认，自动测试不代替人工确认。

新配置中cases为六组：input_coarse/baseline/input_fine比较2/1/0.5μm输入；output_coarse/input_fine/output_fine比较2/1/0.5μm输出；output_fine/fov_large比较±150/±300μm。thresholds为本项目筛查容差，不是论文标准。状态completed只表示本批计算/交付完成；scientific_status单独表示是否达到已测试容差。

同一输入组先直接传播到最大最密输出，再抽取严格重合的原生坐标点；抽取已与独立小网格传播核对。没有插值或旋转生成PSF。比较输出采样时的双线性强度插值仅用于误差诊断。

02B-1的from-run入口现在执行完整高度/功率/材料重验及传播源码兼容性检查；缺新schema必要字段的旧run可被拒绝，不伪造缺失元数据来放行。分析输出仍为新唯一目录，旧run保留；失败写failed及checkpoint。all最终必需项包含分析、报告和图表，子模式标partial。

已测试的1μm默认输入/输出相对0.5μm参考稳定；最细网格不是解析真值，未做0.25μm输入加密。±150μm截断部分能量，能量/尺寸评价建议扩大到±300μm；扩大后仍不能把捕获量当制造效率。当前为连续高度单完整孔径N=3光学验证，未做像元积分、加工量化、场景或重建。

原文PDF第5页写顺时针，而当前数学x右/y上结果逆时针。正文/补充图3已经查看，但观察面和角坐标对应尚未确定；保持该限制，不镜像数据迎合论文。详细审核与执行记录见工作区`outputs/复审并执行_02B_20261002`及本项目后续完成记录。

## 2026-10-02：阶段02C制造高度与像元积分

本节为最新阶段；上文各日期段保留当时的执行状态。02C已实际计算固定连续高度、最近深度级量化、向下取深度级量化的420/540/660 nm衍射，并完成6.22 μm理想像元积分。本阶段仍是单完整N=3孔径，没有场景、重建或双孔径。

在PyCharm打开`outputs/jeon2019_optics`作为普通Python项目。选择本地Python解释器；本机实际验证的是Python 3.10.4、NumPy 2.2.6、SciPy 1.15.3、Matplotlib 3.10.9。新建环境时在项目终端运行`python -m pip install -r requirements.txt`，安装操作本批没有重复执行。右键`main_stage02c.py`，选择运行；无需参数、Notebook、GPU或Harness。默认CPU计算并保存，路径由入口所在目录解析，不依赖PyCharm的工作目录。

```powershell
python -B main_stage02c.py
python -B main_stage02c.py --no-save
python -B main_stage02c.py --show-plots
```

常用参数集中在`config_stage02c.json`：

|参数|含义与默认值|
|---|---|
|optical|D=1 mm、f=z=50 mm、N=3；三入射波长420/540/660 nm；长度存储为m|
|input|1101×1101、间距1 μm，完整覆盖圆孔|
|fabrication|16个可用深度级、100 nm间距、0.5 mm绝对基底厚度；两种舍入实施假设|
|detector|97×97像元、6.22 μm间距；局部窗口边界±301.67 μm|
|quadrature_per_axis|[4,8]表示每个像元分别用4×4与8×8强度求积点|
|thresholds|项目数值筛查容差，非论文规定：L1≤0.5%、L2≤1%、η绝对差≤0.001、角差≤0.5°、半径差≤1像元|
|runtime|save_results默认true；show_plots默认false，二者独立|
|plots.dpi|保存图像分辨率；默认150|

求积节点直接由Fresnel复振幅传播得到，不通过旋转图片或缩放图片生成。像元功率是强度面积积分，平均强度=像元功率/像元面积；不能先把复振幅平均再平方。显示图逐幅峰值归一化仅用于形状观察，原始像元功率保留窗口通量。光谱响应采用单位辐射响应、满填充像元，尚无CFA/QE、电子数、反射或吸收模型。

深度d=−Δh，量化到k×100 nm，k∈[0,15]，器件绝对厚度为0.5 mm−d。模型使用相对高度，以省去不影响单器件强度的全局相位；它没有把0.5 mm基底误当成1.5 μm浮雕。正文未公开舍入规则，两种规则均标注为假设。默认结果两种规则实际使用0–14共15个级，不强制占用16个级。

每次运行创建新的`results/stage02c/run_...`：`arrays`保存3张高度、18组真实求积复场及9组中心点近似复场；`metrics/validation.json`保存指标、误差、材料相位误差、环境与限制；`summary.csv`可用Excel查看；`figures`保存两张PNG；`report_stage02c.md`为中文报告。双击PNG可用Windows图片查看器查看；`--show-plots`尝试弹出图窗。真实PyCharm点击与弹窗尚未人工验证。

计算/保存异常非零退出并保留failed.json；数值筛查不通过则报告completed但numerical_convergence_passed=false、退出2，表示本批计算完成但尚需加密，不能据此宣布物理验证通过。`--no-save`只在内存计算和绘图，不创建结果目录或日志。阶段结论和实际验证记录见`docs/stage02c_completed.md`。正文旋转方向与本项目坐标的对应仍未确定，当前结果未镜像。

## 2026-10-02：阶段02D-1单通道非相干编码成像

在PyCharm打开本项目、选择上述本地解释器，右键`main_stage02d1.py`运行。依赖仍是requirements.txt中的NumPy/SciPy/Matplotlib，无新增包或GPU需求。默认读取02C已审核run，计算两种固定DOE、三个单色诊断波长、三种32×32自生成场景。先完成前向成像和伴随验证，本批没有重建或网络。

```powershell
python -B main_stage02d1.py
python -B main_stage02d1.py --no-save
python -B main_stage02d1.py --show-plots
python -B -m unittest discover -s tests
```

参数集中在`config_stage02d1.json`，相对路径基于入口文件所在目录：source_run默认`results/stage02c/run_20261002_172338`；devices默认continuous与nearest_depth，可加入floor_depth；scene_shape默认[32,32]；seed为检查随机种子；response固定单位辐射响应；crop为[y0,y1,x0,x1]，终点不包含。默认crop=[16,112,16,112]，从128×128完整卷积图取96×96测量。更改场景大小时也应检查crop索引与输出坐标。波长、6.22 μm间距和q8受本批物理范围约束，不可直接任意更改。

换电脑时保留默认源run全部文件和对应源码；也可以把完整run复制到其他位置后修改source_run。源run的物理数据、坐标、高度指纹和光学源码会重验；若缺失，程序明确失败并保存失败记录，不能用旧图或缺字段的数据补造结果。源码修改后需要回到02C重新验证并生成相容的新run。

一个物点的像由衍射光斑PSF分散到多个像元，多个非相干物点或波长的功率相加。核K=像元积分功率/孔径输入功率，核的和是有限窗口捕获量，不强制变成1，也不再次乘像元面积。场景X是理想像平面点源的波段积分相对功率；没有物距、放大率、宽波段积分、CFA/QE或真实相机电子数模型。

成像用零延拓full线性卷积，避免FFT循环绕回；裁剪是显式投影。伴随把裁剪残差放回完整网格，再用翻转核作valid卷积，它为后续重建提供梯度，**本身不是重建结果**。偶数场景轴为半像元对称坐标，程序保存全部物理轴，不把某个索引擅自视为光轴。

输出新建`results/stage02d1/run_...`，不会覆盖旧run。arrays保存6份场景/三波段核/各波段编码/完整与裁剪测量/坐标；figures保存连续与量化器件编码图；metrics/validation.json保存通量、直接叠加、伴随和梯度误差；source_evidence.json保存源run文件SHA与物理审核记录；report_stage02d1.md为中文报告。图中颜色均为原始相对功率，未将三波段伪彩显示冒充相机RGB。

validation_passed=true才表示本批离散算子筛查通过；异常非零退出并保留failed.json，筛查未通过退出2。--no-save不写结果；--show-plots尝试显示；两开关独立。实际PyCharm点击与真实弹窗仍未人工验证。完整执行与限制见`docs/stage02d1_completed.md`。后续02D-2再做基础非负正则化重建，按“Jeon光学编码复现＋基础重建验证”标注。

## 2026-10-02：02D-2基础非负正则化重建

在PyCharm右键`main_stage02d2.py`运行，无需参数。仍使用已验证的Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9，无新增依赖、GPU、Notebook或Harness。新环境安装步骤沿用`python -m pip install -r requirements.txt`；本批没有重复安装依赖。路径基于入口文件，源数据默认读取`results/stage02d1/run_20261002_203904`并重验它对应的02C光学源。

```powershell
python -B main_stage02d2.py
python -B main_stage02d2.py --no-save
python -B main_stage02d2.py --show-plots
```

**当前八组500轮默认实验都为iteration_limit，不能当作已收敛的光谱恢复。** 程序及独立小规模参考解验证通过，与合成场景恢复充分收敛是两件事。线条场景的测量残差约3.7%，cube误差却约67.6%；5000轮代表场景诊断仍未收敛，不能仅据此断言编码本身不可恢复。

方法为零初始化非负岭投影梯度：min_{X≥0} 0.5||AX−y||²+α||X||²/2。数据项使恢复能够解释测量；α惩罚较大的功率幅值，正则项是本项目基础先验，不是原论文网络。用伴随计算梯度，投影把更新后的负场景功率置零。测量中的机器舍入微负数不剪裁。真实cube不用于迭代或初始化。

参数集中在`config_stage02d2.json`：source_run为只读成像来源；solver.alpha_relative=1e−4，实际α=相对值×算子谱范数数值估计，实际值保存；max_iterations=500；gradient_tolerance=1e−5、objective_tolerance=1e−7需同时满足才能converged；power_iterations=100、power_tolerance=1e−8、seed=2019控制幂迭代；initial_L_factor=1.05、backtracking_factor=2、max_backtracks=50控制回溯。幂迭代不是严格步长上界，因此每步检查二次上界。evaluation.spectrum_norm_threshold=1e−12控制SAM零光谱判定。full_diagnostic_scene默认lines_and_square。

默认六组crop重建和两组full诊断，CPU本机约30秒；机器负载不同会改变耗时。增大迭代数会延长运行，应先记录状态/目标/投影梯度，不能只观察恢复图。所有场景统一相对α，没有根据真值逐场景挑参数；full/crop及器件的有效α因算子不同而略不同，已记录，不应把比较解释成单一因素实验。

结果在新唯一`results/stage02d2/run_...`：arrays保留8份真值/恢复/测量/拟合/残差/核/物理坐标；metrics保存每轮优化CSV/JSON、评价CSV和validation.json；figures保存16张恢复/拟合图与一张优化曲线；report_stage02d2.md为中文报告。图中三波段真值、恢复、绝对误差共用原始功率色标；SAM未定义光谱单独计数，没有编造0°。未计算PSNR。

`status=completed`只表示计算和交付完成；`validation_passed`为来源/算子和实现检查；`all_reconstructions_converged=false`明确表示恢复尚未充分收敛；各实验有solver_status。默认退出0不表示恢复准确或优化最优。回溯失败/来源缺失/保存失败非零退出并保留failed.json；最终完成标记在全部必需文件保存重读后写入。--no-save不保存；--show-plots尝试显示；真实PyCharm点击和弹窗仍未人工验证。详见`docs/stage02d2_completed.md`。

## 2026-10-02：02D-3相同目标的加速非负重建

在PyCharm右键`main_stage02d3.py`运行，无需参数，仍使用Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9及现有requirements.txt，CPU，无新增依赖。路径基于入口文件目录。

本批逐实验从02D-2源读取实际α，保持相同A/y/非负约束/零初始化和停止门槛，采用回溯及单调重启的加速投影梯度。八组577–769轮全部达到原门槛，正式CPU运行约39.7秒；187项回归通过。线条cube误差仍约37%、共点源约86.5%，优化收敛不能替代恢复准确性。

```powershell
python -B main_stage02d3.py
python -B main_stage02d3.py --no-save
python -B main_stage02d3.py --show-plots
```

参数集中在config_stage02d3.json：source_run指向只读PG源；max_iterations=5000；gradient_tolerance=1e-5与objective_tolerance=1e-7同时满足；backtracking_factor=2、max_backtracks=50；其余检查、显示、保存和dpi集中配置。停止门槛需与PG源相同，实际α不重新估计。迁移电脑须保留三层来源run完整文件与相容源码。

输出唯一results/stage02d3/run_...，含8份NPZ、17张PNG、逐轮JSON/CSV、评价CSV、来源SHA和中文报告。退出0要求optimization_validation_passed=true；未收敛等优化验收未通过退出2；异常留failed.json。实际验证了项目外启动、--no-save文件SHA不变及模拟显示分支，真实PyCharm点击和GUI仍未人工验证。保留旧02D-2未收敛结果作为历史对照。

完整物理解释、算法、八组数值和限制见`docs/stage02d3_completed.md`；本批仍为Jeon光学编码复现＋基础重建验证，不是论文网络或双孔径复现。

## 2026-10-02：02E-1固定模型α敏感性

在PyCharm右键`main_stage02e1.py`无参数运行，CPU，无新增依赖，实测Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；安装沿用requirements.txt。路径基于入口文件，迁移需完整保留02D-3及递归PG/成像/光学来源。

配置集中在config_stage02e1.json：固定α倍数[0.01,0.1,1,10]，continuous/crop三个场景12组主扫描；线条另外两组更严诊断；max_iterations=5000，预算600秒在实验之间检查；证书阈值5%只作标记。真值不参与优化或参数选择。

```powershell
python -B main_stage02e1.py
python -B main_stage02e1.py --no-save
python -B main_stage02e1.py --show-plots
```

12组主扫描均达原门槛；一组更严诊断仍iteration_limit，因此默认退出2、总优化验收false，属于已完成的诊断结果。α×0.01线条cube误差约7.83%，基线约36.96%，但严格诊断和共点源仍有数值不确定性；无噪声结果不能用来指定真实相机最优参数。总功率接近真值不代表光谱/空间分布正确。

正式CPU运行约141.6秒，194项回归及独立审核通过。输出唯一results/stage02e1/run_...，14份原始NPZ/轨迹、4张PNG、评价CSV、来源SHA、逐组progress和中文报告。真实验证项目外启动、缺失源/未收敛/预算/保存故障及--no-save文件SHA不变；mock显示通过，真实PyCharm点击及GUI仍未人工验证。完整推导、数值与限制见docs/stage02e1_completed.md。仍为Jeon光学编码复现＋基础重建验证，非原论文网络或双孔径。

## 2026-10-02：02E-2弱α数值稳定性复核

PyCharm右键main_stage02e2.py无参数运行共点/线条首轮，配置config_stage02e2.json；右键main_stage02e2_extra.py运行一次更严线条诊断，配置config_stage02e2_extra.json。仍是本地CPU、Python3.10.4/NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9，无新增依赖，路径基于入口位置。安装沿用requirements.txt，迁移需完整五层来源run与相容源码。

固定原α×0.01及同一A/y/非负约束/零初始化，最多15000轮，预算300秒在组间检查。首轮门槛1e-7/1e-9，两组收敛但线条距离界1.07%略高于1%，默认退出2；一次追加线条门槛1e-8/1e-10，9792轮通过，退出0。1%阈值保持不变，不将首轮修改为通过。

最终共点cube误差约18.13%、线条约7.93%；相对于固定目标最优解的距离理论上界约0.40%/0.066%，并非真实cube误差界。首轮/追加CPU约64.3/55.9秒；200项回归与独立直接卷积/转置审核通过。

```powershell
python -B main_stage02e2.py
python -B main_stage02e2_extra.py
python -B main_stage02e2.py --no-save
python -B main_stage02e2.py --show-plots
```

新结果唯一results/stage02e2/run_...，含原始NPZ/优化JSON与CSV、对照PNG、评价、progress及中文报告。退出0要求优化与1%界都通过，退出2需查看各标志；报告保存和显示失败留failed.json且不写完成标记。默认无保存及mock显示验证通过，真实PyCharm/GUI未人工验证；追加入口独立显示/无保存命令尚未单独运行。完整记录见docs/stage02e2_completed.md，仍为Jeon光学编码复现＋基础重建验证。

## 2026-10-02：02E-3连续与量化器件弱α对照

在PyCharm右键main_stage02e3.py无参数运行，配置config_stage02e3.json。continuous/nearest_depth各三个场景，使用各自D3实际α×0.01，因此是相同相对正则化尺度对照，实际α略不同，不能全部归因于量化。三单色、单完整孔径、无噪声、单位辐射响应，未扩展双孔径或网络。

参数集中设置devices/scenes、source_run/reference_runs、solver门槛1e-8/1e-10、最多15000轮、1%距离理论界、组间600秒预算及保存/显示/dpi。路径基于入口文件；迁移保留七份源run及相容源码。本地CPU，Python3.10.4/NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9，无新增依赖，安装沿用requirements.txt。

```powershell
python -B main_stage02e3.py
python -B main_stage02e3.py --no-save
python -B main_stage02e3.py --show-plots
```

新结果唯一results/stage02e3/run_...：6份原始NPZ/逐轮JSON和CSV、10张统一功率或停止量PNG、评价CSV、来源SHA、progress和中文报告。完成标记最后写；退出0要求全部优化与1%距离界通过，退出2查各状态，异常留failed.json。真实CLI/无保存SHA/mock显示已验证，真实PyCharm点击/GUI仍未人工验证。

完整数值、206项回归和独立审核、器件差异的优化不确定性解释见docs/stage02e3_completed.md。不要仅按小的点估计差异排序，也不要将最优解距离界当真值误差界。仍为Jeon光学编码复现＋基础重建验证。

## 2026-10-02：02E-4固定共同实际α对照

PyCharm右键main_stage02e4.py无参数运行，集中配置config_stage02e4.json，共同实际α从continuous D3基线×0.01读取，约1.0589016188494555e-6。本批审核复用三组continuous，重新计算三组nearest_depth；α相同，核/测量各自保持物理通光量。配置不相容时明确重算，不伪装成复用。

复用记录execution=reused_verified、new_iterations=0，历史轨迹时间保留；computed才是当前迭代。默认门槛1e-8/1e-10、15000轮、1%距离界及组间300秒预算。三组新增均收敛，六组完整优化/界验收通过；总入口约134.8秒，214项回归和独立复用/卷积/转置/区间审核通过。

```powershell
python -B main_stage02e4.py
python -B main_stage02e4.py --no-save
python -B main_stage02e4.py --show-plots
```

连续/量化cube误差：共点约18.08%/18.78%，分离点约11.08%/11.23%，线条约7.93%/7.99%。线条差异的优化理论区间跨零，不能可靠排序；所有结果仅针对当前无噪声三单色模型。

新结果唯一results/stage02e4/run_...，含6份NPZ/轨迹、10张PNG、评价/执行身份/来源指纹/progress和中文报告。退出0要求全部计划组优化及距离界通过，退出2查各状态；异常留failed.json。实测Python3.10.4/NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9，CPU，无新增依赖；安装沿用requirements.txt，路径基于入口文件，迁移保留8份源run和相容源码。真实CLI/无保存SHA/mock显示已验证，真实PyCharm/GUI仍未人工验证。详细推导、缓存条件和记录见docs/stage02e4_completed.md；仍为Jeon光学编码复现＋基础重建验证。


## 02F-1合成电子数测量（2026-10-04）

在PyCharm直接运行`main_stage02f1.py`；参数在`config_stage02f1.json`，默认CPU、18组、保存结果、不弹窗。噪声前向、单位换算、输出与实际验证说明见[阶段说明](docs/stage02f1_completed.md)。这是合成测量验证，尚无噪声重建。正式结果目录为`results/stage02f1/run_20261004_093359`；失败/预算目录保留，不应自动选择最新目录。


## 02F-2含噪基础重建（2026-10-04）

PyCharm直接运行`main_stage02f2.py`，配置`config_stage02f2.json`，G=10000共12项，CPU约35.5秒。详见[中文阶段说明](docs/stage02f2_completed.md)。电子量纲和优化收敛通过，但恢复误差较大，不宣称优质重建或完整论文复现。正式目录`results/stage02f2/run_20261004_095323`。


## 02F-3.1冻结目标诊断（2026-10-04）

PyCharm运行`main_stage02f3_1.py`，配置`config_stage02f3_1.json`。固定原权重/alpha，12项理想控制；结论与解释见[中文阶段说明](docs/stage02f3_1_completed.md)。理想基线误差明显高于本次噪声扰动，尚未确定各偏差来源，不宣称恢复质量改善。最终正式目录`results/stage02f3_1/run_20261004_101012`。


## 02F-3.2仅观测留出选参（2026-10-04）

PyCharm运行`main_stage02f3_2.py`，参数`config_stage02f3_2.json`。7373训练像元、1843验证像元，四候选＋全数据重拟合CPU约68秒。见[中文阶段说明](docs/stage02f3_2_completed.md)。本场景选中最弱网格因子1e-5，最终相对误差约0.680（旧基线0.944），不是全局最优或普遍改善结论。正式目录`results/stage02f3_2/run_20261004_231220`。
