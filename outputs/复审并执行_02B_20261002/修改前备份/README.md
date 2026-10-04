# Jeon2019 光学复现 —— 阶段 01 与阶段 02A (含 02A-R 修正)

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
