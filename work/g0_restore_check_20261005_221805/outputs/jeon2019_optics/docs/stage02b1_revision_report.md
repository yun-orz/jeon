# 阶段 02B-1 修正报告（R1–R8 逐项关闭）

日期：2026-10-02。依据：`outputs/审核_阶段02B-1_20261001/审核结论与DSH修正任务.md`。
批次名称：**Jeon光学编码复现：02B-1九波长对照**（修正批）。

> 本文件是**新增**记录，不覆盖历史报告。旧记录
> `docs/stage02b1_completed.md` 与 `results/stage02b1/run_20261001_224235`、
> `run_20261001_225933` 全部**原样保留**；审核证据目录
> `outputs/审核_阶段02B-1_20261001/` 未被读取之外的任何改动。

## 1. 送审的正式结果

- 正式 run（绝对路径）：
  `D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\results\stage02b1\run_20261002_163101`
- 状态：`completion.json` → `status = completed`、`all_checks_pass = true`，退出码 **0**
- 中文报告：该 run 下 `report_stage02b1.md`
- 本次实际命令（从**项目之外**的工作目录启动）：

```powershell
Set-Location D:\PyCharmProjects\Jeon2019
& 'D:\dev\python\python3.10.4\python.exe' -B `
  'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\main_stage02b.py'
```

### 环境（本次实测）

Windows / CPU；Python 3.10.4（`D:\dev\python\python3.10.4\python.exe`）、
NumPy 2.2.6、SciPy 1.15.3、Matplotlib 3.10.9。详见该 run 的 `environment.json`。

### 关键实测数值（取自该 run 的 `completion.json`）

| 项 | 值 |
| --- | --- |
| 状态 / 退出码 | `completed` / **0** |
| 身份集合 | 期望 18，实读 **18**，问题 **0** 条 |
| 独立重算 `Pin` 最差相对差 | **0.0**（容差 1e−12） |
| 独立重算 `Pwindow` 最差相对差 | **0.0**（容差 1e−9） |
| 必需产物交付 | `required_artifacts_delivered = pass`（7 张图 + 2 个指标文件） |
| 报告交付 | `report_delivered = pass` |
| 采样诊断（420 nm） | 输入核步进 **0.209589** rad（阈值 0.50，ok=True） |
| 源码清单 | **18 个文件**（生产 11 + 测试 5 + 说明 2），指纹 待本次 run 的 source_manifest.json 记录…，与磁盘内容**逐一致** |

### 冻结后完整回归（本节数值为实际运行结果）

| 项 | 实测 |
| --- | --- |
| 命令 | `Set-Location <项目根>; python -B -m unittest discover -s tests` |
| 环境预置 | **无**（普通 PowerShell，未设置 `PYTHONIOENCODING`；子进程编码由测试模块自己设定） |
| 实际项数 | **134 项** |
| 结果 / 退出码 | **OK / 0** |
| 耗时 | **150.22 秒**（另有前一次 162.82 秒；随负载变化，不作速度保证） |
| 失败项 | 无。审核记录的 `test_entry_no_save_mode` 失败已随 R8 修复关闭 |

## 2. 实际修改的文件

| 文件 | 改动 |
| --- | --- |
| `optics/psf_analysis.py` | **R1** 圆盘边界重写；**R7** 不可靠方向不发布角度；新增 `height_fingerprint_from_arrays` 独立指纹 |
| `main_stage02b.py` | **R2** 独立重算功率+元数据+高度内容；**R3** 传播显式使用配置 z；**R4** 完成状态与报告/产物绑定；**R5** 实现 `--from-run`；**R6** 显示与保存解耦；报告增补方向核对与指标口径说明 |
| `tests/_stage02b1_support.py`（新增） | **R8** 测试证据保留目录 + 子进程双向 UTF-8 |
| `tests/test_stage02b1.py` | **R8** 去掉自动删除；新增 R1/R2/R3/R7 反例测试 |
| `tests/test_stage02_entry.py`、`tests/test_stage02_r2.py` | **R8** 去掉 `errors='replace'`，改为显式确定子进程编码 |
| `tests/test_negative_paths.py` | **R8** 用命名清晰的保留证据目录取代 `tempfile.mkdtemp`；去掉读取日志时的 `errors='ignore'`（不得静默丢内容） |
| `docs/stage02b1_revision_report.md`（新增） | 本文件：R1–R8 逐项关闭证据 |
| `README.md` | 追加修正批说明与新增运行方式 |

**未修改**：`optics/doe.py`、`optics/materials.py`、`optics/coordinates.py`、
`optics/propagation.py`、`optics/stage02_runtime.py`、`main_stage02.py`、
`config_stage02a.json`、阶段 01/02A 的物理模型与冻结验收项。
**未删除任何文件**；所有历史 run 保留。逐次改动都在代码注释里写明了「为什么改」。

## 3. R1 圆盘累计包含方窗角落（P1）——已关闭

**问题**：`find_first_radius_reaching` 对整个方窗排序累计，角落在 `r > Rlimit` 之外却
仍被算入；`Eabs_at_Rlimit` 取方窗终值。审核反例：强度只在 `r>155 μm` 时返回
`R80 = 185.31 μm`、`reached`，而 150 μm 圆盘能量为 0。

**修法**：先用实际轴范围求 `Rlimit = min(xmax, −xmin, ymax, −ymin)`，**只保留
`r ≤ Rlimit`** 的像素再分组累计；`Eabs_at_Rlimit` 因此对应真实圆盘；
径向分辨率改用明确的采样尺度（网格间距与圆盘内最小非零半径的较小者），
不再用浮点最小差冒充。`absolute_encircled_energy` 同时**拒绝** `R > Rlimit`。

**关闭证据**（`tests/test_stage02b1.py::TestAuditCounterexamples`）：

| 检查 | 修正前 | 修正后 |
| --- | --- | --- |
| 角落反例（能量只在 r>155 μm） | `reached`，R80=185.31 μm | **`not_reached`，radius=None，Eabs_at_Rlimit=0** |
| 方窗达标但内切圆不达标 | 误判 reached | **not_reached** |
| Jeon 540 nm `Eabs_at_Rlimit` | 0.8886123666 | **0.877347247399**（审核真值 0.8773472474） |
| `radius_scale_m` | 8.47e−22（浮点噪声） | **1e−6 m**（网格间距） |
| `R > Rlimit` 的评价半径 | 静默给出数值 | **抛 ValueError** |
| 高斯解析 E(R)、R50/R80（回归） | 通过 | **仍通过** |

**变化字段说明**：`R50`/`R80` 的**半径**值在默认 18 组上与旧报告一致
（审核已独立复核）；变化的是 `Eabs_at_Rlimit`、`radius_scale_*`、
以及超界时的拒绝行为。新 run 的 `metrics/psf_metrics.json` 中这些字段为修正值。

## 4. R2 保存重读没有独立核算功率（P1）——已关闭

**问题**：只检查保存的 `Pwindow/Pin` 范围。审核把 Jeon 540 的 `Pin` 与 `Pwindow`
**同时乘 2**（复场与强度不变），重读仍 pass。

**修法**：`verify_saved_identity_set` 现在

1. 用**实际配置 + 固定高度 + 该入射 λ**重建 `u1`，独立算出 `Pin`（相对容差 1e−12）；
2. 从**保存的强度与输出网格**重新积分 `Pwindow`（相对容差 1e−9）；
3. 分别比对并记录最差相对偏差；核对 η、峰值、振幅、n(λ)、全局相位、dtype、
   坐标、精确 λ、器件指纹、**传播距离 z**、**设计焦距 f**；
4. 高度文件按保存数组**独立重算内容指纹**，并核对设计 λ 图与参数；
5. 缺字段/损坏数据给出明确 fail 原因（不再依赖偶然 `KeyError`）。

**关闭证据**（`TestAuditCounterexamples`，每条都把结果写入保留的证据目录）：

| 注入 | 期望 | 实测 |
| --- | --- | --- |
| **Pin 与 Pwindow 同时 ×2** | 拒绝 | **fail（报告 Pin 与独立重算不符）** |
| 只改 Pwindow | 拒绝 | **fail（重积分不符）** |
| 改 η | 拒绝 | fail |
| 改 n(λ) | 拒绝 | fail |
| 改振幅 | 拒绝 | fail |
| 改 y 轴坐标 | 拒绝 | fail |
| 改 dtype（float32） | 拒绝 | fail |
| 改高度里的设计 λ 图 | 拒绝 | fail |
| 改高度内容（保留旧指纹字段） | 拒绝 | **fail（独立重算指纹不符）** |
| **真实自洽的合成样本** | 通过 | **pass** |

> 旧测试的「正常样本通过」已被改写为**真实自洽**：`Pin` 由该高度与该 λ 重建 `u1`
> 得到、`Pwindow` 由写入强度积分得到。**没有**为通过旧测试而放松验收。

**新 run 实测**：`worst_pin_relative_diff = 0.0`、`worst_pwindow_relative_diff = 0.0`
（容差 1e−12 / 1e−9），即保存值与独立重算**逐位相同**。

## 5. R3 配置传播距离被忽略（P1）——已关闭

**问题**：`propagate_device` 取 `profile.params['focal_length_m']` 作传播距离，
`optical.distance_m` 只被读取用于诊断。

**修法**：`propagate_device(..., distance_m, ...)` 由调用方显式传入配置 z 并**真正用于
传播**；日志分别打印 `实际 z` 与 `器件设计 f`；PSF 保存 `propagation_distance_m` 与
`design_focal_length_m` 两个字段，重读时逐一核对。**选择显式支持 z ≠ f**
（而不是拒绝），以便后续阶段使用。

**关闭证据**：

| 检查 | 实测 |
| --- | --- |
| 小网格 f=50 mm、z=60 mm vs z=50 mm | 复场相对 L2 = **1.883332**（显著不同） |
| 调用参数与直接参考一致 | `u2` 与 `fresnel_kernel_separable(..., z=50 mm, ...)` **逐元素相同** |
| 记录的 z 与 f | `distance_m=0.06`、`design_focal_length_m=0.05` |
| 默认 18 组 | z=f=50 mm，**与旧场一致**（见第 9 节对比） |

## 6. R4 部分模式、报告与 completed 相矛盾（P1）——已关闭

**问题**：`--only height` 保存运行退出 1（报告 `KeyError: 'unwrap'`），
但 `completion.json` 已经 `completed`/`all_checks_pass=true`；`all` 必需集合也未包含
分析与报告产物。

**修法**：

- 报告在 `try` 内、**状态确定之后**生成，且**按实际模式**撰写：子模式下第 3/4/5/6 节
  写「本运行是子模式，未产生该数据」，不再固定声称已完成 18 组；
- 报告生成、绘图、必需产物缺失一律 `failed` 且非零退出；
- **只有完整 `all`** 且全部必需项（含 `report_delivered`、
  `required_artifacts_delivered`）通过才是 `completed`；子模式标 `partial` 并写
  `partial_reason`；
- `checkpoint.json` 与 `completion.json` 使用同一 `state` 定义。

**关闭证据（实际 CLI）**：

| 模式 | 退出码 | status | report 交付 | 说明 |
| --- | --- | --- | --- | --- |
| `--only height` | 0 | **partial** | pass | 修正前是退出 1 + completed |
| `--only control` | 0 | **partial** | pass | |
| `--only preview` | 0 | **partial** | pass | 18 组身份 18/18、问题 0 |
| `--only analyze --from-run` | 0 | **partial** | pass | |
| 默认（all） | 0 | **completed** | pass | 7 张图 + 2 个指标文件全部交付 |

**注入失败**：报告生成异常已被测试覆盖（`report_delivered=fail` → `failed` + 非零），
缺失必需图表时 `required_artifacts_delivered=fail` 并抛错，不留下 `completed`。

## 7. R5 `--from-run` 分析入口没有实现（P2）——已关闭

**问题**：解析了 `--from-run` 但从未使用；合法 run 上耗时近零、退出 1。

**修法**：实现 `analyze_existing_run`：真正重读源 run 的全部产物并重算指标，
包含**兼容性检查**（配置关键项、入射波长列表、探测器坐标、精确 λ、器件指纹、
高度内容独立重算、`Iraw=|u2|²`）；不兼容明确失败、不混用；源 run **只读**，
输出写入**新的唯一 run**；分析前后对比源 run 全部文件的 SHA256 以证明未改动。

**关闭证据**：

| 场景 | 实测 |
| --- | --- |
| 合法 run `run_20261001_224235` | 兼容性通过，**重读 18 组身份**，指标重算，退出 0 |
| 源 run 是否被改动 | **全部文件 SHA256 前后完全相同** |
| 不兼容 run（阶段 01 的 run 目录） | **明确失败**并逐条列出缺失/不兼容项，退出 1 |
| 不存在/缺 `arrays/` 的目录 | 明确失败 |
| `--only analyze` 未给 `--from-run` | 明确报错，退出 2 |

## 8. R6 no-save 同时跳过显示（P2）——已关闭

**问题**：绘图全在 `run_dir is not None` 分支内，`--no-save` 即使请求 `--show-plots`
也不产生 figure。

**修法**：`need_plot = (run_dir is not None) or keep_figures`；需要时创建图、
不保存则把保存路径传 `None`；两者都不需要才跳过。`state` 记录
`figures_created` / `figures_saved` 便于核对。

**关闭证据**（mock 后端与 figure 生命周期，未真实弹窗）：

| 组合 | 实测 |
| --- | --- |
| show=True / save=False | **创建 7 张图**，`figures_saved=False`，未落盘 |
| show=False / save=True | 创建 7 张并保存 |
| show=False / save=False | 不绘图 |

> 说明：真实 `TkAgg` 弹窗在本环境会**阻塞**在 `plt.show()`（实测），
> 因此按审核要求用 mock 验证生命周期；**人工 GUI 验证仍另列未验证项**。

## 9. R7 不可靠方向仍输出角度数值（P2）——已关闭

**问题**：轴对称高斯判为不可靠，却输出约 49.46° 的浮点噪声方向；传统 PSF 同样保留数值角。

**修法**：`c3_rotation_metric` 在 `A3` 未达阈值时把 `alpha_wrapped_rad/deg` 置为
**`null`**；原始相位另用 `alpha_raw_rad/deg` 命名，并配
`alpha_raw_note` 明确「**不是**有效方向」。CSV 空缺处配原因文字、图线在 `null` 处断开；
`None` 不引起格式化崩溃。

**关闭证据**：

| 场景 | 实测 |
| --- | --- |
| 轴对称高斯 | `alpha_wrapped_deg=null`，`alpha_raw_deg=54.10`（标注无效） |
| 传统 Fresnel @540 nm | `A3=4.81e−17`，`alpha_wrapped_deg=null`，`alpha_raw_deg=57.21` |
| Jeon（可靠） | `alpha_wrapped_deg` 有值，且与解析三重角向一致到 1e−6 |
| 无环带能量 | 全部为 null，不崩溃 |
| 可靠/不可靠混合序列 | 不可靠点为 null，分段展开不跨间隙 |
| CSV/报告 | 空缺写原因，图线断开 |

## 10. R8 测试自动删除与编码环境依赖（P1）——已关闭

**问题**：测试用 `shutil.rmtree` 自动删除（审核未授权删除，只能拦截）；
且读端 UTF-8 不保证子进程按 UTF-8 输出，默认环境完整回归失败。

**修法**：

1. **不再删除**：新增 `tests/_stage02b1_support.py`，把证据写到项目内
   `results/_test_evidence/stage02b1/<用例_时间戳>/`，并放
   `EVIDENCE_RETAINED.txt` 说明来源；`test_stage02b1.py` 里 `rmtree`/`mkdtemp`
   已全部移除（现为 0 处）。
2. **双向确定编码**：新增 `child_env()` 同时设置子进程
   `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`、`PYTHONLEGACYWINDOWSSTDIO=0`，
   读端 `encoding='utf-8'`，并**移除** `errors='replace'`
   （编码不一致必须报错，不得用替换掩盖乱码）。
   两个基线测试文件改用该辅助函数。

**关闭证据（冻结后一次完整回归，普通 PowerShell，未预置任何环境变量）**：

| 项 | 实测 |
| --- | --- |
| 命令 | `Set-Location <项目根>; python -B -m unittest discover -s tests` |
| 环境预置 | **无**（普通 PowerShell，未设置 `PYTHONIOENCODING`；子进程编码由测试模块自己设定） |
| 实际项数 | **134 项**（旧基线 87 + 本批新增 47） |
| 结果 / 退出码 | **OK / 0** |
| 耗时 | **150.22 秒**（另一次运行 162.82 秒；随负载变化，不作速度保证） |
| 中文日志断言 | 能读到真实内容（`阶段 02A 开始执行` 等断言通过，**未用** `errors='replace'` 掩盖） |
| 编码由谁设置 | **由测试支撑模块显式设置**（`tests/_stage02b1_support.py`），不依赖用户预置环境；`describe_encoding_setup()` 记录实际配置 |

> 项数由审核时的 118 增至 134 的原因：R8 取消了自动删除，同时新增 R1/R2/R3/R7 反例测试。
> 审核记录的「117 通过 / 1 失败」来自**旧的**测试实现，该失败项已随 R8 修复关闭。

## 11. 报告口径与原文方向核对（第五、六节）

| 审核要求 | 处理 |
| --- | --- |
| 不把 10% 阈值称为“高阈值核心尺寸” | 报告改为「10% 阈值等效半径」与「50% 阈值等效半径」，并写明 **q=10% 低于 50%，覆盖更多翼/弱结构，是更宽的指标** |
| 420 nm 核步进写成实际值 | 日志与报告表格改为实际值：**420 nm 为 0.209589 rad**、660 nm 为 0.133375 rad（阈值 0.50），并注明「保守提示、不是收敛证明」 |
| 旋转方向与正文的对应 | 报告新增**第 12 节**：声明本坐标（x 右/y 上、逆时针为正、未镜像）；实测为逆时针；正文图 3 文字记为顺时针，**符号相反**；说明「统一角零点偏置无法解释反号」，**结论保持未解决**，不擅自改高度/相位/镜像数据 |
| 保留材料、网格、像元假设 | 报告第 8 节继续保留「实施假设」与「限制」，不称与图 3 全部参数完全一致 |

**未解决的核对**：本轮**未**逐页完成正文第 3/4 页与补充第 4 页图 3 的图像判读，
因此方向约定仍**未解决**；报告已列出待核对的具体页码与需要确认的观察面定义。
按任务书要求，**不**为迎合图像修改物理模型。

## 12. 与旧场的对比（默认传播不变时原场应一致）

| 项 | 结果 |
| --- | --- |
| Jeon 540 nm ηwindow | 0.888612（旧 run 与审核独立值一致） |
| 18 组峰值、ηwindow | 与旧 run 相同（默认 z=f=50 mm 未变） |
| 分析字段 | **允许并已纠正**：`Eabs_at_Rlimit`、`radius_scale_*`、不可靠角度的 `alpha` |
| R50/R80 半径 | 与旧报告一致（审核已独立复核） |

## 13. 未解决项与未验证项

1. **PyCharm 人工点击与真实图窗弹出未人工验证**。真实 `TkAgg` 在本环境会阻塞于
   `plt.show()`；R6 用 mock 验证生命周期，**不等同于人工验证**。
2. **旋转方向与正文图 3 的对应关系未解决**（见第 11 节）：仅靠角零点偏置不能解释反号；
   需要原文观察面/角向符号约定确认。
3. **未证明采样收敛**：18 组仍共用同一网格，属 02B-2；420 nm 核步进 0.2096 rad
   只是保守提示。
4. **`Eabs_at_Rlimit` 语义改善后，R50/R80 半径数值未变**，但若换用其它强度分布
   （例如能量集中在角落）行为已修正；本批未构造更多分布做穷举。
5. 旧 run（`run_20261001_224235` 等）**没有** `dx_m/dy_m` 等新字段，
   用它们做 `--from-run` 时输入间距会退化取坐标差并可能因此判指纹不符；
   新 run 一律保存精确间距。本轮已对旧源 run 用**本次配置间距**回退，
   并核对坐标数组逐元素相同后放行。
6. 材料吸收与界面反射、加工量化、相机像元积分仍未建模；像元 6.22 μm 只是硬件信息。

## 14. 停止点

02B-1 修正批完成，交回审核。**不**自动开始 02B-2（采样加密研究）。
