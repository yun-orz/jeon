# Jeon2019 —— 《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》复现

本仓库是对 Jeon et al., ACM TOG 38(4), 2019 的**分阶段可审核复现**。
每个阶段先出结果、再交独立审核，审核通过后才进入下一阶段；所有历史结果、
失败记录与审核证据**原样保留**，不覆盖、不删除。

> **当前状态（2026-10-07）：停在 M3 数据与最终算子交付完成；M4 可恢复训练器尚未启动。**
>
> 正式流程位于 [`outputs/jeon2019_optics/reproduction_v1/`](outputs/jeon2019_optics/reproduction_v1)，
> 当前协议已冻结为 M0–M7 八个里程碑。
>
> **阅读入口：[`START_HERE.md`](START_HERE.md)**（当前状态与正式交付链接）、
> [项目地图与复现路线](docs/PROJECT_GUIDE.md)（目录、模块输入输出、命令副作用）。
>
> 仓库同时保留两条历史线：`outputs/jeon2019_optics/` 的 **stage01/stage02** 早期光学工程，
> 以及 **G0–G5** 小规模 CPU 闭环。它们记录结论如何变化，不再是当前开发主线。
>
> ⚠️ Git 精选结果与本地正式来源**并非同一清单**。数组（`.npz`/`.npy`）、模型权重、
> 原始数据与 `work/environments`（GPU 环境）**不随仓库分发**，因此克隆后不能直接运行完整流程。

## 里程碑状态（M0–M7）

状态来自既有正式验收记录；本轮整理只核对文件与指纹，未重跑 GPU、光学、数据处理或训练。

| 里程碑 | 状态 | 正式依据 |
| --- | --- | --- |
| **M0** 协议与任务包冻结 | 通过 | [M0 报告](outputs/jeon2019_optics/results/reproduction_v1/m0/run_20261006_230835_592a12d3/report.md)、[M0 索引](outputs/jeon2019_optics/reproduction_v1/m0_index.json) |
| **M1** GPU 环境与执行一致性 | 通过（含后续修复与重验） | [M1 修复报告](outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/M1_fix_report.md)、[修复回执](outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/fix_receipt.json) |
| **M2** 物理光学与 RGB 算子 | 通过 | [M2 父包报告](outputs/jeon2019_optics/results/reproduction_v1/m2/run_20261007_104321_03d47940/report.md) |
| **M3** 数据准备与最终算子 | 通过 | [M3 正式运行](outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/report.md)、[M3 交付说明](outputs/jeon2019_optics/results/reproduction_v1/m3/delivery_20261007_160046/M3_delivery.md) |
| **M4–M7** | 未启动 | 训练器 → 预演诊断 → 两器件正式训练 → 封存测试评价 |

**M1 特有情况：** 首次验收通过后，独立审核发现两处入口缺陷（下载前缀缺失时检查器抛
`KeyError`；报告所述恢复命令实际不可用）。二者已修复并新增回归测试。随后 CUDA 曾一度
不可用，原因是**显卡被切到安静模式使独显从硬件树消失**（设备报 Code 45），切回后经实际
CUDA 自检恢复（`cuda_available=true`、RTX 4060 Laptop 8 GiB）。详细时间线与证据见
[M1 修复报告](outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/M1_fix_report.md)。

## M3 已交付什么

- **数据版本**：238 幅场景 —— 203 训练 / 25 验证 / **10 幅 KAIST 封存测试（留到 M7）**，
  以及 **30,000 条 256×256 训练补丁索引**（索引按需读取，未复制补丁文件）。
- **最终算子**：绑定该数据版本与训练集共同增益的 Jeon / Fresnel 两套算子包
  （共同增益 `0.003682718635788138`）。M2 父包保留为来源，不直接当作训练包（其共同增益尚未拟合）。
- **交接对象**：[场景清单](outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/scene_manifest.json)、
  [数据版本](outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/data_version.json)、
  [补丁索引](outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/patch_index.jsonl)、
  [测试封条](outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/test_seal.json)。
- **当前数据配置**：`config_m3_masked_index_20261007.json`。根目录的原始 `config_m3.json`
  属旧阶段配置，当前数据版本使用 masked_v3 配置。
- **ICVL 官方数据获取与只读验收**：[获取与验收报告](work/datasets/reproduction_v1/icvl_acquisition/acquisition_report.md)
  —— 官方 202 幅 MAT 全部取得，逐文件 SHA256 与来源仓库自述摘要 **202/202 一致**；
  HDF5 元数据 **202/202** 通过；420–660 nm 25 波段齐全。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `START_HERE.md` | **当前入口**：状态、正式交付链接、运行前确认 |
| `docs/PROJECT_GUIDE.md` | 项目地图：模块输入输出、命令副作用、M4–M7 路线 |
| `outputs/jeon2019_optics/reproduction_v1/` | **当前正式流程**：M0–M7 源码、配置、协议、测试、任务书 |
| `outputs/jeon2019_optics/reproduction_v1/main.py` | 统一命令入口（无参数等同 `check`） |
| `outputs/jeon2019_optics/results/reproduction_v1/` | 当前正式结果与审核证据（M0–M3） |
| `outputs/jeon2019_optics/optics/`、`main_stage*.py` | 早期 stage01/stage02 光学工程（历史） |
| `outputs/jeon2019_optics/results/g0…g5*/` | G0–G5 小规模 CPU 闭环（历史） |
| `outputs/审核_*`、`复审_*`、`接手*`、`执行_*` | 审核方与执行方的独立证据 |
| `baseline/g0_20261005/` | G0 冻结快照说明 |
| `work/` | 分析脚本、诊断日志与数据获取证据（论文 PDF 不随仓库分发） |
| `CURRENT_STATUS.md` | 按时间追加的研究状态记录（含各阶段历史结论） |

## 怎么运行

```powershell
# 依赖：Python 3.10、numpy、scipy、matplotlib
pip install -r outputs/jeon2019_optics/requirements.txt

cd outputs/jeon2019_optics/reproduction_v1

# 无参数等同 check：检查环境与来源，并写入一条新的检查记录
python -B main.py

# 核对现有 M2/M3 来源（会实际读取来源，不只是显示文字）
python -B main.py report

# 全量测试
python -B -m unittest discover -s tests
```

**命令副作用须知：** `check` 会写新的结果记录，`report` 会读取并核对来源，
`optics` 与 `data` 会**重跑**相应工作（只看现有结果时无需执行）。
`smoke` / `train` / `evaluate` 目前是**预留命令**，实际执行会报未实现。
每次运行都会新建 `results/<阶段>/run_<时间戳>/`，**不覆盖也不清理旧 run**。

**运行前请确认：**

- CPU 解释器 `D:/dev/python/python3.10.4/python.exe`；
  GPU 解释器 `work/environments/jeon_gpu_cu121/Scripts/python.exe`（训练用后者）。
- **需保持 NVIDIA 独显启用**。已有验收记录**不保证**每次启动时硬件都可用 —— 曾因切换到
  安静模式导致独显从硬件树消失、CUDA 不可用。
- ICVL 原始数据位于 `H:/我的云端硬盘/Jeon2019_data/reproduction_v1/icvl/mat`，
  该盘为 Google Drive 映射盘，需确认已挂载且文件可读。

## 物理与实现约定（重要）

- 内部长度单位一律 **米**；坐标约定：`x` 向右、`y` 向上、`z` 朝探测器，
  `theta = mod(atan2(y,x), 2*pi)`，**逆时针为正**；数组 `a[j,i]` ↔ `(x[i], y[j])`，
  绘图 `origin='lower'`。
- 传播用可分离的完整位移菲涅耳核；相干复振幅求和后取 `Iraw = |u2|^2`，
  保留 `dx'*dy'` 与 `1/(lambda*z)` 因子；**不叠加**额外理想薄透镜相位。
- 熔融石英色散用 Malitson (1965) Sellmeier；空气折射率取 1。
- `Eabs(R) = sum_{r<=R} Iraw * dA / Pin`，`Pin` 取自孔径透过场；
  绝对包围能量半径 `R50/R80` 与峰值相对阈值面积 `r_eq,q` 是**两个不同口径**，分别报告。
- **ICVL 数据实际轴序**：`rad` 按存储为 `[31, 1392, W]` —— **谱轴在最前**，`dtype=float64`，
  `bands` 为 `[31,1]`；宽度在 1018–1300 间变化。读取方**不得**假定 `(H,W,C)` 轴序或固定宽高。

## 本仓库收录范围（精选）

工作区中的大型二进制**不随仓库分发**（体积巨大且可由源码与配置重算）：

- ✅ 全部源码、配置、协议、测试、文档、审核证据与运行日志
- ✅ 每个阶段的**代表性 run** 的图（PNG）、指标（JSON/CSV）、报告
- ✅ 数值指标完整保留，因此结论可核对
- ❌ `.npz`/`.npy` 数组、`.pt`/`.pth` 权重、原始数据（ICVL/Harvard/KAIST）、
  vendored 二进制依赖、`work/environments` GPU 环境、
  论文 PDF（版权）、IDE 个人配置、字节码缓存

代表性 run 清单见 [`_select_runs.json`](_select_runs.json)；排除规则见 [`.gitignore`](.gitignore)；
数组数据保留说明见 [`_include_arrays.md`](_include_arrays.md)。

## 已知未解决项

- **论文光学定量对齐未通过**（`paper_alignment_passed=false`）：
  旋转方向与论文图 3 的对应关系未解决 —— 本仓库坐标下实测为**逆时针**，
  论文正文文字记为顺时针，符号相反；统一角零点偏置无法解释反号，
  需要原文观察面/角向符号约定确认。详见
  `outputs/jeon2019_optics/docs/stage02b1_revision_report.md` 第 11–13 节。
- 旋转核心的形状尺寸与绝对包围能量半径不是同一个量；尚不能宣称论文图 3 的方向、
  尺寸和全部参数已严格对齐。
- **真实相机响应未取得**：G2 的 RGB 响应为 `synthetic_not_calibrated`。
- **作者精确网络与训练协议未知**：G3 的阶段数、通道表、激活与正参数化等为明确记录的
  实施假设，不是作者官方架构。
- **测量共同增益尚未独立验证**：M2 阶段共同增益未拟合，M3 使用训练集拟合值。
- **M4–M7 未启动**：训练器、预演诊断、两器件正式训练、封存测试评价均未执行；
  **封存测试集至今未被读取**。
- PyCharm 人工点击与真实图窗弹出**未人工验证**。
- 材料吸收、界面反射、实际加工掩膜及真实相机响应仍未建模或取得。

## 说明

- 本仓库为复现研究用途，论文原文与其中图表版权归原作者与 ACM 所有。
- 各阶段历史结论保留在 [CURRENT_STATUS.md](CURRENT_STATUS.md) 中，**以文件末尾最新记录为准**。
- 新生成的检查目录、失败 run 与时间更晚的目录**不自动**成为新的正式数据版本。
