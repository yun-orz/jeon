# Jeon2019 —— 《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》复现

本仓库是对 Jeon et al., ACM TOG 38(4), 2019（论文 117）的**分阶段可审核复现**。
每个阶段先出结果、再交审核，审核通过后才进入下一阶段；所有历史结果与审核证据**原样保留**。

论文的旋转衍射编码：用一片连续面型的 DOE 把入射光按波长分配不同的三翼取向，
从而在**单次拍摄**中同时编码光谱与空间信息。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `outputs/jeon2019_optics/` | 主项目：源码、配置、测试、文档、结果 |
| `outputs/jeon2019_optics/main.py` | 阶段 01：传播基线与采样对照 |
| `outputs/jeon2019_optics/main_stage02.py` | 阶段 02A：DOE 高度设计 + 单色控制检验 |
| `outputs/jeon2019_optics/main_stage02b.py` | 阶段 02B-1：九波长对照 |
| `outputs/jeon2019_optics/main_stage02b2.py` … `main_stage02d2.py` | 后续阶段入口 |
| `outputs/jeon2019_optics/optics/` | 物理模块：坐标、材料色散、DOE 设计、传播、指标 |
| `outputs/jeon2019_optics/tests/` | 单元与反例测试 |
| `outputs/jeon2019_optics/docs/` | 各阶段中文完成/修正报告 |
| `outputs/jeon2019_optics/results/` | 运行结果（本仓库只收录代表性 run，见下） |
| `outputs/审核_*`、`复审_*`、`接手*`、`执行_*` | 审核方与执行方的独立证据 |
| `work/` | 分析脚本与诊断记录（论文 PDF 不随仓库分发） |

## 怎么运行

```powershell
# 依赖：Python 3.10、numpy、scipy、matplotlib
pip install -r outputs/jeon2019_optics/requirements.txt

cd outputs/jeon2019_optics

# 阶段 02B-1：两张固定 DOE 高度 × 九个波长 = 18 组真实衍射复场
python -B main_stage02b.py

# 不写任何项目文件的自检
python -B main_stage02b.py --no-save

# 只做高度设计 / 单场试算 / 九波长预览（子模式，结果为 partial）
python -B main_stage02b.py --only height

# 用已有 run 重新分析（源 run 只读，输出到新的唯一 run）
python -B main_stage02b.py --only analyze --from-run results/stage02b1/run_20261002_163101

# 全量测试
python -B -m unittest discover -s tests
```

每次运行都会新建 `results/<阶段>/run_<时间戳>/`，**不覆盖也不清理旧 run**。

## 物理与实现约定（重要）

- 内部长度单位一律 **米**；坐标约定：`x` 向右、`y` 向上、`z` 朝探测器，
  `theta = mod(atan2(y,x), 2*pi)`，**逆时针为正**；数组 `a[j,i]` ↔ `(x[i], y[j])`，
  绘图 `origin='lower'`。
- 传播用可分离的完整位移菲涅耳核；相干复振幅求和后取 `Iraw = |u2|^2`，
  保留 `dx'*dy'` 与 `1/(lambda*z)` 因子；**不叠加**额外理想薄透镜相位。
- 熔融石英色散用 Malitson (1965) Sellmeier；空气折射率取 1。
- `Eabs(R) = sum_{r<=R} Iraw * dA / Pin`，`Pin` 取自孔径透过场；
  绝对包围能量半径 `R50/R80` 与峰值相对阈值面积 `r_eq,q` 是**两个不同口径**，分别报告。

## 本仓库收录范围（精选）

工作区**完整**数据约 16 GB，其中 96% 是 `.npz` 数组（复场/强度/高度），
可由源码与配置重算，因此**不随仓库分发**。仓库收录：

- ✅ 全部源码、配置、测试、文档、审核证据
- ✅ 每个阶段的**一个代表性 run** 的图（PNG）、指标（JSON/CSV）、报告与日志
- ✅ 数值指标（JSON/CSV）完整保留，因此结论可核对
- ❌ `.npz` 大数组、论文 PDF（版权）、测试自动产物、IDE 个人配置、字节码缓存

代表性 run 清单见 [`_select_runs.json`](_select_runs.json)；
排除规则见 [`.gitignore`](.gitignore)；数组数据的保留说明见
[`_include_arrays.md`](_include_arrays.md)。

## 已知未解决项

- **旋转方向与论文图 3 的对应关系未解决**：本仓库坐标下实测为**逆时针**，
  论文正文文字记为顺时针，符号相反；统一角零点偏置无法解释反号，
  需要原文观察面/角向符号约定确认。详见
  `outputs/jeon2019_optics/docs/stage02b1_revision_report.md` 第 11–13 节。
- PyCharm 人工点击与真实图窗弹出**未人工验证**。
- 采样收敛性未证明（属 02B-2 范围）。
- 材料吸收/界面反射、加工量化、相机像元积分未建模。

## 说明

本仓库为复现研究用途，论文原文与其中图表版权归原作者与 ACM 所有。
