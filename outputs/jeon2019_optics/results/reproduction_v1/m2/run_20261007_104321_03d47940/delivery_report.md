# M2 最终交付审核

状态：**passed**。正式交付目录：`D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\results\reproduction_v1\m2\run_20261007_104321_03d47940`。

Jeon 与 Fresnel 均完成全部 25 波段、1 μm/0.5 μm 输入及 q4/q8 四组比较。共 50 个器件波段、200 组采样比较全部通过；正式物理计算及算子数值审核耗时 120.53 秒。

默认 main.py check 已自动选中本交付目录并通过；M0 历史保护、G0 源码、协议以及 M2 合同/配置/核/响应/映射指纹全部通过。19 项独立单元测试通过（M0 14 项、M2 5 项），保存数组的独立重算审核通过。

## 原阈值与实际最大误差

| 指标 | 最大误差 | 原阈值 |
|---|---:|---:|
| pixel_l1 | 0.001372374268 | 0.005 |
| pixel_l2 | 0.0009451732815 | 0.01 |
| kernel_l1 | 0.001374036672 | 0.005 |
| kernel_l2 | 0.0009551036312 | 0.01 |
| eta_absolute | 3.00398558e-05 | 0.001 |
| angle_deg | 0.01554193684 | 0.5 |
| radius_pixels | 0.1494876017 | 1.0 |

角度可靠性、R50/R80 状态匹配均按 G1 原定义审核，未放宽任何阈值。无可靠方向的 Fresnel 轴对称结构不宣称可靠旋转角。

## 核、单位与响应

| 方法 | 有限窗口效率范围 | 面积映射质量绝对误差 | FP64 核对最大相对误差 |
|---|---:|---:|---:|
| jeon | 0.864772107–0.942122440 | 2.22e-16 | 2.41e-12 |
| fresnel | 0.872712741–0.983410156 | 1.11e-16 | 7.86e-12 |

原生核为 97×97、6.22 μm；训练物理核为 49×49、12.44 μm。映射按二维像元面积交集分配质量，保持几何原点；目标窗外侧的空面积保存在 area_mapping.npz。独立矩形遍历、中心/偏置/边缘点质量、对称核和质心核对通过。任意分布的离散质心量化偏差实际记录在 mapping_audit 中，未冒称严格逐核质心恒等。

核值为探测器像元功率/入射孔径功率，核质量就是有限窗口捕获效率，未逐波段归一化到 1。输入为相对能量谱密度/nm，每个中心代表一个 10 nm 谱带，带能量=谱密度×10 nm；等效谱带边界为 415–665 nm。采用中心波长单色核与分段常数近似，不宣称带内连续积分已收敛。带能量输入等价地使用单位权重，禁止再次乘 10；未使用 λ/540 光子换算。

RAWtoACES Canon EOS 5D Mark III 原始 JSON 为 380–780 nm/5 nm 相对响应，按实际表选取 420–660 nm。共同相对尺度保持原样，没有独立通道或逐波段放大；不是绝对 QE，也不是 Jeon 作者标定。原始文件及 Apache-2.0 许可证已保存，访问内容 SHA、模型和来源见 response_metadata.json。

固定 CW/N3 连续高度指纹为 3469c7a4717a765d31e0433d9180c015520e00b99d08a0c656e791ac2b917f51。D=1 mm，f=z=50 mm；Fresnel 固定 550 nm 设计。两器件共享已审核材料、传播、采样、探测器和响应。Sellmeier 公式及全部系数/温度/单位见 effective_config.json；源码及历史审核来源见 source_fingerprints.json。未按无标尺图形调参。

## M3 交接

**共同训练增益保持 pending、值为 null；M2 到此停止。** 两个包是审核通过的物理包，不是已拟合最终训练包。M3 只能从训练集拟合两器件共同增益，再生成包含本物理包父指纹的新最终包；不得直接改写这些包。正式模型须从头训练，旧 G1–G5 模型只作历史诊断。

NumPy/Torch 前向、零延拓伴随内积、自动微分及中央差分方向导数的 FP64 相对误差全部 ≤1e-10。pending 状态调用最终训练算子会拒绝。

采样审核是全 25 波段的有限窗口输入/像元求积检查；历史三个控制波长的 5120 审核不扩展为无限域或全 25 波段填充收敛。paper_alignment_passed=false。

## 实际可执行命令

以下命令按项目根目录运行；重跑 optics 会创建新目录。

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\main.py" "optics" "--config" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\config_m2.json"
```

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\tests\audit_m2_result.py" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\results\reproduction_v1\m2\run_20261007_104321_03d47940"
```

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\main.py" "check"
```

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\main.py" "report"
```

```powershell
& "D:\dev\python\python3.10.4\python.exe" "-m" "unittest" "discover" "-s" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\tests" "-p" "test_m2.py" "-v"
```

```powershell
& "D:\dev\python\python3.10.4\python.exe" "-m" "unittest" "discover" "-s" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\tests" "-p" "test_m0.py" "-v"
```

## 文件

- jeon_package.json / fresnel_package.json：物理包合同、单位与指纹。
- *_native.npy / *_training.npy：原生/半尺寸 25 波段核。
- audit.json：逐波段采样、数值检查、验收与未完成列表（本目录为空）。
- independent_audit.json：保存数组及二维面积交集独立重算。
- delivery_manifest.json：最终检查结果与本报告/审核工具内容指纹。

之前来源失败、代码修订和中断运行全部保留，不作为正式交付；中断运行的 interruption.json 记录已完成和未完成项。无删除操作；最终历史保护检查确认旧模型、核和器件指纹未变。
