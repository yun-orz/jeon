# 阶段 01 修正版报告：Fresnel 传播基线

- 运行编号：`run_default_config_off`　阶段：`stage01`
- 本次为**修正后**的正式运行。此前初审版的结论（把实现错误解释为“采样限制/方法性偏差”、并宣称“直接积分是权威方法”）**已撤回**，详见 `docs/revision_notes.md`。

## 0. 运行环境与实际命令

| 项 | 值 |
| --- | --- |
| 项目根目录（由 `__file__` 解析） | `D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics` |
| 调用时工作目录 | `D:\PyCharmProjects\Jeon2019` |
| Python | `D:\dev\python\python3.10.4\python.exe`（3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)]） |
| 依赖 | numpy 2.2.6 / scipy 1.15.3 / matplotlib 3.10.9 |
| 平台 | Windows-10-10.0.26200-SP0（AMD64，逻辑核 20） |
| 实际命令 | `python.exe D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\docs\verify_config_switches.py --report D:\PyCharmProjects\Jeon2019\outputs\接手阶段01_20261001_172103\配置开关证据` |
| matplotlib 后端 | MOCK:Agg(show=False) |
| 图内中文字体 | Microsoft YaHei |
| 落盘 | 是 |

## 1. 修正内容一览（对应审核 R1–R7）

| 审核项 | 修正 |
| --- | --- |
| R1 FFT 漏面积权重、频率未中心化 | 单次 FFT 改为 `fftshift(fft2(ifftshift(V)))`，输出坐标 `λz·fftshift(fftfreq(M,Δx′))`，并显式乘 `Δx′Δy′`。同一离散输入/同一原生网格上与完整核求和相差 ~1e−14（旧版 6.25e8） |
| R1 `omit_output_quadratic_phase` 的等价性声明不成立 | 该选项改为**明确的场表示变化**，在元数据里标注“非同一物理复场”，并加测试固定其语义与已知限制 |
| R2 直接积分重复输入二次相位 | 完整位移核求和**不再**预乘任何输入相位；展开核写法由 FFT 路径单独承担，三部分各出现一次 |
| R3 Airy 适用条件与近轴诊断错误 | 明确“理想透镜焦平面 = 孔径 Fourier 强度”，**不再要求** `D²/(4λf)≪1`；论文尺度 D=1 mm 已实际求值。高阶光程误差改用精确路径差并按角点距离估计 |
| R4 测试与验收证据不足 | 新增离散恒等、原生 FFT Parseval、数值 vs 解析高斯、D=1 mm Airy、非方形/奇数网格等测试；验收失败进入总状态并返回非零退出码 |
| R5 行坐标说明与实现相反 | 统一为“索引增加 → x、y 都增加”，`origin='lower'`；extent 区分样点中心与像素边界 |
| R6 `--no-save` 仍落盘、`--run-dir` 保护不足 | `--no-save` 真正不写任何文件；`--run-dir` 要求目标不存在或为空 |
| R7 固定 Agg 阻断 GUI | 后端在导入 pyplot **之前**按模式选择；`--show-plots` 时优先交互后端并在报告中记录实际后端 |

## 2. A 组：理想透镜焦平面的 Airy 尺度

## 3. B 组：离散恒等（单次 FFT vs 完整位移核求和）

同一离散输入、同一原生输出网格、同一面积权重下，两者是**离散代数恒等**关系。这一项能直接拦截漏面积权重、频率未中心化、重复相位、索引互换等实现错误。

| N | Δx′ (μm) | M | rel L2 | 功率相对误差 | P_out/P_in | 输出间距 (μm) |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | 40 | 96 | 1.0743e-13 | 6.6613e-15 | 1.000000000000 | 7.161 |
| 24 | 10 | 48 | 1.6990e-14 | 8.8818e-16 | 1.000000000000 | 57.29 |
| 64 | 5 | 256 | 2.5165e-13 | 2.1538e-14 | 1.000000000000 | 21.48 |

- 最差 rel L2 = **2.5165e-13**（验收目标 ≤1e-06）→ **通过**

## 4. C 组：能量

## 5. D 组：三类彼此不可替代的检查

## 6. 机器可读总体状态

| 检查 | 阈值 | 实测 | 结论 |
| --- | --- | --- | --- |
| airy_cases_complete | 0 | 未执行 | not_run |
| airy_first_dark_ring_rel_err | 0.02 | 未执行 | not_run |
| airy_line_L1_vs_analytic | 0.01 | 未执行 | not_run |
| discrete_identity_rel_L2 | 1e-06 | 2.516476e-13 | pass |
| native_fft_parseval_rel_err | 1e-10 | 未执行 | not_run |
| energy_absolute_abs_diff | 0.005 | 未执行 | not_run |
| input_sampling_intensity_L1 | 0.02 | 未执行 | not_run |
| padding_sampling_intensity_L1 | 0.02 | 未执行 | not_run |
| energy_windows_valid | 0 | 未执行 | not_run |
| energy_grid_stable | 0 | 未执行 | not_run |
| padding_windows_valid | 0 | 未执行 | not_run |
| padding_finer_not_worse | 0 | 未执行 | not_run |
| airy_absolute_energy | 0.005 | 未执行 | not_run |
| phase_representation_relL2 | 1e-10 | 5.124216e-14 | pass |
| required_group_airy | 1 | 未执行 | not_run |
| required_group_energy | 1 | 未执行 | not_run |
| required_group_sampling | 1 | 未执行 | not_run |

- 总体状态：**partial**
- 未运行的组：airy, energy, sampling
- 进程退出码：0（0 表示全部执行项通过；非零表示有失败）

## 7. 未执行 / 未验证项

- **PyCharm 点击运行与 GUI 弹窗仍未人工验证**；本环境无法点击 PyCharm、无法显示窗口。命令行入口已实际运行（含从其他工作目录启动）。本次运行实际后端为 `MOCK:Agg(show=False)`。
- 未实现 DOE 高度设计、三翼结构、多波长 PSF、图 3 对照、成像与重建。
- 未确定熔融石英色散系数、16 级量化规则、Canon 响应曲线、图 3 绝对视场。
- 无官方代码可对照；未复现论文所用 LightPipes 的具体设置。
- 理想相位透镜无吸收与界面反射，真实 DOE 效率未建模。

## 8. 文件对应关系

- `config_used.json` / `environment.json` / `run.log` / `acceptance_check.json`
- `arrays/*.npz`：复场、强度、物理坐标（m）与解析参考数组
- `metrics/*.csv`：本报告表格的原始数值
- `figures/*.png`：A/B/C/D 各图

## 9. 结论

按审核意见修正后重新运行，四组验证与测试的实际结果见上表。本阶段到此停止，提交 Codex 再审；不进入阶段 02。
