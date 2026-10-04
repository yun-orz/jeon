# 阶段02A-R2接手完成与使用记录

日期：2026-10-01。执行与复核：用户授权后由原审核者直接接手。

**本批修正已完成并通过验证。完整87项测试通过，默认正式入口通过，C1–C7反例已关闭。**
这是连续DOE与三波长预览程序的可靠性验收；图3九波长、DOE采样收敛、加工量化、相机像元积分和重建仍未执行。

## 1. 实际运行与证据

正式结果：`D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\results\stage02a\run_20261001_215848_02`。
备份与证据：`D:\PyCharmProjects\Jeon2019\outputs\接手阶段02A-R2_20261001_214556`。

| 验证 | 实际结果 |
| --- | --- |
| 冻结版本完整unittest发现 | 87项通过，退出0，104.42秒 |
| 从项目外执行无参数默认保存入口 | 退出0，约12.16秒；每次生成唯一run |
| 项目外相对config＋完整no-save | 退出0；项目全部文件清单与内容SHA前后不变 |
| 新旧高度/控制/三PSF对照 | 对应原始数组逐元素相同；修改未改变光学公式或默认参数 |
| 正式run源码manifest | 与最终源码指纹一致 |
| 控制复场乘2的主入口反例 | 实际返回2；功率超上界被拒绝，不能被归一化曲线掩盖 |
| zero/NaN/wrong-shape控制场 | RuntimeError、failed状态及失败报告；不是未定义变量造成的偶然失败 |
| 显示/保存开关 | 配置、CLI合成与后端选择mock通过；图形生命周期通过 |
| GUI/PyCharm人工点击 | 本批未人工验证，不用mock结果代替人工验证 |

耗时会随负载变化。同一工作过程中完整回归还曾约33/63秒、正式入口约4.5秒，不作固定速度保证。
前几轮日志和run保留。最后文件名边界补充前的run不能替代冻结版本的源码指纹；最终证据以 `最终执行与独立核验.json` 为准。

环境：Windows/CPU，Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；具体路径与平台字符串见正式run的environment.json。

## 2. 关闭的具体问题

| 项 | 实现与证据 |
| --- | --- |
| C1路径 | CLI直接使用统一配置解析；相对config及run-dir均按项目根，不按启动cwd |
| C2显示 | 导入pyplot前读取配置并合成CLI；默认Agg，用户请求时探测Tk依赖和隐藏窗口，再选择TkAgg；不可用时记录原因并回退Agg |
| C3保留 | 手动指定非空run-dir在写日志之前拒绝；直接计算函数也拒绝已有产物；测试旧文件SHA/清单不变 |
| C4波长 | 本阶段预览接口明确限定整数nm，拒绝540.05nm；缓存只精确复用同一请求值；整数文件标识去重，防止浮点容差引起同名覆盖 |
| C5功率 | 控制和预览共用shape、有限、非零、模平方和0＜Pwindow/Pin≤配置上界检查；控制失败立即停止，不继续九点或归一化补救 |
| C6报告 | 必需runtime/acceptance字段、有限offset受检查；报告使用实际控制λ、阈值和材料值；未执行N3器件的指纹为not_run |
| C7完整性 | 显式必需检查集合，固定器件终检，保存后重读数组/物理坐标/精确λ/原始场/高度/功率；缺项不得completed；计算保存与物理状态分开记录 |

新增模块 `optics/stage02_runtime.py`；新增17项专项测试在 `tests/test_stage02_r2.py`。
已有生产代码仅局部修改main_stage02.py，未改材料、高度、坐标、传播或阶段01配置；README只追加本记录入口。
没有删除文件，也未清空/覆盖既有docs报告或历史run。

## 3. 面向PyCharm的手动运行

1. 在PyCharm打开 `D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics`。
2. 选择解释器 `D:\dev\python\python3.10.4\python.exe`，或安装requirements.txt对应依赖的本地解释器。
3. 打开main_stage02.py，点击运行。默认执行高度、550nm控制、420/540/660nm预览并保存。
4. 参数集中在config_stage02a.json，内部长度m。n、spacing_m、half_width_m必须相互一致。
5. 默认不弹窗；结果在results/stage02a的本次run/figures。可直接用Windows图片查看器打开PNG。
6. 配置runtime.show_plots=true或传--show-plots，程序尝试TkAgg；无Tk/窗口系统则回退并明确记录未弹窗。本批未做真实弹窗人工确认。

从任意cwd使用绝对入口亦可：

```powershell
& 'D:\dev\python\python3.10.4\python.exe' -B 'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\main_stage02.py'
& 'D:\dev\python\python3.10.4\python.exe' -B 'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\main_stage02.py' --config config_stage02a.json --no-save
```

no-save不创建run/磁盘日志/报告；显示与保存是独立开关。
only=height/control/preview成功属于partial，未运行项not_run；完整all全部必需项通过才completed；失败非零退出。
控制数据保留历史文件名 `fresnel_control.npz` 和检查键 `control_550nm`；实际照明波长以数组wavelength_m及动态图/报告为准，不从检查键猜参数。

## 4. 物理结论与未验证范围

固定高度与相位响应继续使用正文式(3)、(4)、(9)–(12)，材料采用已记录的Malitson实施假设。
高度仅生成一次，各λ变更(n(λ)−1)/λ的相位比例，传播的是复场；相机强度取模平方。
控制暗环仍约33.55951μm，相对解析33.54092μm误差0.05542%。三波长方窗能量比仍约79.54/88.86/72.55%。
有限窗口的能量比不是实际加工效率。峰值归一化图只观察形状，不修改raw I或Pin。
目前尚未证明N3器件的输入/输出采样收敛，亦未证明旋转角度、顺时针趋势和尺寸恒定。

本轮按02A-R2批次要求停止。后续可进入02B-1九波长＋传统Fresnel对照；先生成真实18组场并评价角度/尺寸/能量，再单独审核02B-2加密研究。
