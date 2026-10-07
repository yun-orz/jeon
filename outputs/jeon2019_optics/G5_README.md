# G5：验证集指标诊断（尚未建立正式Jeon性能基线）

这是“Jeon光学编码复现＋基础重建验证”的后续评价。G4最佳模型冻结不再训练，用原img4的两个验证上下文重载推理，并逐元素核对保存预测。img4参与过epoch选择，不能称独立测试场景。只评价每个128×128上下文中央32×32×25区域；halo不保证多阶段网络不受有限域边界影响。

## PyCharm运行

打开本地项目，解释器选Python 3.10.4，沿用requirements_g3.txt。右键运行main_g5.py，无参数CPU运行；从其他启动目录调用也可。入口按自身位置读取config_g5.json，不需要聊天、Harness、Notebook、Docker或网络。第一次需要已有G2/G3/G4正式结果与Harvard本地数据。

命令行：`python -B main_g5.py`；`--no-save`只计算不保存；`--show-plots`手动显示图像（窗口探测回退情况写入结果）。PyCharm参数框也可填这些参数。config_g5.json的source_g4/source_audit按工程目录解析；cpu_threads控制线程；data_range默认1，修改会改变PSNR和SSIM定义；zero_norm_threshold是SAM零向量阈值；runtime控制保存/显示；plots.dpi控制保存分辨率。

## 数值定义与来源

目标沿用G4：前25波段420:10:660nm、Harvard相对灵敏度校正，再按能量谱解释乘λ/540换算相对光子表示，除以仅训练块拟合的P99尺度。原始绝对单位尚未标定，不能称真实绝对辐亮度。没有逐图、逐波段归一化，也不裁剪负预测或大于1的值；结果记录这些计数。

- MSE：所有被评价像元及波段误差平方均值。PSNR=10log10(L²/MSE)，L=1是训练尺度下预先固定的参考幅度，**不是测试图实际最大值**；训练P99不保证目标小于1。汇总主值按两个等大小ROI合并MSE计算，不是逐波段PSNR的算术平均。完全一致时PSNR数学上为正无穷，JSON写null并附明确状态，CSV相应为空。
- SSIM：Wang等2004式(13)，11×11、σ=1.5高斯权重，K1=.01/K2=.03，人口加权矩（无N/(N−1)修正），L同PSNR。每波段在32×32 ROI内valid卷积，所以局部图22×22；不延拓、不下采样，然后对空间、波段、块等权平均。根据公式独立实现，不复制作者MATLAB源码。[作者说明与原论文链接](https://www.cns.nyu.edu/~lcv/ssim/)。
- SAM：arccos(点积/(两谱范数))；仅限制余弦的浮点误差到[-1,1]，不截断光谱。保存rad与deg，主汇总为有效像元的平均rad。任一范数≤1e−12时无定义，图中NaN、JSON均值仅取有效像元，另报无定义数量；全部无定义时均值null。正比例谱角为0，正交为π/2，反向为π。[ENVI官方角度说明](https://www.nv5geospatialsoftware.com/docs/SpectralAngleMapper.html)。负预测也照原值算角，并单独报计数。

PSNR观察幅度误差，SSIM观察各波段的局部空间结构，SAM观察一个像元的25维谱形；SAM对整条谱乘正系数不敏感，但乘波长这一逐波段变换会改变角度。因此不能把相对光子谱的SAM当成能量谱SAM而不说明。

## 论文比较边界

本地正文work/papers/main.pdf第8页表1与第9页表2：作者十幅测试谱图的Ours为35.88dB/.93/.12。当前核对到的表格没有明确SAM单位，不能仅凭习惯断言.12为rad；本程序显式声明自身rad。原文指标尺度/逐图逐波段聚合和SSIM边界设置未完整公开，待进一步核对。这里不生成“达到论文”结论或公平差值排名。

尚有差异：G1光学尺寸定量对齐未完成、G2使用合成RGB响应、G3有网络实现假设、G4只有两场景12步训练且验证选模型、不是原KAIST十图协议、目标谱表示未对齐。paper_alignment_passed始终false。**G5指标程序已运行，不等于图示G5正式Jeon simulation baseline验收通过；暂不进入G6新DOE。**

## 输出与验证

每次保存建立全新results/g5/run_*，不覆盖旧结果。metrics/scores.json保存每块每模型指标；per_band.csv便于查看25波段；protocol.json说明全部约定；validation.json保存汇总、环境、重载核对及限制；arrays/g5_evaluation.npz保存目标、未裁剪预测、指标图和有效掩码；figures自动保存逐波段趋势与空间图；source_manifest/source_evidence保留源码与G4来源SHA。

正式运行结果：未训练初始化15.179838dB、SSIM .528299、SAM .494921rad；最佳epoch2为17.975286dB、SSIM .595462、SAM .463059rad。仅此小样本验证诊断有改善，不说明独立测试泛化或论文性能。

实际环境：CPython3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9、PyTorch2.14.1+cpu。无需scikit-image新增依赖。6项解析/独立窗口测试；独立逐窗口中心化SSIM及atan2光谱角审核；异启动目录无保存SHA核对。正式入口与模型重载已运行，人工PyCharm点击、GUI/GPU未验证。完整审核见outputs/执行_G5_20261006/最终审核.json。
