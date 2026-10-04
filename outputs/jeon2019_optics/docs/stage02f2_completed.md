# 02F-2：合成含噪电子测量的基础重建

日期：2026-10-04。范围为“Jeon光学编码复现＋基础重建验证”，不是论文网络或双孔径。continuous与nearest_depth表示单孔径连续高度/最近深度级制造表示，不是双通道。

## PyCharm运行

解释器使用Python3.10，安装requirements.txt依赖，打开main_stage02f2.py点击运行。参数集中在config_stage02f2.json，无参数默认CPU、G=10000、六组器件/场景、两方法共12项。所有路径基于入口位置，无Harness/Notebook/Docker/云端依赖。

终端：python -B main_stage02f2.py。--config指定配置，--no-save不写结果，--show-plots请求显示。结果在results/stage02f2的新唯一目录；失败/预算目录保留，不自动选择最新目录作为已审核来源。真实PyCharm点击和GUI尚未人工验证。

## 物理、单位和目标

沿用固定DOE高度及波长传播核；不旋转图片或为每个波长重设计，非相干波段按强度卷积叠加。X仍是相对波段积分辐射功率。

A_e X=G*sum_b [(QE_b/QE_ref)*(lambda_b/lambda_ref)*response_b*(K_b*X_b)]，再裁剪。参考540nm，三波段420/540/660nm。单光子能量hc/lambda，长波在相同功率下光子数更多；QE决定记录电子的效率。

G=10000、QE=0.5、读出标准差2电子、背景B=0均为合成假设，非相机标定。保留有限窗口核通光效率，不重新归一化核，不再乘像元面积；共享探测器物理坐标。

Z是02F-1同一次泊松计数＋高斯读出样本，两方法共享Z。负读出值保留，目标为Z-B。预测A_e X+B、残差预测-Z为电子数，恢复X仍为相对功率。

非加权：min_{X>=0} 0.5||A_e X-(Z-B)||²+0.5 alpha||X||²。
固定权重：min_{X>=0} 0.5||W[A_e X-(Z-B)]||²+0.5 alpha||X||²。

v_hat=max(max(Z,0)+sigma_read²,v_floor)，v_floor=4电子数平方，W=1/sqrt(v_hat)。只对方差估计中的计数部分取非负，目标Z不剪零。Z已含背景计数，不重复加B。W仅求解前计算一次，来自同一观测、有偏且相关，只是Gaussian近似基础验证，不是精确Poisson-Gaussian似然，不声称严格卡方置信。

加权前向W A_e，伴随A_e^T W^T，先乘权重再回投影。岭梯度A_e^T W²(A_e X+B-Z)+alpha X。G与波长/QE响应同时进入前向/伴随，只乘一次。

## 参数与数值验收

独立为每种目标估计L_data，固定alpha_factor=1e-3、alpha=alpha_factor*L_data。power.seed=2019，最多100次、相对门槛1e-8；初始L=1.05*(L_data+alpha)。幂迭代只是估计，回溯负责验步长。两种目标实际alpha不同，这是预声明相对规则，不是共同实际alpha或最优调参。

求解接口不接收truth、理论方差或无噪声期望；真值仅作来源物理核验与求解后评价。测试更换外层真值时恢复和alpha逐值不变。全零初值，最多2000次，投影梯度门槛1e-5、目标变化门槛1e-9；独立检查1%最优解距离证书||g_min||/alpha。精确零可抵消正梯度，小正值不能当零。证书只证明接近当前正则化最优解，不证明接近真值。

budget_seconds=600，在完整任务之间检查；未收敛、预算截断或证书失败非零退出。显示/落盘失败留下failed，完成标记最后写。冻结来源包括F1及此前九个正式run，指纹前后核验。

## 实际结果

点源试运行：results/stage02f2/run_20261004_095231。
正式12项：results/stage02f2/run_20261004_095323，全部收敛，数值与1%证书通过。

CPU耗时35.515秒；环境Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9，无新增依赖。

|器件|场景|方法|迭代|cube相对L2|距离相对上界|
|---|---|---|---:|---:|---:|
|continuous|coincident_points|unweighted|308|0.944230|0.000261464|
|continuous|coincident_points|fixed_weighted|297|0.961938|0.000642869|
|continuous|separated_points|unweighted|313|0.854533|0.000387068|
|continuous|separated_points|fixed_weighted|348|0.929343|0.000432092|
|continuous|lines_and_square|unweighted|288|0.644844|0.00222415|
|continuous|lines_and_square|fixed_weighted|422|0.621212|0.00102372|
|nearest_depth|coincident_points|unweighted|318|0.941518|0.000322676|
|nearest_depth|coincident_points|fixed_weighted|301|0.960347|0.000385552|
|nearest_depth|separated_points|unweighted|309|0.851465|0.000505576|
|nearest_depth|separated_points|fixed_weighted|336|0.928371|0.000427948|
|nearest_depth|lines_and_square|unweighted|390|0.643287|0.000218948|
|nearest_depth|lines_and_square|fixed_weighted|338|0.619002|0.000319203|

恢复误差约0.619至0.962，质量仍差，有明显扩散与幅值偏差。加权点源更差，线段/方块稍好，没有普遍优势。可能涉及正则化偏置、编码病态性和噪声，尚未分离；未根据真值选alpha，也不把低残差当成准确恢复。下一阶段冻结本次alpha/W，用同目标无噪声理想控制诊断误差，再扩增益/种子。理想控制的模拟期望是特权信息，不能用于实际重建或调参。

## 输出与审核证据

arrays/*.npz保存Z、背景、W/方差估计、电子响应/G/QE/lambda、核、坐标、恢复、预测、残差、alpha/L，真值明确叫evaluation_truth。metrics保存轨迹JSON/CSV和评价/证书。figures用原始相对功率，共用色标，不作峰值归一化；本批恢复峰值低于真值，同场景方法间色标一致。

独立审核用逐点前向与空间相关伴随，不调用生产重建模块，复核12项目标/梯度/证书、负观测、权重和轨迹CSV。完整回归232项通过，203.320秒。错误来源退出1，预算/未收敛退出2，篡改计数副本被拒绝退出1；mock显示仅一次且关闭全部图，显示和报告故障没有完成标记。mock不是实际GUI验证。

无保存全目录SHA与最终指纹审核正在单独执行，最终结果完成后追加。外层证据目录D:/PyCharmProjects/Jeon2019/outputs/执行_阶段02F2_20261004。论文旋转方向对应、真实GUI/PyCharm点击、真实相机标定、论文网络仍未验证。


## 最终交付验收

最终审核通过：正式12项、232项回归、独立空间域审核、坏来源/计数篡改/预算/未收敛/显示和报告故障、异地无保存全部验收。无保存执行前后12381个项目文件的集合及SHA不变；十个正式来源run及冻结源码指纹最终复核相同。最大最优解距离相对证书0.002224152607（约0.2224%），但真值误差仍约0.619至0.962，不能把两者混同。

最终记录：`D:/PyCharmProjects/Jeon2019/outputs/执行_阶段02F2_20261004/最终审核.json`。下一阶段规划：同目录`下一阶段02F3_1详细规划.md`；将冻结W/alpha，用理想测量控制诊断基线误差和单次噪声扰动。该阶段尚未运行。全部故障目录与试运行保留；本轮没有删除文件或清空已有文档。
