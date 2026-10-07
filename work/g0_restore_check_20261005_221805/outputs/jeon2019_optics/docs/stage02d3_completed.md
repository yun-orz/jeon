# 02D-3完成记录：相同目标的加速非负重建

范围：Jeon光学编码复现＋基础重建验证。本阶段由Codex在本地Windows/CPU实际实现、执行、修正及审核；不是原论文网络复现，不是双孔径实验。

## 为什么做这一步

02D-2的500轮投影梯度八组均达到迭代上限，不能区分算法尚未解完和重建模型本身的误差。本阶段保持相同A、y、实际α、非负约束、零初始化、停止门槛，采用带回溯和单调重启的加速投影梯度。固定器件与测量，没有通过更换DOE或调整每场景α改善结果。

光学源为02C run_20261002_172338，成像源为02D-1 run_20261002_203904，PG源为02D-2 run_20261002_210155；正式输出为results/stage02d3/run_20261002_222854。三层来源文件和源码指纹均被复核，运行前后SHA未变。

## 物理与数学解释

传播程序计算复振幅，相机测量是各单色分量强度/功率的非相干叠加。本阶段使用之前已计算的固定高度DOE像元积分PSF，不再传播或按波长设计器件。三诊断波长420/540/660 nm映射到相同6.22 μm探测器像元；核保留有限窗口通光量，不将每个核归一到1。单位辐射响应、无噪声、平移不变理想像平面场景及有限32×32支持仍是模型假设。

重建目标F(X)=0.5||AX−y||²+α||X||²/2，X≥0，梯度g=A*(AX−y)+αX。A把各波段卷积结果按强度相加，A*是已经验证的离散伴随。这里的岭惩罚是基础重建先验，不是论文原网络。

加速器根据前几步的变化作外推，再做非负投影；外推变量允许为负，接受的物体功率必须非负。回溯验证二次上界，避免数值估计步长过大。候选目标上升时重置外推，从上一接受点重新计算候选；若仍不合格则失败，不能静默继续。

投影梯度映射m=L[X−max(0,X−g/L)]，以||A*y||归一（零数据使用数值下限）。归一化映射≤1e−5且相邻目标相对变化≤1e−7才标为converged。max_iterations=5000，八组均在577–769轮满足原门槛。这个数值判据不等于恢复真值，也不是相对于精确最优解的严格误差证书。

实际α逐实验从PG基线读取，不重新估计：continuous crop约1.0589016188494556e−4、full约1.0589754246732999e−4；nearest_depth crop约1.0032299973098796e−4、full约1.0033001813460194e−4。full/crop的α略有不同，不能把比较解释成纯裁剪因素实验。

## 正式运行结果

|实验|轮数|状态|数据相对残差|cube相对L2误差|SAM均值/deg|
|---|---:|---|---:|---:|---:|
|continuous_coincident_points_crop|581|converged|0.0472553|0.864866|3.21548|
|continuous_separated_points_crop|679|converged|0.0891312|0.625498|3.90911|
|continuous_lines_and_square_crop|765|converged|0.00988371|0.369597|7.98131|
|continuous_lines_and_square_full|765|converged|0.00988369|0.369486|7.97962|
|nearest_depth_coincident_points_crop|577|converged|0.047249|0.865011|3.04637|
|nearest_depth_separated_points_crop|677|converged|0.0891038|0.625655|3.96727|
|nearest_depth_lines_and_square_crop|769|converged|0.00985469|0.370197|7.95535|
|nearest_depth_lines_and_square_full|769|converged|0.00985468|0.370087|7.95381|

全部八组目标不高于对应PG500基线，实际α保持一致。连续器件crop线条场景cube误差从PG500约67.6%下降到约37.0%，共点源从约94.9%下降到约86.5%。共点源SAM仅约3.2°，但功率幅值误差很大：谱角主要反映光谱方向，不能替代幅值和空间误差。

连续器件两个PG5000诊断仍未收敛，线条cube误差约43.9%、共点约89.2%；它们只是两个代表诊断，不是全部八组5000轮对照。加速轮数少不能直接当作同等倍数的运行速度提升，每轮成本不同。

## 实际验收

- 本地CPU正式运行39.743秒，退出0；completed、validation_passed、optimization_validation_passed、all_reconstructions_converged均为true。运行时长随负载改变。
- 最终完整unittest回归187项全部通过，131.005秒，无pytest依赖。新增10项涵盖独立NNLS参考解、回溯、重启、负外推/非负接受、零数据、上限状态、非法输入、来源篡改和完整报告JSON序列化。
- 独立审核不使用生产FFT前向/伴随：逐点卷积重算拟合、直接转置重算梯度映射，并核对目标、实际α、误差、状态和三层来源SHA；八组通过。
- 正式结果8份NPZ逐字段重读，8组优化JSON/CSV轮数核对；17张PNG、评价CSV、中文报告、来源证据和源码指纹全部存在。已实际查看连续器件线条crop恢复图，原始功率统一色标可读，仍有幅值损失和串扰。
- 从项目外启动目录实际执行默认--no-save，退出0、八组优化通过；随后模拟show分支，show调用一次且图窗全部关闭。前后项目全部文件SHA及集合一致。
- 缺失来源CLI非零退出并留下failed.json；真实max_iterations=1 CLI退出2、计算completed但优化验收false；报告写入故障保留failed.json而无最终完成标记。故障证据均保留。
- 首次加强报告时，NumPy布尔值不能直接JSON序列化导致保存失败；已转为Python bool，增加完整报告序列化测试并重新正式运行。失败run未删除，不当作正式交付。

实测Python3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9；解释器D:/dev/python/python3.10.4/python.exe。Windows CPU，无新增依赖。真实PyCharm点击与GUI弹窗尚未人工验证，mock显示测试不能替代这两项。

## 手动运行

在PyCharm打开本项目目录outputs/jeon2019_optics，配置本地解释器，首次新环境执行python -m pip install -r requirements.txt；右键main_stage02d3.py即可无参数运行。配置在config_stage02d3.json，所有相对路径基于入口文件目录，换电脑须完整保留三层来源run及相容源码。

```powershell
python -B main_stage02d3.py
python -B main_stage02d3.py --no-save
python -B main_stage02d3.py --show-plots
python -B -m unittest discover -s tests
```

配置：source_run指向PG只读源；solver控制最大轮数、两个停止门槛、回溯倍数与次数；evaluation控制SAM零光谱判定；check_seed/thresholds控制离散算子检查；runtime控制保存和显示；plots.dpi控制出图。02D-3比较要求停止门槛与PG源一致，修改这两门槛会被拒绝；更严门槛诊断应在下一阶段独立入口实现。

每次新建唯一results/stage02d3/run_...，不覆盖历史。arrays包含真值/恢复/拟合/残差/核/物理坐标，metrics包含轨迹与验收，figures包含恢复及拟合图。退出0要求优化验收通过；退出2表示计算完成但优化验收未通过；异常退出并保留failed.json。--no-save无文件结果，--show-plots请求手动查看，两个开关独立。

## 未完成与下一步

原论文旋转方向的坐标对应仍未解决，未作镜像修正；不是完整Figure 3或网络复现。宽波段积分、真实材料/镜头误差、相机响应/噪声、真实场景及双孔径互补性未验证。达到停止门槛仅说明此固定目标的数值优化通过，不证明编码充分或实际相机恢复准确。

下一阶段02E-1详细规划保存在项目外执行证据目录，预先固定α倍数，先验证收敛证书，再研究相同测量下的正则化偏差；本阶段尚未运行该参数扫描。
