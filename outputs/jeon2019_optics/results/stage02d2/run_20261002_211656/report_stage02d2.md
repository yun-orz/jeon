# 02D-2基础重建实际运行报告

Jeon光学编码复现＋基础重建验证；非原论文网络复现。
方法：零初始化、非负岭投影梯度与回溯；F=0.5||AX−y||²+α||X||²/2。

|实验|状态|轮数|数据残差相对值|cube相对L2|SAM/deg|
|---|---|---:|---:|---:|---:|
|continuous_coincident_points_crop|iteration_limit|500|0.12467|0.94948|4.7044671622992595|
|continuous_separated_points_crop|iteration_limit|500|0.26009|0.88192|9.755586687653304|
|continuous_lines_and_square_crop|iteration_limit|500|0.037305|0.67578|24.598468043746053|
|continuous_lines_and_square_full|iteration_limit|500|0.037312|0.67572|24.594371879835954|
|nearest_depth_coincident_points_crop|iteration_limit|500|0.12467|0.94955|4.93705685251245|
|nearest_depth_separated_points_crop|iteration_limit|500|0.25963|0.88199|9.928016298944781|
|nearest_depth_lines_and_square_crop|iteration_limit|500|0.037168|0.67557|24.434687609549474|
|nearest_depth_lines_and_square_full|iteration_limit|500|0.037175|0.67552|24.430774928027397|

## 限制

- 无噪声、三单色诊断、理想像平面平移不变模型，单位辐射响应。
- α由统一相对参数乘算子谱范数估计；幂迭代不是严格上界，回溯检查步长。
- iteration_limit表示达到迭代上限，不能声称约束最优解已收敛。
- 真值仅用于来源验证和评价，不用于初始化/迭代或每场景参数选择。
- 小数据残差不能代替光谱恢复误差；有限场景支持是先验。
- 原论文旋转方向对应、真实PyCharm点击和弹窗仍未确定/人工验证。

逐轮目标/梯度/步长及评价见metrics；原始恢复与测量在arrays；所有图为相对功率。
