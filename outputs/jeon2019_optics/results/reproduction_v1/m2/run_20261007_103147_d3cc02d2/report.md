# M2 光学与RGB算子审核

状态：failed。已完成0/50器件波段，耗时0.1秒。

实际执行命令：

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\main.py" "optics" "--config" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\config_m2.json"
```

共同训练增益：pending，未拟合；M3须创建带父指纹的新最终包。原生97×97/6.22μm，半尺寸49×49/12.44μm，保留有限窗口质量与几何原点。

参数、Sellmeier公式及系数见 effective_config.json；原始响应、许可证、访问内容SHA见 response_metadata.json；全部输入源码SHA见 source_fingerprints.json。

逐波段四组比较保持G1原定义和原阈值；parseval只检查数值功率，不证明无限域收敛。Fresnel无可靠旋转结构时按G1状态比较，不虚构角度。独立面积映射、点质量、对称核、质心和NumPy/Torch FP64核对见各方法审核文件。

失败与未完成情况见 audit.json；禁止把本次状态解释为论文对齐。下一阶段M3，仅物理包审核通过后推进。

失败诊断：先查看每波段input_q4/input_q8与quadrature比较定位输入或探测器误差，不放宽阈值。续接命令：

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\main.py" "optics" "--config" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\config_m2.json" "--resume" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\results\reproduction_v1\m2\run_20261007_103147_d3cc02d2"
```
