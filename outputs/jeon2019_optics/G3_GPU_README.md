# GPU环境与HQS执行一致性验证

本阶段是Jeon光学编码复现＋基础重建验证的运行环境验收，不是正式训练或论文基线通过。保留G4扩充epoch8及原CCW光学核/合成RGB响应；不加载img5，不选择新模型，不改变已有checkpoint。

## PyCharm手动运行

打开普通项目，在设置→项目→Python解释器→添加现有解释器，选择D:/PyCharmProjects/Jeon2019/work/environments/jeon_gpu_cu121/Scripts/python.exe。此虚拟环境在原CPython3.10.4基础上创建，继承已有科学计算包，新Torch/SymPy只安装在虚拟环境内；它不是完全独立复制所有依赖的环境，原科学计算包必须保留。不要给原CPU解释器安装本GPU清单。原CPU入口仍可选D:/dev/python/python3.10.4/python.exe。

运行main_g3_gpu.py，无需参数。config_g3_gpu.json集中设置来源、CPU线程数、门槛、一次检查的学习率、保存/显示。路径按入口位置解析，异启动目录也可运行。CUDA不可用则明确报错，不会将CPU执行报告为GPU。默认自动保存results/g3_gpu/run_*；--no-save只读复算，--show-plots请求弹窗。不依赖Harness、Agent、Notebook、Docker或云端。

GPU环境采用Torch2.5.1+cu121、SymPy1.13.1，科学计算版本NumPy2.2.6/SciPy1.15.3/Matplotlib3.10.9。官方版本来源：https://docs.pytorch.org/get-started/previous-versions/ 。CUDA12.1的Windows最低兼容驱动527.41，当前实测537.53；出处：https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html 。实际CUDA调用、设备与版本必须以正式结果为准，安装成功本身不是执行验证。

本次官方轮子由work/download_gpu_cu121.ps1下载，保存于work/dependencies/gpu_cu121_20261006，校验官方SHA256。虚拟环境安装时使用--ignore-installed --no-deps --no-cache-dir --no-compile，避免替换原环境；Torch之外的兼容依赖继承原环境。首次网络/TLS失败不等于GPU不支持。本项目无需额外安装完整CUDA Toolkit来运行预编译Torch轮子。

## 验证内容及物理含义

GPU双精度光学算子与冻结NumPy算子比较前向、伴随及输入梯度；带符号随机测试避免仅对正图像凑巧匹配。Φ是各波段PSF线性卷积及RGB响应求和，Φᵀ是其数学伴随，不是求逆。自动微分的½||Φx−y||²梯度应等于Φᵀ(Φx−y)。卷积采用充分零填充后裁中心，不是周期成像边界。

对原img4八个验证块，用新环境CPU和GPU加载同一冻结epoch8，比较完整128上下文预测及第一块每个HQS阶段；同时与原CPU环境已保存的八块预测比较，区分版本迁移差异和设备差异。三个对照都使用原测量/尺度，不重新生成实验目标。

只用原img3训练第一块做CPU/GPU各一次全128前向反向，以中央32、halo48计算L1损失，新Adam学习率1e−4、梯度裁剪1；比较损失、全参数原始梯度、一步更新参数并检查实际变化。旧checkpoint未保存优化器，故这是新优化器的执行检查，不是优化器续训。检查后的参数不作为候选模型保存或用于正式评价。

全部模型FP32，光学验算FP64，TF32/AMP关闭，确定性算法开启。cuFFT中卷积尺寸不总为2的幂，不能未经验证切换到半精度。双精度门槛1e−10，预测/阶段/损失相对1e−5、全梯度相对1e−4、更新参数相对1e−5预先配置。失败保留结果，不放宽阈值迎合结果。

## 输出与验收边界

arrays/gpu_checks.npz保存独立光学验算、八块测量与三组预测、第一块CPU/GPU阶段轨迹；metrics/validation.json保存一致性、训练通路、实际Torch位置/CUDA/设备、峰值显存与耗时；figures保存同尺度CPU/GPU和差图。source_manifest/evidence保留来源指纹，保存重载逐元素检查。

实际运行、独立审核、原CPU依赖和G0不变核验见工程外outputs/执行_GPU验证_20261006。没有最终审核文件前，不算验收完成。人工PyCharm点击、GUI、全数据训练、AMP、完整论文网络精确对齐及正式性能基线未验证，paper_alignment_passed=false。现有数据规模、合成响应和光学尺寸差异不会因使用GPU自动解决；当前不进入G6。
