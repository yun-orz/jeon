# G5补充：冻结模型的独立场景小规模测试

新增Harvard官方img5.mat（显示器和书本场景），不参与G4训练、验证或epoch选择。使用已选定的G4最佳epoch2模型，仅推理，0次优化更新。img3训练、img4验证，img5文件/内容与两者不同；同一数据库及可能共有的室内环境仍可能相关，这不是跨数据集泛化证据。

## PyCharm直接运行

打开本工程，在PyCharm选Python3.10.4解释器，依赖沿用requirements_g3.txt，直接右键main_g5_test.py运行，不需要参数。已有本地数据和旧正式run后完全离线，不依赖Harness或聊天。路径按入口文件位置解析；程序不会自行下载数据。

集中配置为config_g5_test.json：source_g4/source_audit锁定模型，source_g5/g5_audit锁定已审核指标版本；test_dataset.directory为本地img5目录，patches默认4、seed=20261006。在获得任何测试成绩之前已固定这两个参数，后续改块数或种子属于新增实验，不能挑分最高的运行冒充预定测试。CPU线程4，可调1至16；data_range=1及zero_norm_threshold=1e−12固定为已审核G5协议，改变时会拒绝运行。runtime控制保存/显示，plots.dpi控制图像分辨率。

命令行可用`python -B main_g5_test.py --no-save`或`--show-plots`；PyCharm参数框也可填对应开关。每次保存到新的results/g5_test/run_*，拒绝覆盖旧结果。无保存不写产物；未人工核验PyCharm界面点击与弹窗。

## 数据与物理量

原始1040×1392×31，取420:10:660nm前25波段；只选掩码非零、全部25波段有限非负的完整128×128上下文，网格非重叠，预定种子选择4块。只评价各块中央32×32，halo48用于97×97核的一次前向；多阶段网络仍可能受有限域边界影响。

目标 = (ref / Harvard相对灵敏度) × wavelength_nm/540 / G4训练尺度。尺度0.019641629936370968只由img3训练上下文拟合，**不在img5上拟合、逐图归一化或裁剪目标/预测**。绝对原始能量单位尚未完整标定，能量谱解释与相对光子换算沿用G4声明，不是绝对辐射标定。目标可能大于1，程序记录超参考幅度数量。

编码测量使用原G2 float64物理核及合成RGB响应，波段非相干强度线性叠加；测量再转float32交给冻结Torch模型。没有更换CW诊断变体、学习核、修改DOE、加入噪声或CFA。完整上下文重新编码，不从未知整幅测量裁切。

PSNR固定L=1、按合并MSE；SSIM逐波段11×11高斯valid窗口；SAM按有效像元平均rad。定义详见G5_README.md。原始谱表示、指标聚合和相机响应尚未与论文完全对齐，所以不把当前指标与论文35.88/.93/.12直接作公平差值比较。

## 成绩与范围

四块共4096中心像元来自**一幅**测试场景，不能称四幅独立图像。未训练比较模型由G4原种子重建，其在旧验证输入上的输出先逐元素核对为G4保存的原初始化结果，之后才用于img5比较。

| 模型 | PSNR / dB | SSIM | SAM / rad |
|---|---:|---:|---:|
| 原初始化 | 7.888232 | .566526 | .528601 |
| G4冻结epoch2 | 10.330627 | .645658 | .493732 |

不因成绩偏低重新选场景、尺度或模型。相对于初始化有改善；与此前img4验证不同，当前误差较大且不能由单场景证明普遍泛化。正式Jeon simulation baseline仍未通过：光学尺寸、真实RGB响应、网络细节、完整训练及原论文十图评价协议尚有差异。暂不进入G6。

## 输出、来源与验证

arrays/heldout_test.npz含完整上下文目标、测量、两种预测、波长与中央SAM/SSIM图；metrics/scores.json逐块、per_band.csv逐波段、validation.json汇总与范围；dataset_manifest记录选块坐标、训练尺度与校正SHA；source_manifest/source_evidence记录旧模型/指标/原始数据的哈希。figures保存重建及指标趋势：每列目标/预测共用完整数值显示范围，色条单位是训练尺度下相对光子值，未对图像单独增强后评价。

数据从[Harvard官方项目](https://vision.seas.harvard.edu/hyperspec/)取得，限非商业研究，学术使用引用Chakrabarti与Zickler 2011。下载脚本work/download_harvard_g5_test.py只执行过一条成功网络请求，读到img5即停止，失败目录保留。普通运行入口不调用该脚本；非空下载目录拒绝复用。MAT及校正来源记录在work/datasets/harvard_g5_test_20261006_retry1，场景预览与审核在outputs/执行_G5独立测试_20261006。

实际环境沿用已验证CPython3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9、PyTorch2.14.1+cpu。3项新增数据测试、6项G5指标测试；独立原始MAT换算/掩码/选块、SciPy卷积测量及局部SSIM审核；异启动目录无保存核验。实际CPU测试推理、文件保存和图像查看已执行；不是GPU或完整训练实验。
