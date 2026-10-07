# G4：真实HSI两场景小规模训练验证

本阶段已经实际跑通真实数据预处理、光学编码、训练、验证集选模型、保存重载与独立审核。准确范围为“Jeon 光学编码复现＋基础重建验证：实现假设、合成RGB响应、Harvard小规模训练”。不是完整Jeon训练或论文性能基线。

## 在Windows / PyCharm运行

打开outputs/jeon2019_optics普通Python工程，选择已有解释器。实际验证：Python 3.10.4、NumPy 2.2.6、SciPy 1.15.3、Matplotlib 3.10.9、PyTorch 2.14.1+cpu，Windows CPU，4线程。G4没有新增第三方依赖，沿用requirements_g3.txt。

```powershell
python -m pip install -r requirements_g3.txt
python main_g4.py
python main_g4.py --no-save
python main_g4.py --show-plots
python -m unittest discover -s tests -p test_g4.py -v
```

也可以在PyCharm打开main_g4.py点击运行，无需参数。config_g4.json集中参数，相对路径按入口__file__定位；已从项目根这个工程外部启动目录验证。程序默认保存、不弹窗，可手动打开PNG，或设置runtime.show_plots=true。真实PyCharm点击与桌面GUI未人工验证；已有环境实际运行，没有重新建立全新虚拟环境验证安装流程。

原始数据已保存到项目根work/datasets/harvard_g4_20261006，主入口离线读取，不自动下载，不依赖聊天/Harness/云端/Notebook/Docker/GPU。复制项目时保留原始数据、校正文件、G1/G2/G3来源run及审核记录的相对路径和内容。

每次运行创建独立run，拒绝覆盖。正式默认训练/保存计时约18秒，来源检查和MAT读取另计；未测峰值内存，其他电脑可能更慢。MAT加载会读取完整场景，不是零内存流式读取；目前仅两幅图像、4个训练上下文块、2个验证上下文块，不适合直接推断大规模训练资源需求。

## 数据来源与下载记录

Harvard官方项目：https://vision.seas.harvard.edu/hyperspec/ 。使用条件：https://vision.seas.harvard.edu/hyperspec/download.html 。本轮仅作本地非商业学术研究，不发布原始数据。

数据引用：Ayan Chakrabarti and Todd Zickler, “Statistics of Real-World Hyperspectral Images”, CVPR 2011。官方论文：https://vision.seas.harvard.edu/hyperspec/CZ_hss.pdf 。官方归档中的README说明：ref为N×M×31，波长420:10:720nm；lbl=0的像元无效；calib.txt为31波段相机相对灵敏度。

本轮从官方人工/混合照明归档CZ_hsdbi.tgz流式保留img3.mat、img4.mat、README.txt后关闭连接，没有保存整个2.2GB归档。共读取168,704,000压缩字节。初次筛选漏存归档开头的calib.txt，随后只补读10,240压缩字节取得497字节校正文件，没有重复下载图像。下载脚本属于一次性取数记录，非训练依赖；非空目标会拒绝重复执行，无删除或清空行为。

- img3.mat：95,094,913字节，SHA256 d1d59fb149bbf127a354ef8893d801eeb06f21266c7d8af8b34b213609a863b6。
- img4.mat：77,636,260字节，SHA256 4f9e10f7873c6d1c06fb66a24edb05b052392e92d3190cc2f7254a0dcd107f3d。
- calib.txt：SHA256 e8b0945a2e1f42fbc0219570b2d1f67112cacaded91a7b13859f22c78231801b。

download_manifest.json与calibration_download_manifest.json保存URL、归档成员名、读取字节及指纹，入口会核验。两幅完整预览已实际查看，img3为室内桌椅场景，img4为自行车场景；训练/验证文件与内容指纹不同。只有这两个场景，不代表Harvard完整数据集或Jeon的238幅训练集。

## 光谱预处理与单位

数据原论文PDF第2页Section 3明确说明原始图像没有按相机灵敏度归一化。程序取ref的前25波段，匹配420:10:660nm；不插值，不为各波长改变DOE，不逐波段峰值归一化。

采用下列显式约定：

```
energy_proxy[b] = ref[b] / calibration[b]
relative_photons[b] = energy_proxy[b] * wavelength_nm[b] / 540
target[b] = relative_photons[b] / train_scale
train_scale = 训练上下文块全部波段数值的99百分位
```

如果把校正后的相对HSI数值解释为能量谱，光子数与能量的关系N=Eλ/(hc)给出λ因子；用540nm作共同参考，把未知绝对常数并入训练尺度。官方文件没有完整绝对单位、曝光与标定链，故这是声明清楚的相对光子表示约定，不是绝对辐射或光子标定。未来获取更完整的灵敏度定义时，应重新核查能量/光子响应的约定，避免重复进行λ转换。

Harvard的calib.txt校正的是数据来源的高光谱相机，**不是Canon RGB响应**。G2/G3的真实RGB响应缺失状态仍保留；本轮RGB观测使用同一明确标注的合成QE。输出单位为relative_linear_response，不称为实际相机绝对电子计数。

尺度只由训练块拟合，验证图像使用同一尺度；没有用验证图像拟合尺度，也没有把验证标签用于梯度更新。观察图的共同颜色显示上限可以参考展示目标，它只控制绘图，不改变训练/验证数组或损失。

## 场景划分、掩码与边界

img3只用于训练，img4只用于验证，拒绝相同文件名或相同内容指纹。每幅图按128×128非重叠网格选择上下文块，任何lbl=0、非有限或负数像元都会使整个上下文块被排除，不填补运动/无效区域。随机种子固定20261006，具体坐标及候选数量保存到dataset_manifest.json。

默认上下文128×128，中心监督区域32×32，四周48像元halo；它覆盖97×97 PSF单次前向卷积所需范围。对完整上下文重新编码，再对网络输出中心区域计算L1，而不是把全图RGB裁剪当作零延拓小域的配对数据。

48像元halo不保证多阶段ΦᵀΦ和U-net都完全不受上下文边界影响。当前仍是有限上下文实验，不能称为完整场景的无边界误差重建。HSI空间像元被当作模拟投影像元，没有为数据集场景标定真实像面倍率、离轴PSF或物距。

## 训练设置与实际结果

沿用G3架构实施版：3阶段、64初始特征、四分辨率U-net、软阈值，各阶段独立参数；未公开细节和PyTorch/TensorFlow差异见G3_README.md。模型从固定种子的随机初始化开始，没有把G3两步合成验算权重当作正式预训练。

固定25波段光学核与合成响应均为模型buffer，不参与优化；器件指纹仍为8527724f6aeffc0314d09313d89fdca0c2c864827e6a47308480968bcabc1350。未切换CW诊断高度，也没有双孔径。

默认batch=1、4个训练块、2个验证块、3个epoch、12次Adam更新，学习率1e−3，梯度范数上限1.0，中心25波段L1损失。梯度裁剪是本轮工程设置，不是声称作者同样采用；没有原文完整增强、噪声、学习率日程和训练数据规模。

| epoch | 训练中心L1 | 验证中心L1 |
| --- | --- | --- |
| 随机初始化0 | 0.272691 | 0.145124 |
| 1 | 0.257409 | 0.136486 |
| 2 | 0.186575 | 0.099954 |
| 3 | 0.230629 | 0.134364 |

最佳模型按验证L1选取epoch 2，并实际重载核对；不是默认保存最后epoch 3。损失回升已保留，不据此宣称收敛。验证集参与模型选择，且只有一个验证场景、没有独立测试集；上述下降不是泛化性能结论。

这是训练尺度归一化后的相对光子目标中心L1，不是Jeon论文PSNR/SSIM/SAM，也不直接对应辐射谱或反射率误差。输出仍较平滑，未声称正确恢复全部细节或光谱。进一步训练与论文指标对齐应先解决数据/标定/协议差异。

## 输出与核验

正式目录：results/g4/run_20261006_074407。results/g4/run_20261006_075127是用同一真实数据、较小模型模拟保存失败的审核证据，无completed标记，保留且不能当正式训练输出。

- arrays/g4_subset.npz：训练/验证目标与测量、初始化/最佳模型验证预测、波长、尺度、灵敏度、器件指纹和单位。
- arrays/best_checkpoint.pt：验证集选择的模型state_dict、架构配置、最佳epoch、数据协议与明确的合成响应状态。不是官方权重；未保存Adam状态或提供断点续训。
- dataset_manifest.json：官方来源、原图指纹、校正向量、场景划分、mask有效性、裁块坐标与训练尺度。
- metrics/training.json：逐epoch损失、训练顺序、梯度范数、模型选择与更新数。
- figures/validation_real_hsi.png、training_validation_curve.png：验证目标/输出对照和实际训练曲线，已查看。
- config_effective.json、source_manifest.json、source_evidence.json、report_g4.md、metrics/validation.json：配置、来源及状态。

4项新增预处理测试通过。独立审核重新读取原始MAT与lbl，校正和训练尺度与保存目标完全一致；重新编码的RGB逐元素一致；最佳模型与随机初始化均可重现，独立重算验证L1为0.0999539122。固定光学核和响应未被学习或改变。

完整--no-save从工程外部目录运行，14415个工程文件前后SHA不变；原始数据与G3来源也逐文件保持不变。重复训练/验证场景在创建run前拒绝；模拟torch.save失败保留failed.json，无完成标记。G0全部767个冻结文件通过，未删除旧文件。

审核证据位于项目根outputs/执行_G4_20261006。本说明与状态索引在无保存核验后新增/更新，不在当时14415文件快照中。GPU、人工GUI、大规模训练、完整场景测试与G5论文指标对齐均未验证。
