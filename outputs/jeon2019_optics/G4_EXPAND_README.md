# G4训练规模扩充：原权重热启动，保留固定物理模型

本步仅使用img3训练、img4验证，不读取img5或其测试数组，不评价独立测试成绩。G4原训练仅12步，这里扩大为16个训练上下文、8个验证上下文、8个epoch，共128次更新。原4个训练块和2个验证块放在各自列表前部，新增块不重复；同一验证ROI可作前后比较。

## PyCharm与运行参数

打开本地Python工程，解释器选Python3.10.4，安装requirements_g3.txt中的依赖，直接右键运行main_g4_expand.py，无参数CPU训练。已有本地数据、旧G4/G5正式run及审核记录后不需要联网、Harness、聊天、Docker或Notebook。相对路径均按入口所在工程解析，不依赖启动目录。

集中配置config_g4_expand.json：dataset.train_patches=16、validation_patches=8；core_size=32和halo=48保持原域；seed=20261006决定新增块和训练顺序；training.epochs=8、learning_rate=.001、gradient_clip_norm=1。half_lr_after_epochs=4为每完成4个epoch后学习率减半的预定工程设置，**不是原论文完整训练日程**。max_seconds=600是训练循环CPU时间预算，超出会报错并保留失败run，不能把它算完成；前处理与保存另计。

cpu_threads=4，data_range=1、zero_norm_threshold=1e−12保持G5已审核定义。runtime控制保存/显示；plots.dpi控制图像分辨率。训练/验证块数量及epochs设有CPU规模上限，避免误填导致无界运行。

命令行`python -B main_g4_expand.py`；`--no-save`不保存但仍训练完整配置，因此耗时相当。`--show-plots`在最后显示图像（本轮GUI人工操作未验证）。若仅核对已经保存的模型，使用：

```text
python -B main_g4_expand.py --evaluate-only --checkpoint results/g4_expand/run_具体编号/arrays/best_checkpoint.pt --no-save
```

这两个参数必须一起给出。只读评价不训练、不改变原checkpoint，对原模型与新模型作同域推理比较；无保存审核实际采用该模式，**没有再次运行完整128步训练**。评价模式返回optimizer_steps=0、epochs_completed=0和模式说明。checkpoint中的数据元信息必须与当前配置所选块一致，不能悄悄换数据后称重现。

## 固定数据尺度与成像模型

沿用原G4训练块拟合的一个尺度0.019641629936370968，不用扩充块重新拟合，也不对验证图或波段单独归一化。目标=(ref/calib)×λ/540/原尺度；这是已声明的相对能量谱解释到相对光子表示换算，不是绝对辐射标定。原25个波段420:10:660nm、97×97 G1核、G2合成RGB响应和强度叠加保持不变；测量由float64核编码后转float32交给模型。没有新DOE、双孔径、噪声注入、CFA或gamma测量。

128×128上下文中只监督中心32×32；halo48对一次前向对应核支持，不能保证整个HQS/U-net无有限域边界影响。只有一幅训练场景和一幅验证场景，扩块与更多更新不等于场景多样性已足够。

## 训练、选择与比较

从G4最佳epoch2的网络权重开始。原G4没有保存Adam状态，本步创建**新的Adam**，所以是权重热启动，不能称优化器无缝续训；新checkpoint也不保存优化器状态，当前不提供恢复训练接口。

损失为中央光谱cube的L1；每epoch评估全部训练/验证块。最佳模型只由8个验证块L1选择，原G4模型是epoch0候选：若新训练都更差，就保留原模型，明确记录best_epoch=0。保存每epoch所有损失、学习率、训练顺序、最大梯度范数，不隐藏反弹；PSNR/SSIM/SAM作为诊断，不用于另选模型。模型的Φ、Φᵀ对应核与相机响应不参与学习。

报告有四组指标：扩充验证8块的训练前/后，以及原验证2块的训练前/后。各组同一目标、原尺度与指标定义，预测不裁剪。PSNR按合并MSE，SSIM高斯11窗口valid域，SAM为有效谱的平均rad，详细定义见G5_README.md。验证参与模型选择，不能称独立测试泛化成绩；不将本轮更新后的模型在已查看过的img5上反复调整后宣称盲测。

## 输出及实际验证范围

每次建立全新results/g4_expand/run_*，保留所有旧文件和失败记录。metrics/training.json保存全部训练历史，scores.json逐块指标，validation.json保存模式/范围与汇总；dataset_manifest记录原/新增坐标、尺度与校正SHA；arrays/expanded_subset.npz保存目标、测量、训练前后预测；best_checkpoint.pt保存最终选中模型，并实际重载逐元素核对预测；figures保存曲线与同范围重建图；source_manifest/source_evidence保存原来源完整SHA。

环境沿用CPython3.10.4、NumPy2.2.6、SciPy1.15.3、Matplotlib3.10.9、PyTorch2.14.1+cpu。新增3项数据选择检查，以及独立MAT换算/测量/指标、模型只读评价和来源完整性审核；最终结论以outputs/执行_G4扩充_20261006/最终审核.json及正式run为准。当前仍不是完整原论文网络、训练或性能基线，paper_alignment_passed=false；暂不进入G6。
