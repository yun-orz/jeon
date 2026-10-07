# Jeon2019正式仿真基线 v1

此目录是新流程，历史G0–G5保持原位。当前实施M0：统一复现协议、来源、参数证据、保护清单和八份Goal任务。GPU安装、正式算子、数据、训练器、训练及最终评价需逐里程碑验收；M0通过不等于整个论文复现完成。

入口：

```powershell
& D:/dev/python/python3.10.4/python.exe -B D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/main.py
& D:/dev/python/python3.10.4/python.exe -B D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/main.py report
```

无参数默认check，路径依据入口位置解析，可从其他工作目录启动。check只读核对全部历史保护SHA、G0、协议/来源/任务及M0交接对象SHA，同时报告当前解释器和D盘空间。它不安装依赖，不证明GPU可用或大数据空间充足。尚未实现的optics/data/smoke/train/evaluate明确返回非零，不生成成功占位物。

首次M0冻结：执行本目录build_m0.py；已有m0_index.json时拒绝覆盖。历史文件保护清单和中文验收位于新结果目录，位置由m0_index.json给出。所有失败运行保留。保护检查全量读取大型文件，耗时取决于磁盘，不加载完整数组。

交接文件：

- protocol.json：锁定默认值、路径、训练与评价条件，以及仍需训练数据拟合的增益。
- parameter_evidence.json：逐项登记论文确认、实现验证、实施假设和未知，保留计数/SAM/网络/标定歧义。
- sources.json：本地来源SHA和有限网络检索登记。网页读取不是本地下载，未冻结内容SHA保留null。
- contracts.json：算子、数据、恢复checkpoint及里程碑报告的必需字段。
- tasks：M0–M7八份Goal任务、前置条件、可修改范围、验证、失败与停止规则。
- m0_index.json：仅验收通过后创建，绑定上述交接文件和M0证据的SHA。

所有新数据使用work/datasets/reproduction_v1，新结果使用outputs/jeon2019_optics/results/reproduction_v1。根CURRENT_STATUS.md只追加；任何删除须先取得用户明确同意。paper_alignment_passed=false，文献35.88 dB/.93/.12不是工程验收线。
