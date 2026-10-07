# Jeon2019：从这里开始

更新日期：2026-10-07，北京时间。当前工作停在 **M3 数据准备完成**；下一阶段是 M4 可恢复训练器，尚未启动。

这是《Compact Snapshot Hyperspectral Imaging with Diffracted Rotation》的复现项目。我们的目标是先完成当前 M0–M7 协议下的正式仿真、Jeon/Fresnel 对照与测试评价，再结合最终结果回看模块原理和代码。作者精确数据名单、部分网络细节与标定信息仍未知，完成这条路线不自动表示论文数值完全一致。

## 平时只需要从这几个入口开始

| 你想做什么 | 打开哪里 |
| --- | --- |
| 看目录、模块和数据流 | [项目地图与复现路线](D:/PyCharmProjects/Jeon2019/docs/PROJECT_GUIDE.md) |
| 看当前正式代码 | [正式流程目录](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1)；[统一命令入口](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/main.py) |
| 看 M3 已交付什么 | [M3 正式交付报告](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/delivery_20261007_160046/M3_delivery.md) |
| 看下一阶段要完成什么 | [M4 可恢复训练器任务](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/tasks/M4_resumable_training.md) |
| 看当前协议与明确的假设 | [冻结协议](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/protocol.json)、[参数证据](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/parameter_evidence.json) |

## 当前到了哪里

下面的完成状态来自既有正式验收记录。本轮整理只核对文件与指纹，不重跑 GPU、光学、数据处理或训练。

| 阶段 | 当前状态与正式依据 |
| --- | --- |
| M0：冻结协议 | 通过；[正式报告](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m0/run_20261006_230835_592a12d3/report.md)，交接路径由 [M0 索引](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/m0_index.json)绑定 |
| M1：GPU 环境与一致性 | 首次验收后又完成修复与 GPU 重验；以 [最新修复报告](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/M1_fix_report.md)及其 [回执](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/fix_receipt.json)为准 |
| M2：物理光学与 RGB 算子 | 通过；正式父包在 [run_20261007_104321_03d47940](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m2/run_20261007_104321_03d47940/report.md)；此阶段共同增益尚未拟合 |
| M3：数据与最终算子 | 通过；正式运行是 [run_20261007_154644_d46e6171](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/report.md)；最终算子已绑定数据版本与训练集共同增益 |
| M4–M7 | 尚未启动；训练器 → 预演诊断 → 两器件正式训练 → 封存测试评价 |

M3 已冻结 **238 幅场景：203 幅训练、25 幅验证、10 幅 KAIST 封存测试**，以及 **30,000 条 256×256 训练补丁索引**。索引按需读取数据，没有复制 30,000 个补丁文件。测试图像预处理与评价留到 M7。

日常看结果时，使用上述正式链接。新生成的检查目录、失败运行和时间更晚的目录，并不自动成为新的正式数据版本。

## 当前配置与正式交接

- 物理配置：[config_m2.json](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/config_m2.json)。数据配置：[config_m3_masked_index_20261007.json](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/config_m3_masked_index_20261007.json)。原始 `config_m3.json` 是旧阶段配置，当前数据版本使用 masked_v3 配置。
- 数据交接：[场景清单](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/scene_manifest.json)、[数据版本](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/data_version.json)、[训练索引](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/patch_index.jsonl)、[测试封条](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/test_seal.json)。
- 训练使用 M3 的 [最终 Jeon 算子](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/operator_jeon/package.json)和 [最终 Fresnel 算子](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/operator_fresnel/package.json)，共同增益为 `0.003682718635788138`。保留 M2 父包作为来源，不把尚未拟合增益的 M2 包直接当作最终训练包。

## 运行前先确认

CPU 解释器是 `D:/dev/python/python3.10.4/python.exe`；GPU 解释器是 `D:/PyCharmProjects/Jeon2019/work/environments/jeon_gpu_cu121/Scripts/python.exe`。后续 GPU 训练使用后者，并保持 NVIDIA 独显启用；已有验收通过记录不保证每次启动时硬件都可用。

ICVL 原始数据位于 `H:/我的云端硬盘/Jeon2019_data/reproduction_v1/icvl/mat`。本轮在当前执行权限下访问该路径被拒绝，未读取原始图像，不能据此判断数据丢失。后续运行需确认 H 盘已挂载、账号可访问且文件可读；D 盘仍承载项目内的预处理数据和结果。

`main.py` 无参数等同于 `check`，检查后会写一个新的结果记录。`report` 也会核对现有 M2/M3 来源，并非只显示文字。`optics` 和 `data` 会重跑相应工作，查看现有结果无需执行它们。`smoke`、`train`、`evaluate` 目前只是预留命令，实际执行会报未实现；具体命令见项目地图。

## 旧说明怎样阅读

根 [README](D:/PyCharmProjects/Jeon2019/README.md)描述 G0 快照；[旧光学工程 README](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/README.md)记录早期阶段；[正式流程 README](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/README.md)仍描述 M0 启动时状态。它们保留历史用途，当前导航从本文件开始；[CURRENT_STATUS](D:/PyCharmProjects/Jeon2019/CURRENT_STATUS.md)保留按时间追加的研究记录。

以后每个里程碑验收后，更新本文件和项目地图中的当前状态、正式交付链接与验收日期，并在状态记录末尾追加说明。保留旧报告原文，以便追溯结论变化。
