# M1 缺陷修复与恢复验收

2026-10-07，北京时间。两个代码入口缺陷和设备诊断误导已修复；用户恢复 GPU 模式后，RTX 4060 实际 CUDA 自检及完整执行一致性重验通过。当前 GPU 环境可以用于后续阶段。

## 修复内容

- [m1_environment.py](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/m1_environment.py)：旧下载前缀不存在时完整返回 remaining_bytes=2449372784，检查流程继续执行；下载前缀与已安装环境分别记录。已安装 CUDA 版 Torch 但设备不可用时返回 gpu_device_unavailable，指向显卡连接／模式／驱动排查，停止误导性重装建议。
- [m1_download.py](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/m1_download.py)：缺失旧前缀时传 prefix=None，从零下载；已有前缀仍只读续传，整包官方 SHA 必须通过的门槛保持不变。
- [新增回归测试](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/tests/test_m1_missing_prefix.py)：覆盖缺失前缀／缺失工作目录从零获取、已安装环境检查、保留已有前缀续传、错误整包SHA拒绝、设备不可用不建议重装。5项通过；既有独立审核适配层10项测试通过。所有新fixture保留，没有删除操作。

## GPU 根因与恢复

系统排查时 RTX4060 为 Disconnected、nvlddmkm 未运行。用户确认切换为安静／仅核显模式，英伟达显卡未打开；用户恢复 GPU 模式后，Windows 将其列为 Started。当前 Torch 2.5.1+cu121 来自 GPU 环境自身目录，CUDA runtime 12.1，设备 RTX4060 Laptop。

本次硬件扫描曾因 Windows 管理员权限不足被拒绝；没有执行卸载、删除驱动或强制重启。恢复通过用户的 GPU 模式切换完成，未重装 Torch 或更改原 CPU 环境。后续 CUDA 训练须保持 NVIDIA 显卡可用；关闭独显的模式仍无法运行 GPU 运算。

## 实际验收

真实 CUDA 矩阵自检：FP64 相对误差 `7.6094414e-16`，FP32结果有限；环境检查 `gpu_ready=true`。

使用原 M1 独立审核适配层，将 PERSIST_DIR 重定向到本新目录，运行完整正式 GPU 入口的无保存重跑。所有冻结预测、HQS 阶段、全梯度、损失与一次 Adam 更新的汇总值，与 20261007_103806 历史 run 逐字段完全一致；未放宽任何阈值。前向／伴随／梯度的独立 SciPy误差均小于1e-10，独立式(21)三阶段最大误差约7.67e-08。

源码、来源模型树、原 CPU 依赖、G0及引擎运行前后完整性核对通过；未读取新测试预测、未保存新模型候选。所有新报告在本目录，未覆盖旧 M1 记录。

修复后实际执行 main.py check，M0 历史保护（15942文件）、G0（767项）、M2及M3全部通过。M3数据／算子冻结指纹保持原值。main.py 的 milestones.M1 仍为现有静态 dependency_pending 占位，本次真实 M1 验收以本目录 independent_audit.json / 最终审核.json 为依据，环境字段已实际报告 gpu_ready；未修改冻结入口 main.py，以保持已通过 M3 的源码指纹。

## 入口和证据

恢复下载入口仍为 `python -B outputs/jeon2019_optics/reproduction_v1/m1_download.py download`，现在在原前缀与工作目录缺失时可从零获取；已用64KiB本地HTTP样例实际验证，不重复下载2.3GiB或进行安装。轮子官方SHA冻结值仍为 `9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f`。

```powershell
python -B -X utf8 -m unittest discover -s outputs/jeon2019_optics/reproduction_v1/tests -p 'test_m1_missing_prefix.py' -v
python -B -X utf8 -m unittest discover -s outputs/jeon2019_optics/reproduction_v1/tests -p 'test_m1_audit.py' -v
python -B -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py check --m3-run outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171 --stage M1_entry_fixes_20261007
```

CUDA 自检通过导入原 m1_cuda_selfcheck 后设置 M1 为本目录执行；独立审核通过 importlib 导入原适配层后设置 PERSIST_DIR 为本目录执行；两者都未修改原判定规则。

- [真实CUDA自检](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/cuda_selfcheck.json)
- [完整独立GPU审核](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/independent_audit.json)
- [本次最终GPU审核](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/最终审核.json)
- [完整GPU重跑日志](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/fix_20261007_163655/independent_gpu_rerun.log)
- [修复后main.py check](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_164140_M1_entry_fixes_20261007/main_check_M1_entry_fixes_20261007.json)
- [新增5项测试日志](D:/PyCharmProjects/Jeon2019/work/m1_prefix_device_fix_tests_20261007.log)
- [10项审核适配测试日志](D:/PyCharmProjects/Jeon2019/work/m1_audit_adapter_after_fix_tests_20261007.log)
