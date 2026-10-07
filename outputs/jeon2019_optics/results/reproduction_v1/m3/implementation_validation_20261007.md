# M3 实现验证与数据获取状态

本文件记录代码验证，不表示正式 M3 验收通过。当前正式数据运行状态为 `dependency_pending`，不得进入 M4。

## 已执行的验证

```powershell
python -X utf8 -m unittest discover -s outputs/jeon2019_optics/reproduction_v1/tests -p test_m3*.py -v
python -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py data --config outputs/jeon2019_optics/reproduction_v1/config_m3.json --resume outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_113235_bfc0e21e
```

12 项合成及离线测试通过，日志为 `work/m3_tests_acceptance_verified_20261007.log`。覆盖 MAT 校正、实际波长/HDF5 轴、EXR 名称映射、面积平均/掩码、关联组与历史排除、训练专用尺度、测试读取拒绝、30000 索引复现、实际缓存测量、共同增益标量与空间边界独立计算、完整分段续接、来源 SHA 拒绝以及最终包附件错误拒绝。所有 fixture 明确标记为合成，长期保留，不作为正式场景。

已下载第一幅官方 KAIST EXR；按实际 `w420nm` 至 `w660nm` 通道名读取 25 个谱通道。只做完整性及固定整图重复元数据核对，未预测、评分、选 ROI 或拟合测试数据。多通道扫描行输出块实测为 43212800 字节（约 41.2 MiB），该场景元数据检查耗时约 1.531 秒；测量见 `work/m3_actual_kaist_reader_verified_20261007.log`。此数字是输出块大小，不能代替正式训练缓存审核或宣称全进程内存峰值。

正式续接结果在 `run_20261007_113609_2196ca91/`，包含真实源文件列表、预登记协议、分组规模差异、场景级状态、失败/未完成说明及续接命令。M2 原始物理包未修改，旧入口的精确来源快照桥接单独登记。M3 数据版本和最终包必须通过独立审核后才能冻结。

## 尚未完成

- Harvard 两个官方归档与剩余 KAIST EXR 仍由有界下载任务获取，日志为 `work/m3_parallel_acquisition_20261007.log`；下载及提取进度保存在 `work/datasets/reproduction_v1/m3_20261007_v1/`。
- 用户已说明由 dsh 处理 ICVL 官方 Hugging Face 授权。当前未收到合法原始 MAT 目录，未接受账号联系信息共享条款或绕过授权。
- 全部来源到齐后须重新核对实际空间、来源/通道/组及数量差异，再正式拟合仅训练尺度和两器件共同增益、生成及重放索引、冻结数据/算子指纹与十幅测试封条，执行独立审核及 `main.py check`。

收到合法 ICVL MAT 目录后，将本地目录登记进 `config_m3.json` 的 `local_icvl_dir`；配置变化写入新的运行，不修改原报告。官方镜像的文件名/长度核对公开目录，实际 SHA 冻结；若未提供来源端 SHA，不宣称已验证来源端 SHA。

续接命令：

```powershell
python -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py data --config outputs/jeon2019_optics/reproduction_v1/config_m3.json --resume outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_113609_2196ca91
```

测试数据在 M7 前只允许来源、通道、完整性及重复元数据核对，半尺寸预处理亦推迟 M7。未取得真实数据时不得写 `passed`，不得填造场景，不得进入 M4。所有旧文件及失败尝试保留，未执行删除。

## 共享入口诊断修复后的续接

第一次 `main.py check` 发现旧 M1 检查器在原下载前缀文件不存在时未返回 `remaining_bytes`，导致共享入口抛出 `KeyError`。旧 M1 文件保持不变；共享入口现在明确登记 M1 诊断异常及 GPU 未验证，并继续独立的 M3 数据核对，不把该异常状态视为 M1 通过。

新增对应回归测试后，13 项测试全部通过，日志为 `work/m3_tests_entry_verified_20261007.log`。入口变更后的新正式数据运行在 `run_20261007_114230_94e8e9ae/`，状态仍为 `dependency_pending`；此前运行保留为历史实现版本，不代表当前源码冻结。

```powershell
python -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py data --config outputs/jeon2019_optics/reproduction_v1/config_m3.json --resume outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_113609_2196ca91
python -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py check --m3-run outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_114230_94e8e9ae --stage M3_dependency_pending_verified
```

共享入口修复后的检查日志为 `work/m3_main_check_verified_20261007.log`。最新待完成运行的续接入口如下；仍须先提供合法 ICVL 数据并完成其余官方下载。

```powershell
python -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py data --config outputs/jeon2019_optics/reproduction_v1/config_m3.json --resume outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_114230_94e8e9ae
```
