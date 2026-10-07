# M0：冻结复现协议与任务包

目标：将历史状态、论文参数、实施假设和保护边界汇合成可核验的统一依据。本任务只完成M0，不下载大包、不训练、不进入M1。

输入：根 `CURRENT_STATUS.md`、G0清单、既有阶段配置/说明/审核、正文及补充PDF。读取 `sources.json` 中限定来源。没有作者精确信息则记录未知并使用已声明方案，不无限延长检索。

允许写入：`outputs/jeon2019_optics/reproduction_v1` 和 `outputs/jeon2019_optics/results/reproduction_v1/m0`；只追加根状态文档。任何历史代码、原始数据、审核、结果及原CPU依赖不得改变。

实施：

1. 登记正式来源run、审核结论、已通过门槛、尚未完成事项和版本禁混规则；GPU只有CPU预审，不能写成GPU通过。
2. 建立四态参数证据表，核对手性、材料、连续/量化高度、RGB响应、网络、数据名单、噪声和指标。保留238与排除10之间的计数歧义，不补造名单。
3. 冻结本计划默认值、三种交接对象和八份任务；说明仍待M2/M3拟合或决定的字段。
4. 对历史文件建立相对路径/字节数/SHA256保护清单。状态文档保护原始字节前缀，允许追加。明确排除新流程、新结果、解释器缓存及待安装的GPU环境/续传包。
5. 提供默认只读 `check` 与 `report`。新增尚未实现命令必须拒绝，不能输出完成结果。校验相对路径、重复项、越界、内容篡改及追加保护，不能仅检查文件数量。
6. 执行G0全量、历史保护全量、论文指纹和审核链核对；从其他工作目录运行；验证新检查器的失败路径。输出实际验收后再登记M0通过。

实际入口命令：

```powershell
& D:/dev/python/python3.10.4/python.exe -B D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/build_m0.py
& D:/dev/python/python3.10.4/python.exe -B D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/main.py check
& D:/dev/python/python3.10.4/python.exe -B D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/main.py report
& D:/dev/python/python3.10.4/python.exe -B D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/tests/test_m0.py
```

成功：保护全量与G0通过；重要参数均有来源、默认值或明确待定节点；八任务都有输入/输出/停止条件；报告无GPU或论文伪通过。输出由 `m0_index.json` 指向本次冻结清单、阶段索引、验收和中文报告。后续重新执行 `build_m0.py` 应拒绝替换已冻结基线；如需新协议另建版本，不覆盖旧证据。

失败/预算/空间停止：保留未完成运行目录和错误，不生成通过索引；可继续只读 `check`。无恢复checkpoint，下一阶段输入是通过的M0报告和所有协议指纹。本任务完成后停止。
