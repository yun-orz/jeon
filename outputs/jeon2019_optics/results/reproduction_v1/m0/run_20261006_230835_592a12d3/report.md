# M0实际交付报告

M0通过；M1–M7尚未实施。程序核对通过不等于论文对齐。

创建时间：2026-10-06T23:08:35.412524+08:00。保护15942个历史文件、20181153673字节；G0全量767项通过；39项参数证据、8份Goal任务。

历史核对逐一比较SHA与字节长度；CURRENT_STATUS.md只追加，原始前缀SHA不变。历史GPU仍待安装；现有Torch为CPU版本。没有下载大包、训练、测试开封或删除文件。

实际命令：

```text
"D:\dev\python\python3.10.4\python.exe" -B "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\build_m0.py"
"D:\dev\python\python3.10.4\python.exe" -B "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\tests\test_m0.py"
```

有效协议：入口目录protocol.json。输入/输出指纹、关键数值、测试输出和验收详情见audit.json；历史来源索引见historical_stages.json；保护清单见protection.json。首次任务完整性检查失败已修复，过程记录见入口docs/validation_history.md；最终未解决失败为空。后续空间需求需各阶段预检。

继续入口：main.py无参数或check进行只读核对，report查看交接。M0已冻结时build_m0.py拒绝覆盖。下一阶段为tasks/M1_gpu_environment.md与tasks/M2_freeze_optics.md；本工作包完成后停止。

与论文未对齐项：作者响应、精确训练/测试名单、网络层表、2019传播脚本、SAM单位。替代响应、CW坐标、连续高度、Fresnel550nm、单位、边界、数据划分和40/60轮为明确实施假设。共享增益仍待训练拟合，不占位宣称已完成。paper_alignment_passed=false。
