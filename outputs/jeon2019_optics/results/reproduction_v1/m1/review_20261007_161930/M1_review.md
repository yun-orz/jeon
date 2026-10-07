# M1 独立审核报告

2026-10-07，北京时间。审核对象：dsh 的 M1 执行产物，不执行安装、下载、删除、训练或测试预测。

**结论：历史 GPU 验收存档复核通过；当前 CUDA 重验未通过，当前 M1 不能重新签为 passed。发现两处可复现的检查／恢复入口缺陷。** 原安装文件完整，不应把 `remaining_bytes` 异常解释为 Torch 未安装。

## 需要处理的事项

1. **检查入口缺陷。** [m1_environment.py:161](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/m1_environment.py) 在旧下载前缀不存在时只返回 exists/path/bytes，未返回 remaining_bytes；check() 第171行直接访问该键，实际复现 `KeyError(remaining_bytes)`，尚未检查 GPU 就退出。应在前缀缺失分支返回完整剩余字节（TORCH_BYTES），并对安装前空间规划与安装后运行环境检查分别标明含义。修复后增加“没有前缀但已安装环境存在”回归用例。

2. **报告所述恢复命令不可用。** [m1_download.py:42](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/m1_download.py) 无条件设置 prefix=PREFIX，[m1_resume.py:163](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/m1_resume.py) 对缺失前缀明确报错。当前旧部分包与整包目录均不存在，直接调用生产 Resumer.ensure_prefix() 实际复现“找不到已有前缀文件”；未联网或下载正文。dsh 报告“重新执行 download 即可从头下载”的描述与代码不符。建议支持显式从零模式，或在无可核验前缀时传 prefix=None，保持整包官方 SHA 必须通过的安装门槛，并增加丢失前缀与工作目录恢复用例。

3. **当前 CUDA 设备不可见，原因尚未确定。** GPU 解释器 Torch 2.5.1+cu121 可正常导入，但 cuda_available=false、device_count=0；小型 CUDA 矩阵运算报 No CUDA GPUs are available。沙箱外诊断同样失败：CUDA Driver API cuInit(0) 返回 100 / CUDA_ERROR_NO_DEVICE；nvidia-smi 返回4并提示管理员/TCC设备条件。注册表存在 RTX4060 驱动，但这不能证明当前进程能访问设备。无法据此断言电脑 GPU 损坏，也不能把此次失败确定归因为沙箱；需要在可访问 GPU 的执行环境中重新运行 CUDA 自检和正式独立审核。

## 已通过的独立检查

| 项目 | 本次结果 |
|---|---|
| 正式 run 6个文件 SHA | 与历史最终审核全部一致 |
| 正式 source_manifest 57个源码 SHA | 全部一致 |
| 4个模型／数据来源树 SHA | 全部一致 |
| 原 CPU 依赖快照 | 完全一致 |
| M0 历史保护 | 15942文件，20181153673字节，通过 |
| G0 只读核对 | 767项，无跳过，通过 |
| Torch 已安装 RECORD | 10818文件／4549978622字节，无缺失或SHA差异 |
| SymPy 已安装 RECORD | 1558文件／26280417字节，无缺失或SHA差异 |
| M1 回归测试 | 26项通过 |
| 配置／原阈值／AMP与TF32记录 | 配置一致，未放宽门槛，历史AMP/TF32关闭 |

| 数值项目 | 本次独立重算误差 | 冻结门槛 |
|---|---:|---:|
| FP64 forward | 5.043957e-16 | 1e-10 |
| FP64 adjoint | 4.378535e-16 | 1e-10 |
| FP64 gradient | 3.763084e-16 | 1e-10 |
| GPU／新CPU预测 | 5.345474e-07 | 1e-5 |
| HQS阶段最大项 | 8.522682e-07 | 1e-5 |
| 独立式(21)三阶段最大项 | 7.666523e-08 | 1e-5 |
| 已保存损失标量差异 | 2.367254e-07 | 1e-5 |

以上数学核对读取的是已保存的训练／验证数组，是独立存档复核，不是本次真实 GPU 运算。全梯度与 Adam 更新后的完整参数数组没有保存；其历史误差 3.147610e-06（≤1e-4）和 1.805231e-09（≤1e-5）有原运行与当时无保存重跑一致的审核记录支持，但当前 CUDA 不可用，本次没有重新计算这两项。

## 证据与说明的边界

官方 wheel 和续传块已不在本地，无法本次重算其官方整包 SHA；只能核对保存的下载 SHA、字节数与任务冻结值，以及当前安装目录的 RECORD 完整性。RECORD 是安装文件一致性证明，不代替对官方 wheel 的认证。两个下载目录确已缺失；报告中的删除者归因不能仅凭并发 M2/M3 日志得到证实，本次不判定删除者。

dsh 报告“原审核器会覆盖历史产物，因此必须改写输出路径”与原审核器当前源码不符：其保存逻辑已有文件名 repeat 循环避让。适配层仍可使用，但关于必要性的说明应更正。其 AST 测试并非整份主审核逻辑的 AST 等价证明，仅一部分函数作等价检查，其余多为阈值与文本断言；本次采用实际源码／存档指纹与独立数值重算补充验证。没有发现原数值阈值被放宽。

之前 M3 报告将 GPU 环境注明未就绪，依据是检查器异常。准确表述应为：M1 已安装且历史执行一致性有通过证据；检查器在下载前缀被移除后出错；当前 CUDA 可用性还须以真实设备自检为准。本次既不否定已通过的历史数值存档，也不将它直接当成当前 GPU 可用性的证据。

## 本次执行与产物

运行既有适配层时，通过 importlib 加载原文件并只把 PERSIST_DIR 重定向至本新审核目录；未修改审核条件或跳过 CUDA 门槛。它因 CUDA 不可用在门槛处拒绝验收，完整错误保留。随后另用只读脚本进行存档核对，明确分开两种结果。

```powershell
python -B -X utf8 -m unittest discover -s outputs/jeon2019_optics/reproduction_v1/tests -p 'test_m1*.py' -v
python -B -X utf8 outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/offline_archive_review.py
```

CUDA、安装 RECORD 与生产入口异常的实际调用及返回值分别保存在下列 JSON；没有重新下载2.3 GiB安装包，没有改动生产代码、旧报告、模型、M3 冻结版本或CPU环境。

- [独立存档审核](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/offline_archive_review.json)
- [只读存档审核脚本](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/offline_archive_review.py)
- [当前CUDA设备诊断](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/current_gpu_access.json)
- [安装文件完整性](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/installed_integrity.json)
- [入口缺陷复现](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/current_entry_failures.json)
- [现有独立审核器重跑失败日志](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/independent_adapter_rerun.log)
- [26项测试日志](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m1/review_20261007_161930/m1_regression_tests.log)
- [dsh原报告](D:/PyCharmProjects/Jeon2019/outputs/执行_M1_GPU环境_20261007/报告.md)
- [dsh历史最终审核](D:/PyCharmProjects/Jeon2019/outputs/执行_M1_GPU环境_20261007/最终审核.json)
