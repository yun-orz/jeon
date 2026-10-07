# M3 正式交付报告

2026-10-07。M3 正式运行、外部独立审核和 `main.py check` 均通过。按任务文档要求完成后停止，未启动 M4、训练或封存测试。

[任务文档](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/reproduction_v1/tasks/M3_data_pipeline.md)

## 数据与划分

从 309 个实际候选场景（Harvard 77、ICVL 202、KAIST 30）按相关场景组与种子 20261006，选取 238 幅正式场景：

| 数据集 | 训练 | 验证 | 封存测试 | 合计 |
|---|---:|---:|---:|---:|
| harvard | 52 | 6 | 0 | 58 |
| icvl | 135 | 15 | 0 | 150 |
| kaist | 16 | 4 | 10 | 30 |

共 203 幅训练、25 幅验证，228 幅完成半尺寸预处理。10 幅 KAIST 测试不生成预处理图或补丁，封存至 M7。各库数量均达到本任务声明的新名单目标；作者精确子集未知，未宣称复现作者原名单。

Harvard 历史 img3/img4/img5 以实际 SHA 对应官方来源，其关联组全部排除正式训练、验证和测试，原诊断数据保留。ICVL 文件名前缀、原始 SHA 和固定整图重复描述用于分组；Master2900k/5000K/组合色温按同一照明采集族保守归组，在划分前登记。实际组无跨集。

ICVL 来源为用户完成访问授权取得的官方 `ICVL-BGU/ICVL_HS_2016`，固定提交 `d2cf6714224029431cf4cec551ca6753ed59bc52`。202 幅文件的实际 SHA 与已认证官方 LFS OID 逐项一致；本地源位于 `H:/我的云端硬盘/Jeon2019_data/reproduction_v1/icvl/mat`，新结果在项目 D 盘。原始候选清单及各文件 SHA 见 [raw_source_records.json](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/raw_source_records.json)。

## 波长、尺度与共同增益

使用 420–660 nm、10 nm 间隔的 25 波段。Harvard 按实际 ref/lbl/calib 读取并校正相对灵敏度，ICVL 从实际 400–700 nm 发布 bands 取索引 2–26，KAIST 按 EXR 的 w420nm…w660nm 通道名匹配并保留反射率。基础分辨率为 2×2 等面积平均，奇数末边登记丢弃，在线增强为 0.5/1/2。

Harvard/ICVL 各使用本库全部有效训练像元和 25 谱带的共同 RMS；读取时除以下列尺度，KAIST 不拟合：

| 数据集 | 尺度 |
|---|---:|
| harvard | 0.01276081180389756 |
| icvl | 740.4920826076886 |
| kaist | 1 |

每个训练场景从冻结索引中选取哈希最小的一条补丁；两个器件、RGB 三通道与有效像元等权统计。实际物理 RMS `67.88463217649466`，有效统计值数 `77507136`，共同增益 `0.003682718635788138`，对应目标 RMS 0.25。两器件使用完全相同的增益，M2 核、谱响应、映射及来源附件逐字节保持一致。

跨数据集尺度和 KAIST 平坦照明谱代理是实施假设，不称绝对辐射标定。10 nm 谱带因子由 M2 能量算子计入一次，不额外执行 λ/540 换算。`paper_alignment_passed=false`，本交付验证任务合同，不宣称论文全部指标复现。

## 索引与真实掩码

生成 30,000 条轻量 256×256 训练补丁索引，没有复制 30,000 个补丁文件。先按 scene_id 排序为每个训练场景登记一条，剩余按固定种子随机生成；所有 203 幅训练场景均覆盖。同种子重放 SHA：

`94ce8724db3f8a0a88621273fff08c5a76b6a3565fc9da6497be7bef49eaae16`

真实 Harvard imgb4/imgc9 在允许尺度下不存在全有效 256×256 区域，旧版额外全有效补丁约束导致增益冻结失败。旧失败运行和全部文件保留；新协议明确保留逐像元 mask，拒绝全无效补丁，拟合只统计有效像元。源数据、场景分组、划分与预处理结果均与旧版一致，修订前没有增益或最终算子冻结。

独立整数积分图核对全部 30,000 条索引：部分掩码补丁 `2912` 条，单条有效像元最少 `46`。后续训练必须使用返回的真实 mask，对损失按有效像元计算，不能把无效位置视为观测。

读取接口为 `m3_index.BoundedReader(prepared_scenes, scales, max_bytes=67108864)`；`read(scene_id, scale, y, x)` 返回 `(25×256×256 光谱补丁, 256×256 bool mask)`。训练尺度已在读取时应用，禁止重复归一化。读取测试补丁在 M7 前会被拒绝。

## 实际验证

21 项 M3 单元／回归测试通过。正式内部审核通过，外部审核独立核对 313 个来源文件 SHA、309 个真实格式／波长头、1304 个捕获／SHA 关联关系、全部索引掩码。三个库各取一幅训练和一幅验证的固定像元检查，半尺寸／校正误差为 0，mask 精确一致；训练增强独立标量核对及两个器件直接卷积／FFT 核对通过，最大相对差约 4.32×10⁻¹⁶。测试没有读取预处理像元、预测、成绩或 ROI。

实际缓存峰值 `66191360` 字节（63.12 MiB），全局配置 64 MiB，单场景峰值 `6619136` 字节。源读块峰值 `26214400` 字节（25 MiB）。缓存审核进程 RSS `258355200` 字节，增益拟合读样时实测 RSS 峰值 `314884096` 字节；RSS 与缓存分别记录，未把整个数据集加载到内存。

批量下载前以及续接前检查 D 盘空间；实际全部图像尺寸核算和 checkpoint／缓存保留空间证据保留。最终检查时 D 盘可用 88,238,600,192 字节。没有删除文件、清理历史或移盘。历史保护实际核对 15,942 文件、20,181,153,673 字节，通过；G0 本地 767 项核对通过。

M1 的只读 GPU 环境诊断仍记录 `KeyError(remaining_bytes)`，GPU 环境未通过，本交付没有把 M1 宣称为 passed。M3 文档明确 GPU 非必要，M0/M2/M3 验收已通过；后续 GPU 训练需完成对应环境里程碑。

实际成功执行命令（工作目录 `D:/PyCharmProjects/Jeon2019`）：

```powershell
python -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py data --config outputs/jeon2019_optics/reproduction_v1/config_m3_masked_index_20261007.json
python -X utf8 outputs/jeon2019_optics/reproduction_v1/tests/audit_m3_result.py --run outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171
python -X utf8 -u outputs/jeon2019_optics/reproduction_v1/main.py check --m3-run outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171 --stage M3_masked_v3_verified
python -X utf8 -m unittest discover -s outputs/jeon2019_optics/reproduction_v1/tests -p 'test_m3*.py' -v
```

## 正式产物与指纹

- [正式运行验收](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/audit.json)
- [外部独立审核](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/review_840f6c497c57/independent_completion_audit.json)
- [最终 main.py check](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_160046_M3_masked_v3_verified/main_check_M3_masked_v3_verified.json)
- [分组与实际规模](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/group_audit.json)
- [场景交接清单](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/scene_manifest.json)
- [数据版本](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/data_version.json)
- [训练尺度审核](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/normalization_audit.json)
- [共同增益审核](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/gain_audit.json)
- [训练索引](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/patch_index.jsonl)
- [测试封条](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/test_seal.json)
- [缓存实测](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/cache_audit.json)
- [最终 Jeon 算子](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/operator_jeon/package.json)
- [最终 Fresnel 算子](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/operator_fresnel/package.json)
- [稀疏掩码修订证据](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/index_revision_evidence.json)
- [实际尺寸与空间核算](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/actual_shape_space_verification.json)
- [历史诊断原始 SHA 对应](D:/PyCharmProjects/Jeon2019/outputs/jeon2019_optics/results/reproduction_v1/m3/run_20261007_154644_d46e6171/legacy_anchor_verification.json)
- [21项测试日志](D:/PyCharmProjects/Jeon2019/work/m3_tests_masked_v3_final_20261007.log)

| 类型 | SHA-256 指纹 |
|---|---|
| data | `4d966a84c3a19e5744f73b81277dddbdf3631d91d9df71c21e90a34bc7f0ce2b` |
| test_seal | `d16432ba619f99db4d7a417d299cd5f2cc4275239bb3b046f19647ac6a4d7a6c` |
| jeon | `2257b7e533bfa356a819cb787e33f9666042138bbd438664ff7923b27e0ba146` |
| fresnel | `9250ed2ef46ce7a2eb86ccd94d6a18825541e559a1b478c21436ca368d85ba1d` |

M3 交接状态：通过。上述算子绑定同一数据指纹并各保留 M2 父指纹。测试封条绑定来源、数据和两个算子，仅在 M7 按对应任务打开。当前工作停在 M3。
