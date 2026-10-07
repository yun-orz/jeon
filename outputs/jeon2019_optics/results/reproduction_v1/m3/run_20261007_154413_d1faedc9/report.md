# M3 数据流水线运行报告

状态：failed。未全部通过前不进入M4。

实际执行命令：

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\main.py" "data" "--config" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\config_m3_masked_index_20261007.json" "--phase" "all"
```

续接命令（写新运行目录、保留已有源数据和场景进度）：

```powershell
& "D:\dev\python\python3.10.4\python.exe" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\main.py" "data" "--config" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\reproduction_v1\config_m3_masked_index_20261007.json" "--phase" "all" "--resume" "D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics\results\reproduction_v1\m3\run_20261007_154413_d1faedc9"
```

原始文件列表见raw_source_records.json，来源页面/实际HEAD尺寸见source_review目录；预登记估计器见preregistered_protocol.json。
分组/真实规模差异、每场景状态、失败及未完成事项见audit.json/group_audit.json。未取得数据不能用fixture代替，不生成假scene或passed。
测试只允许来源/通道/完整性/重复元数据核对，半尺寸预处理推迟M7。Harvard历史img3/img4/img5及其关联组明确排除正式训练/验证/新测试。
校正为ref/相对灵敏度，不作λ/540光子换算；ICVL按实际bands，KAIST按w{波长}nm名称并保持反射率。跨数据集尺度与照明谱代理均为实施假设，不是绝对标定。
M2原包保持不变；最终包须带父指纹和数据版本，未拟合前共同增益保持pending。

失败/缺口：
[
  {
    "error": "PermissionError(13, '拒绝访问。')"
  }
]
