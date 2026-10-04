# 关于 `.npz` 数组数据

## 现状

工作区**完整**数据约 **16 GB**，其中 **15.6 GB（6980 个文件）** 是 `.npz`：

| 内容 | 说明 |
| --- | --- |
| 复场 `u2_complex` | `complex128`，探测器平面复振幅 |
| 强度 `intensity_raw` | `float64`，`Iraw = |u2|^2`，未做峰值归一化 |
| DOE 高度 `delta_h_m` / `mask` | 设计面型与孔径掩膜 |
| 采样对照数组 | 阶段 01 的收敛性研究（单个文件最大 425 MB） |

这些是**中间数据**，可由源码 + 配置确定性地重算，因此默认**不随仓库分发**。

## 为什么不能直接推

| 限制 | 数值 | 冲突 |
| --- | --- | --- |
| GitHub 单文件硬限 | 100 MB | 有 3 个文件超限（最大 425.4 MB） |
| GitHub 仓库建议上限 | < 1 GB | 超出 16 倍 |
| GitHub 仓库硬限 | 5 GB | 超出 3 倍 |
| Git LFS 免费额度 | 1 GB 存储/月 | 放不下 16 GB |

## 需要数组时的三种做法

### 1. 自己重算（推荐）

```powershell
cd outputs/jeon2019_optics
python -B main_stage02b.py          # 重新生成 18 组 psf_*.npz 与 2 份高度
```

只要 `config_stage02b.json` 不变，结果与仓库里收录的指标/图一致。

### 2. 用 Git LFS 上传（需要付费额度）

```powershell
git lfs install
git lfs track "*.npz"
git add .gitattributes
git add -f outputs/jeon2019_optics/results/stage02b1/run_20261002_163101/arrays
git commit -m "用 LFS 收录 02B-1 送审 run 的数组"
git push origin main
```

注意：免费额度 1 GB/月，16 GB 需付费；且上传耗时长。

### 3. 换用适合大文件的托管

Zenodo、机构网盘或对象存储（S3/OSS）更适合存放 GB 级科研数据，
可在本仓库 README 中给出外部下载链接与校验和。

## 如果只想补上某一小部分

在 `_select_runs.json` 里找到对应阶段的 run 路径，然后：

```powershell
git add -f <run 目录>/arrays
git commit -m "补录 <阶段> 的数组"
```
