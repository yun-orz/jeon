# ICVL 官方数据获取与验收报告（Google Drive 大数据盘版）

**工作包状态：`passed` ✅**

- 执行者：DeepSeek Harness
- 执行时间：2026-10-07（本地 11:20 起，下载 12:09–13:21，审核至 13:26，UTC+08:00）
- 任务定位：Jeon2019 `reproduction_v1` 的 **M3 前置数据获取子任务**
- 数据根：**H:**（`H:\我的云端硬盘\Jeon2019_data\reproduction_v1\icvl`）

> 本工作包只完成 ICVL 官方数据获取、Google Drive 存储和只读验收。未生成正式 M3 scene split，未拟合 normalization/gain，未修改 M2 operator package，未进入 M4/M6。未删除任何用户文件，未读取/打印/写入任何 Hugging Face token，未修改 `work/environments/jeon_gpu_cu121`，未修改 M1/M3 源码。

---

## 1. 最终结果

| 项目 | 结果 |
| --- | --- |
| `mat/*` 文件数 | **202 / 202**（与 pinned revision 远端清单一致） |
| 总字节数 | **28,199,177,384**（与远端逐字节一致） |
| 唯一 scene stem | **202**（无重复） |
| SHA256（磁盘实算） | **202 / 202** 全部生成 |
| SHA256 ↔ 仓库自身 oid 比对 | **202 / 202 匹配，0 不匹配** |
| HDF5 可打开 | **202 / 202** |
| `rad` / `bands` 存在 | **202 / 202** |
| `bands` 可读 | **202 / 202** |
| 波长严格递增 | **202 / 202** |
| 400–700 nm / 31 bands / 10 nm | **202 / 202** |
| 含 420–660 nm 的 25 bands | **202 / 202** |
| 云盘占位文件 | **0**（无 Offline 属性，实测可本地读） |
| 残留 `.part` / 临时文件 | **0** |
| `raw/*` / `preview/*` 下载 | **未下载**（未请求） |
| 失败项 | **0** |

三份验收 JSON 全部 `status = passed`：

```text
icvl_file_manifest.json        status = passed   (problems = 0)
icvl_hdf5_audit.json           status = passed   (failures = 0)
remote_oid_verification.json   status = passed   (matched 202 / 202)
```

### 实际命令（§11，逐字）

```powershell
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$env:HTTP_PROXY  = 'http://127.0.0.1:7897'

hf download ICVL-BGU/ICVL_HS_2016 `
  --repo-type dataset `
  --revision "d2cf6714224029431cf4cec551ca6753ed59bc52" `
  --include "mat/*" `
  --local-dir "H:\我的云端硬盘\Jeon2019_data\reproduction_v1\icvl" `
  --max-workers 2
```

`exit = 0`，stderr 无非空行。

---

## 2. 实际数据形状（重要，供 M3 显式处理轴序，§17）

审计**按存储原样**记录，**未做任何静默 transpose**。实测结果：

```text
rad_shape_stored     = [31, 1392, W]      <-- band 轴在【最前】，不是最后
rad_dtype            = float64            <-- 不是 float32
bands_shape_stored   = [31, 1]            <-- 列向量，不是 (31,)
bands_nm             = 400,410,...,700    (31 个，全部场景完全一致)
target_band_indices_420_660 = [2,3,...,26]  (由实际波长数值匹配得出)
```

**宽度 W 并非全部相同**（200 个场景为 1300）：

| W | 场景数 |
| --- | --- |
| 1300 | 186 |
| 1083 | 5 |
| 1084 | 2 |
| 1082 / 1202 / 1196 / 1194 / 1191 / 1027 / 1023 / 1018 / 1021 | 各 1 |

即 **H×C 固定为 1392×31，但宽度在 1018–1300 之间变化**，M3 不得假定固定宽高。

每个场景的完整元数据见 `icvl_hdf5_audit.json` 的 `scenes[]`。

---

## 3. 存储与路径

### 3.1 选定 H:（用户决定）

`storage_probe.json`

| 盘符 | 存在 | 云映射盘 | 挂载根可写 | My Drive 可写 | 写测试 | 剩余空间 |
| --- | --- | --- | --- | --- | --- | --- |
| G: | 是 | 是（`bestyongcli@gmail.com`） | **否** | **是** | 通过 | 共享池 |
| **H:** | 是 | 是（`19xingyunli@gmail.com`） | **否** | **是** | 通过 | 共享池 |

按 §2.1 的 G: 优先规则算法优选为 **G:**，但**用户明确选择 H:**，已如实记录偏离（§10 要求）：

```text
algorithmic_preference_drive = G:
selected_drive               = H:   (用户决定 2026-10-07)
```

### 3.2 目标目录的真实形态（§2.1 字面路径在本机不可创建）

Google Drive for desktop 把云盘挂成**虚拟文件系统**：

- `G:\` / `H:\` 的**根目录是虚拟的，写入被静默丢弃** —— `New-Item`／`CreateDirectory` 看似成功，`Test-Path` 立刻返回 `False`，`CreateDirectory` 抛 `未能找到文件`。
- 命名空间只有 4 项：`.shortcut-targets-by-id`、`我的云端硬盘`、`$RECYCLE.BIN`、`.Encrypted`。
- **唯一可写根是 `<盘符>:\我的云端硬盘\`**（My Drive）。
- `H:\My Drive`（ASCII 别名）在本机**不存在**。

因此 §2.1 的字面路径 `H:\Jeon2019_data\reproduction_v1\icvl` 在本机**创建不出来**，实际等价路径为：

```text
H:\我的云端硬盘\Jeon2019_data\reproduction_v1\icvl
```

逻辑根仍记录为盘符无关形式 `H:/Jeon2019_data/reproduction_v1/icvl`（§21）。

### 3.3 ⚠️ G: 与 H: **不是**两块独立存储

初看两盘账号标签不同，像是互为备用。深查**推翻了该判断**：

```text
G: 卷序列号 = 19831116
H: 卷序列号 = 19831116      <-- 完全相同
G: 剩余 = 326325633024 字节
H: 剩余 = 326325633024 字节  <-- 逐字节相同
```

**G: 与 H: 是同一个卷，被分别以两个账号标签挂载，共享同一个可用空间池。**

影响：

1. **"G: 不够就换 H:" 是假备用方案** —— 换盘符不会多出任何容量；
2. 若真遇到 `space_stopped`，只能去 Google Drive 账号本身腾空间，或改用真正的本地盘 —— 两者都超出本工作包权限（§10 禁止 Harness 删除用户文件）。

`storage_probe.json` 已记录 `volume_serial_numbers`、`drive_letters_are_independent = false`。

### 3.4 空间判断（§10）

```text
待下载数据量 = 28,199,177,384 bytes = 26.263 GiB
安全余量     = max(20 GiB, 10%) = 20 GiB
所需最小剩余 = 46.263 GiB
下载前实测   = 301.6 GiB  -> 通过（余量约 6.5 倍）
下载后剩余   = 275.47 GiB
```

云盘报告的"剩余空间"是**账号配额**而非固定本地盘，实测会在 10 分钟内变化约 2.5 GiB。因此 `run_download_and_audit.ps1` **不采信缓存探测值**，而在下载前**重新实测**（step 1b），不足则 `space_stopped` 退出且不做任何删除。

---

## 4. 环境与工具链（§6）

```text
hf executable            : D:\dev\python\python3.10.4\Scripts\hf.exe
hf --version             : 1.33.0
python（CLI）+ huggingface_hub : 3.10.4 / 1.33.0
hf_xet                   : 1.6.0
认证用户                 : yunorz（token 从未被读取/打印/写入）
```

为 §16 的 HDF5 审计新建的**独立轻量环境**（基础解释器无 h5py）：

```text
D:\PyCharmProjects\Jeon2019\work\environments\icvl_download
  python 3.10.4
  h5py   3.16.0
  numpy  2.2.6
  httpx  0.28.1
  huggingface_hub 2.1.1
```

**未修改 `work/environments/jeon_gpu_cu121`**（已只读确认可正常启动），未修改 M1 脚本/下载目录/结果/Python 环境。基础解释器 `D:\dev\python\python3.10.4` 未被改动。

---

## 5. 审核脚本（§19）

| 文件 | 作用 |
| --- | --- |
| `audit_icvl_download.py` | 唯一审核脚本：文件级 SHA256 manifest + HDF5 元数据审计。只读 `.mat`。 |
| `fetch_pinned_tree.py` | **带认证**抓取 pinned revision 清单（见 §6.1） |
| `verify_against_remote_oids.py` | 本地 SHA256 ↔ 仓库自身 LFS oid 逐文件比对 |
| `probe_storage.ps1` | §4 存储探测（含卷身份识别、探测目录自清理） |
| `run_dry_run.ps1` | §9 dry-run 并完整留证 |
| `run_download_and_audit.ps1` | §11–§18 一键执行（先验授权 → 实测量空间 → 下载 → 审核 → oid 校验） |
| `selftest_audit.py` | 自检：多结构变体 + digest 夹具 |
| `gating_probe.py` | 授权与 gating 证据 |

### 5.1 自检发现并修复的两个「假通过」缺陷

自检用合成数据覆盖真实 ICVL 可能出现的**全部轴序 / dtype 变体**（`rad (H,W,31)`、`rad (31,H,W)`、MATLAB 复合 dtype、`bands` 为 `(1,31)` float32、30-band 越界样本），**44/44 通过**。过程中抓到两个会**误判为通过**的真实缺陷：

**缺陷一：** 一个 30 bands / 400–690 nm 的文件恰好含有 420–660 nm 这 25 个波长，原实现只检查"25 个目标波长是否存在"，于是判通过 —— 但违反了文档要求的 400–700 nm / 31 bands / 10 nm 结构。已把 `is_400_700_31_10nm` 纳入失败判据，并要求目标 band 数**恰好 25** 且严格递增。

**缺陷二：** manifest 原先**只比对文件大小**，从不比对摘要 —— 同尺寸损坏会显示 `passed`。已加入本地 SHA256 ↔ 仓库声明 oid 的逐文件比对，并用"大小正确但 oid 错误"的夹具证明该比对确实会拦下。

另外把 `sha256` 的来源明确为**只用磁盘实算**：原先的"优先读 `hf` 缓存 digest"路径有隐患，该字段在 Xet 后端下**可能是 etag 而非 SHA256**，会让 manifest 名不副实。

**用于 420–660 nm 的索引由实际波长数值匹配得出，源码中不存在 `[数字:数字]` 形式的硬编码切片（自检扫描确认）。**

---

## 6. 执行中发现并修复的两个环境陷阱

### 6.1 匿名抓取的清单会**静默**丢失全部摘要

`huggingface.co` 对 **gated 仓库的匿名请求**会把每个 `lfs.oid` 涂成 **64 个 `*`**（已确认是服务端行为：原始响应字节里就是 `*`；对照实验显示公开非 gated 仓库 `princeton-nlp/SWE-bench`、`openai/gsm8k` 返回真实 64 位 hex）。

危险之处在于**它不报错**：清单照样 200 返回、文件数照样 202、大小照样正确，只是摘要无声地失效了 —— manifest 的远端摘要比对会因此变成**空洞的通过**。

**处置：** 新增 `fetch_pinned_tree.py`，**必须带认证**抓取（token 只进请求头，从不落盘/打印）。若检测到涂改则**拒绝写出**清单并 exit 3。重新抓取后 811 个文件条目全部带真实 oid（涂改 0 个）。`verify_against_remote_oids.py` 同样已改为带认证，否则会退化为 `not_verified_oids_redacted`。

> 附带更正一处我先前的判断：我曾据此推断"逐文件摘要校验只能在授权后进行"——这个结论**方向正确但原因说错了**。真实原因是**匿名请求**（而非授权状态）导致涂改，因此即便未登录，只要带 token 就能拿到真实 oid。

### 6.2 `httpx` / `huggingface_hub` 需要显式代理

本机 WinINET 配置了代理 `127.0.0.1:7897`，但**没有**设置 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量。后果：

- `httpx`（`hf` CLI 底层）自动读 WinINET 代理但 TLS 握手失败：`httpcore.ConnectError: EOF occurred in violation of protocol (_ssl.c:997)`；
- 完全不走代理也不通：`getaddrinfo failed`；
- `pip` 同样失败：`SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol'))`。

**解法（已验证）：显式设置环境变量。**

```powershell
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
$env:HTTP_PROXY  = 'http://127.0.0.1:7897'
```

该变量仅影响当前进程，未写入系统环境或任何凭证文件。**不设这个变量会把授权问题误判成网络问题。**

另一个陷阱：`verify_against_remote_oids.py` 在隔离环境里运行时，若该环境**没装 `huggingface_hub`**，`get_token()` 就不可用，脚本会静默退化为匿名请求 → oid 被涂改 → 报 `not_verified_oids_redacted`。已把 `huggingface_hub` 装入隔离环境修复。

---

## 7. 只读性验证（§16：不得修改原始 `.mat`）

审核脚本对 `.mat` 只以 `rb` / HDF5 只读方式打开。已实测验证：

- 下载后总字节数 = **28,199,177,384**，与下载完成时**完全一致**；
- 审核运行后对 3 个文件（含最早与最晚写入者）**重新计算 SHA256**，与 manifest 记录**全部一致** → 审核确为只读；
- 无任何文件被重新保存、重压缩或改格式。

§15 云盘状态检查：**无文件带 `Offline`（占位）属性**；202 个文件全部被 HDF5 成功打开并读取 `bands`，**读取错误 0**；另用普通二进制读取独立复核 3 个文件（成功读到字节）。因此 `cloud_placeholder_detected = false`，且"文件名存在"已被实测的**本地可读**所证实。

---

## 8. Git 与数据安全（§24）

- 原始数据落在 **H:**，属仓库之外的另一卷，结构上不可能被 commit。
- 仓库内 **tracked `.mat` = 0、tracked `.npz` = 0**；无 token / credential 被跟踪。
- `work/datasets/` 整体未跟踪（`?? work/datasets/`），本工作包产物**未被 `git add`**。

**发现并仅作报告（按 §24 不做顺手重构）：**

1. `.gitignore` **没有 `*.mat` 规则**（同类只有 `*.npz`），而仓库工作区内有 5 个 `.mat`（Harvard 下载 + M3 fixtures，约 350 MiB）。目前因 `work/datasets/` 未跟踪而未暴露，但 `git add -A` 会入库。建议由用户/Codex 决定是否补规则；**本工作包未修改 `.gitignore`。**
2. `.gitignore` 无 `work/environments/` 规则（§3 要求环境不得上云/入库）。

---

## 9. 与 Codex/M3 的接口（已核对，未改动 M3）

执行期间观察到 Codex/M3 并行开发与下载（新增 `m3_validate.py`、`test_m3_download.py`、`test_m3_validation.py` 等）。核对其配置后确认：

- `config_m3.json` 的 **`local_icvl_dir` 需要指向含 `.mat` 的目录**；
- `m3_pipeline.source_records()` 会**枚举该目录内的 `*.mat`**，校验"文件名在官方目录中"且"大小等于官方大小"，并要求 `icvl_files >= 150`。

**下载完成后应设为：**

```text
H:\我的云端硬盘\Jeon2019_data\reproduction_v1\icvl\mat
```

**本工作包没有改 `config_m3.json`**（归 Codex/M3 所有）。机器可读版本见 `m3_handoff.json`。

**交叉验证：** M3 自己的 `source_review/.../icvl_catalog.txt` 记录 **202 个 mat / 28,199,177,384 字节**，与本工作包独立拉取的权威清单**完全一致**。

**给 M3 的提示：** M3 的校验**只比文件大小**，抓不到"大小相同但内容损坏"；本工作包额外提供 SHA256 层，并已与仓库自身 oid **202/202 匹配**，可直接作为强校验依据。

M3 的并行下载是 **Harvard** 数据（`CZ_hsdb`、`CZ_hsdbi`、`scene0*_reflectance`），**不是 ICVL**，无数据冲突；带宽共享，故 ICVL 下载保持 `--max-workers 2`（§6）。

---

## 10. §25 成功条件逐条核对

| # | 条件 | 结果 |
| --- | --- | --- |
| 1 | Hugging Face 用户授权有效 | ✅ `hf auth whoami` = Logged in (yunorz) |
| 2 | source revision 已固定 | ✅ `d2cf6714224029431cf4cec551ca6753ed59bc52` |
| 3 | dry-run 已完成 | ✅ 202 文件 / 28.2G |
| 4 | 存储盘已明确选择 | ✅ H:（用户决定） |
| 5 | 空间通过 | ✅ 46.26 GiB 需 vs 301.6 GiB 有 |
| 6 | 仅下载 `mat/*` | ✅ 未请求 raw/preview |
| 7 | 所有 expected MAT 下载完整 | ✅ 202/202，字节数精确一致 |
| 8 | SHA256 manifest 完成 | ✅ 202/202，含与仓库 oid 比对 |
| 9 | 所有文件可作为 HDF5 打开 | ✅ 202/202 |
| 10 | `rad` / `bands` 存在 | ✅ 202/202 |
| 11 | 实际 wavelength 审核完成 | ✅ 202/202，数值已逐场景记录 |
| 12 | 420–660 nm 的 25 bands 全部存在 | ✅ 202/202 |
| 13 | Google Drive 文件实际可读 | ✅ 无占位，实测可读 |
| 14 | 无 M1/M2/M3 共享源码被越权修改 | ✅ 仅新增本工作包自身文件 |
| 15 | 无 token 泄露 | ✅ 从未读取/打印/写入 token |
| 16 | 无原始 MAT 被修改 | ✅ 重算 SHA256 一致 |
| 17 | 报告含实际命令、环境、revision、数据根、文件数、总字节、失败项 | ✅ 本报告 |

**全部 17 项满足 → `status = passed`。**

---

## 11. 交接

```text
ICVL acquisition status: passed
```

### 实际数据根

```text
逻辑根（写入 M3 配置）: H:/Jeon2019_data/reproduction_v1/icvl
M3 的 local_icvl_dir    : H:\我的云端硬盘\Jeon2019_data\reproduction_v1\icvl\mat
```

见 `dataset_roots.json` 与 `m3_handoff.json`。M3 必须通过配置读取 `dataset_root`，**不得在算法代码中硬编码盘符**；数据身份由 `SHA256 + source revision + scene ID` 决定，盘符只是物理位置（§21）。注意盘符迁移（H:→G:）**不带来额外容量**（同卷，见 §3.3）。

### 审核文件

```text
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\icvl_source.json
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\icvl_file_manifest.json
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\icvl_hdf5_audit.json
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\remote_oid_verification.json
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\storage_probe.json
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\dataset_roots.json
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\m3_handoff.json
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\dry_run.txt
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\download_stdout.txt
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\download_stderr.txt
D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition\acquisition_report.md
```

### 声明

> 本工作包只完成 ICVL 官方数据获取、Google Drive 存储和只读验收。未生成正式 M3 scene split，未拟合 normalization/gain，未修改 M2 operator package，未进入 M4/M6。

**scene split 与正式 150-scene 子集选择全部留给 Codex/M3**（§12.2、§23）。本工作包下载了完整 202 scenes，**未随机选、未按文件名前 N 选、未按下载速度选、未按图像内容选**，也未查看任何重建结果或计算任何 test 指标。
