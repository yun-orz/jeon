# 从官方HTTPS源下载GPU轮子，校验官方哈希，不替换旧环境。
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$gpuDownloadDir = Join-Path $PSScriptRoot 'dependencies/gpu_cu121_20261006'
if (Test-Path -LiteralPath $gpuDownloadDir) { throw '下载目录已存在；为保留历史，拒绝覆盖，请另设新目录。' }
New-Item -ItemType Directory -Path $gpuDownloadDir | Out-Null
$page = Invoke-WebRequest -UseBasicParsing -Uri 'https://download.pytorch.org/whl/cu121/torch/' -TimeoutSec 30
$links = @($page.Links | Where-Object { $_.href -match 'torch-2\.5\.1(%2B|\+)cu121-cp310-cp310-win_amd64\.whl' })
if ($links.Count -ne 1) { throw '官方索引中的Torch匹配项不唯一。' }
$torchLink = [string]$links[0].href
$parts = $torchLink -split '#sha256=', 2
if ($parts.Count -ne 2 -or $parts[1] -notmatch '^[a-f0-9]{64}$') { throw '缺少官方Torch SHA256。' }
$meta = Invoke-RestMethod -Uri 'https://pypi.org/pypi/sympy/1.13.1/json' -TimeoutSec 30
$sympy = @($meta.urls | Where-Object { $_.filename -eq 'sympy-1.13.1-py3-none-any.whl' })
if ($sympy.Count -ne 1) { throw 'SymPy轮子不唯一。' }
$items = @(
    @{filename='torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'; url=$parts[0]; sha256=$parts[1]},
    @{filename=$sympy[0].filename; url=$sympy[0].url; sha256=$sympy[0].digests.sha256}
)
$clock = [Diagnostics.Stopwatch]::StartNew()
foreach ($item in $items) {
    Write-Output ('开始下载：' + $item.filename)
    $target = Join-Path $gpuDownloadDir $item.filename
    Invoke-WebRequest -UseBasicParsing -Uri $item.url -OutFile $target -TimeoutSec 900
    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $item.sha256) { throw ('下载哈希不符，文件保留：' + $item.filename) }
    $item['bytes'] = (Get-Item -LiteralPath $target).Length
    $item['verified_sha256'] = $actual
    Write-Output ('官方SHA256通过：' + $item.filename + '，字节数：' + $item.bytes)
}
$report = @{checked_date='2026-10-06'; sources=$items; elapsed_seconds=$clock.Elapsed.TotalSeconds; index='https://download.pytorch.org/whl/cu121/torch/'; cpu_environment_modified=$false}
$jsonPath = Join-Path $gpuDownloadDir 'download_manifest.json'
if (Test-Path -LiteralPath $jsonPath) { throw '拒绝覆盖已有下载记录。' }
[IO.File]::WriteAllText($jsonPath, ($report | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
Write-Output ('下载和校验完成：' + $gpuDownloadDir)
