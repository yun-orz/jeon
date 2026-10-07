# 用户可在普通PowerShell手动运行；当前自动审批恢复后也可由Codex执行。
# 保留旧下载、诊断片段和全部分块；不删除或覆盖已有文件。
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$partial = Join-Path $PSScriptRoot 'dependencies/gpu_cu121_20261006/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'
$resumeDir = Join-Path $PSScriptRoot 'dependencies/gpu_cu121_resume_20261006'
if (Test-Path -LiteralPath $resumeDir) { throw '续传目录已存在；保留历史并拒绝覆盖。' }
if (-not (Test-Path -LiteralPath $partial)) { throw '找不到已保存的部分安装包。' }
$prefixBytes = (Get-Item -LiteralPath $partial).Length
$totalBytes = [long]2449372784
$officialSha = '9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f'
if ($prefixBytes -le 0 -or $prefixBytes -gt $totalBytes) { throw '已有下载大小非法。' }
New-Item -ItemType Directory -Path $resumeDir | Out-Null
$assembled = Join-Path $resumeDir 'torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'
$parts = @()
$gpuOutputStream = [IO.File]::Open($assembled, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try {
    # 已有前缀只读复制，最终整包SHA校验才决定其有效性。
    $gpuInputStream = [IO.File]::OpenRead($partial)
    try { $gpuInputStream.CopyTo($gpuOutputStream) } finally { $gpuInputStream.Dispose() }
    if ($gpuOutputStream.Position -ne $prefixBytes -or (Get-Item -LiteralPath $partial).Length -ne $prefixBytes) {
        throw '下载前缀在复制期间改变，拒绝继续。'
    }
    $blockBytes = [long](32 * 1024 * 1024)
    $index = 0
    for ($offset = $prefixBytes; $offset -lt $totalBytes; $offset += $blockBytes) {
        $end = [Math]::Min($offset + $blockBytes - 1, $totalBytes - 1)
        $target = Join-Path $resumeDir ('part_{0:D3}.bin' -f $index)
        $uri = 'https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-win_amd64.whl?resumePart=' + $index
        Write-Output ('续传分块{0}：{1}–{2}' -f $index, $offset, $end)
        $response = Invoke-WebRequest -UseBasicParsing -Uri $uri -Headers @{Range=('bytes={0}-{1}' -f $offset, $end)} -OutFile $target -PassThru -TimeoutSec 120
        $range = $response.Headers['Content-Range'] -join ','
        $expectedRange = 'bytes {0}-{1}/{2}' -f $offset, $end, $totalBytes
        if ($response.StatusCode -ne 206 -or $range -ne $expectedRange -or (Get-Item -LiteralPath $target).Length -ne ($end - $offset + 1)) {
            throw ('服务端分块范围或大小不符，保留文件：' + $target)
        }
        $gpuChunkStream = [IO.File]::OpenRead($target)
        try { $gpuChunkStream.CopyTo($gpuOutputStream) } finally { $gpuChunkStream.Dispose() }
        $parts += @{filename=[IO.Path]::GetFileName($target);range=$range;sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()}
        $index++
    }
} finally { $gpuOutputStream.Dispose() }
$actual = (Get-FileHash -LiteralPath $assembled -Algorithm SHA256).Hash.ToLowerInvariant()
if ((Get-Item -LiteralPath $assembled).Length -ne $totalBytes -or $actual -ne $officialSha) {
    throw '最终Torch包未通过官方SHA256；保留已有文件，不安装。'
}
$meta = Invoke-RestMethod -Uri 'https://pypi.org/pypi/sympy/1.13.1/json' -TimeoutSec 30
$sympy = @($meta.urls | Where-Object { $_.filename -eq 'sympy-1.13.1-py3-none-any.whl' })
if ($sympy.Count -ne 1) { throw '官方SymPy轮子不唯一。' }
$sympyPath = Join-Path $resumeDir $sympy[0].filename
Invoke-WebRequest -UseBasicParsing -Uri $sympy[0].url -OutFile $sympyPath -TimeoutSec 120
$sympySha = (Get-FileHash -LiteralPath $sympyPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sympySha -ne $sympy[0].digests.sha256) { throw 'SymPy官方SHA256不符；拒绝安装。' }
$report = @{checked_date='2026-10-06';partial_source=$partial;partial_bytes=$prefixBytes;torch_bytes=$totalBytes;torch_sha256=$actual;parts=$parts;sympy_url=$sympy[0].url;sympy_sha256=$sympySha}
[IO.File]::WriteAllText((Join-Path $resumeDir 'download_manifest.json'), ($report | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
$python = Join-Path $PSScriptRoot 'environments/jeon_gpu_cu121/Scripts/python.exe'
& $python -B -m pip install --ignore-installed --no-deps --no-index --no-cache-dir --no-compile $assembled $sympyPath
if ($LASTEXITCODE -ne 0) { throw '独立虚拟环境离线安装失败。' }
& $python -B -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.__file__); assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw '安装后实际CUDA调用仍不可用；不算GPU验收完成。' }
Write-Output ('GPU安装就绪；下一步运行：' + (Join-Path $root 'outputs/jeon2019_optics/main_g3_gpu.py'))
