# 阶段 02B-1 工作区清理：把 `_cleanup_manifest.json` 中列出的项**送进回收站**（可还原）。
#
# 安全闸门（任一不通过即中止，不做任何删除）：
#   1. 每个目标必须真实存在；
#   2. 每个目标必须位于允许的根目录之下，且不等于该根目录本身；
#   3. 目标不得出现在清单的 keep 列表里。
# 删除方式为回收站，不是永久删除；用户可从回收站还原。
param(
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
Set-Location 'D:\PyCharmProjects\Jeon2019'

$manifest = Get-Content '_cleanup_manifest.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$targets = @($manifest.delete)
$keeps = @{}
foreach ($k in @($manifest.keep)) { $keeps[$k.ToLowerInvariant()] = $true }

$allowedRoots = @(
    'D:\PyCharmProjects\Jeon2019\outputs\jeon2019_optics',
    'D:\PyCharmProjects\Jeon2019\work'
)

# ---------------- 安全闸门 ----------------
$problems = New-Object System.Collections.Generic.List[string]
foreach ($t in $targets) {
    $full = [System.IO.Path]::GetFullPath($t)
    if (-not (Test-Path -LiteralPath $full)) { $problems.Add("不存在: $full"); continue }
    $ok = $false
    foreach ($root in $allowedRoots) {
        if ($full -eq $root) { $problems.Add("不允许删除根目录本身: $full"); $ok = $true; break }
        if ($full.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) { $ok = $true; break }
    }
    if (-not $ok) { $problems.Add("越界(不在允许根目录下): $full"); continue }
    if ($keeps.ContainsKey($full.ToLowerInvariant())) { $problems.Add("同时出现在保留清单: $full") }
}
if ($problems.Count -gt 0) {
    Write-Output '安全闸门不通过，已中止，未删除任何内容：'
    $problems | ForEach-Object { Write-Output ('  - ' + $_) }
    exit 2
}
Write-Output ('安全闸门通过：目标 ' + $targets.Count + ' 项')

if (-not $Apply) {
    Write-Output '（预演模式，未删除任何内容；加 -Apply 才真正执行）'
    exit 0
}

Add-Type -AssemblyName Microsoft.VisualBasic
$ok = 0
$failed = New-Object System.Collections.Generic.List[string]
$n = 0
foreach ($t in $targets) {
    $n++
    $full = [System.IO.Path]::GetFullPath($t)
    try {
        if (Test-Path -LiteralPath $full -PathType Container) {
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory(
                $full,
                [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
                [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
        }
        else {
            [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile(
                $full,
                [Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,
                [Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin)
        }
        if (Test-Path -LiteralPath $full) {
            $failed.Add("仍在磁盘上: $full")
        }
        else {
            $ok++
        }
    }
    catch {
        $failed.Add("$full -> $($_.Exception.Message)")
    }
    if ($n % 200 -eq 0) { Write-Output ("  进度 $n/" + $targets.Count + "（成功 $ok）") }
}

Write-Output ''
Write-Output ("已送回收站：$ok / " + $targets.Count)
if ($failed.Count -gt 0) {
    Write-Output ("失败 $($failed.Count) 项：")
    $failed | Select-Object -First 30 | ForEach-Object { Write-Output ('  - ' + $_) }
    exit 1
}
exit 0
