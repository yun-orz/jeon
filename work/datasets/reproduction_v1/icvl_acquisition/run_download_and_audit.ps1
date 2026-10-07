# run_download_and_audit.ps1 -- ICVL acquisition sections 11-18, one command.
#
# ASCII-only source ON PURPOSE: Windows PowerShell 5.1 decodes BOM-less UTF-8
# source as ANSI/GBK, so any non-ASCII literal here would be corrupted. The
# localized Google Drive "My Drive" path component is passed in via -DataRoot.
#
# Sequence:
#   1. require a working Hugging Face login (section 7) -- refuses to download otherwise
#   2. run the EXACT section 11 download command, pinned revision, mat/* only,
#      --max-workers 2, capturing stdout and stderr separately (section 13)
#   3. verify the download actually completed (exit code AND file count)
#   4. run the read-only audit (sections 14-18) -> icvl_file_manifest.json,
#      icvl_hdf5_audit.json
#
# It never deletes anything and never re-saves, re-compresses or reformats a .mat.

param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [string]$OutDir   = 'D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition',
    [string]$Revision = 'd2cf6714224029431cf4cec551ca6753ed59bc52',
    [string]$Proxy    = 'http://127.0.0.1:7897',
    [string]$AuditPy  = 'D:\PyCharmProjects\Jeon2019\work\environments\icvl_download\Scripts\python.exe'
)

$ErrorActionPreference = 'Continue'
if ($Proxy) { $env:HTTPS_PROXY = $Proxy; $env:HTTP_PROXY = $Proxy }
# Line-based logs instead of multi-threaded progress bars, which would otherwise
# spam download_stderr.txt with thousands of redraw lines. This does not change
# what is downloaded, only how progress is reported.
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'

$repoId     = 'ICVL-BGU/ICVL_HS_2016'
$stdoutFile = Join-Path $OutDir 'download_stdout.txt'
$stderrFile = Join-Path $OutDir 'download_stderr.txt'
$manifest   = Join-Path $OutDir 'icvl_file_manifest.json'
$auditJson  = Join-Path $OutDir 'icvl_hdf5_audit.json'

Write-Output '=== step 1/5: authorization check (section 7) ==='
# 'hf auth whoami' writes its failure to stderr; suppress PowerShell's call-site
# noise so the operator sees only the real message. Only the exit code is used,
# and nothing derived from a token is ever captured or printed.
$who = ''
try { $who = (& hf auth whoami 2>$null | Out-String).Trim() } catch { $who = '' }
$whoCode = $LASTEXITCODE
if ($whoCode -ne 0) {
    Write-Output '  status: authorization_pending'
    Write-Output '  hf auth whoami failed. Output (no token is ever printed by this script):'
    Write-Output ('    ' + $who)
    Write-Output ''
    Write-Output '  The user must run:  hf auth login'
    Write-Output '  and must have accepted the gate at:'
    Write-Output ('    https://huggingface.co/datasets/' + $repoId)
    Write-Output '  STOPPING without downloading.'
    exit 3
}
Write-Output ('  authenticated as: ' + $who)

# ---- refuse to start if the target root is not writable ----
if (-not (Test-Path -LiteralPath $DataRoot)) {
    Write-Output ('  creating data root: ' + $DataRoot)
    [void][System.IO.Directory]::CreateDirectory($DataRoot)
}
if (-not [System.IO.Directory]::Exists($DataRoot)) {
    Write-Output ('  FATAL: data root not writable: ' + $DataRoot)
    exit 4
}

# ---- fresh space check (section 10) -----------------------------------------
# A Google Drive mount reports ACCOUNT QUOTA, not a fixed local disk, and it
# changes while other work runs on this machine. Never trust a cached probe:
# re-measure now, because the doc requires free > need + max(20 GiB, 10% of need).
$NEED = [Int64]28199177384
$TEN = [Int64][math]::Floor([double]$NEED * 0.10)
$TWENTY = [Int64]21474836480
$MARGIN = [Int64][math]::Max([double]$TEN, [double]$TWENTY)
$REQUIRED = [Int64]($NEED + $MARGIN)
$letter = $DataRoot.Substring(0, 1)
$free = (New-Object System.IO.DriveInfo($letter)).AvailableFreeSpace
Write-Output ''
Write-Output '=== step 1b/5: fresh space check (section 10) ==='
Write-Output ('  drive            : ' + $letter + ':')
Write-Output ('  need             : ' + $NEED + ' bytes')
Write-Output ('  margin           : ' + $MARGIN + ' bytes  (max of 20 GiB and 10% of need)')
Write-Output ('  required free    : ' + $REQUIRED + ' bytes')
Write-Output ('  available now    : ' + $free + ' bytes  (' + [math]::Round($free / 1GB, 2) + ' GiB)')
if ($free -lt $REQUIRED) {
    Write-Output '  status: space_stopped -- not enough free space. Nothing will be deleted.'
    Write-Output '  NOTE: G: and H: are the SAME volume (identical volume serial number), so'
    Write-Output '  switching letters does NOT gain space. Free space must be reclaimed on the'
    Write-Output '  Google Drive account itself, or the data root moved to a real local disk.'
    exit 6
}
Write-Output ('  projected free after download: approx ' + [math]::Round(($free - $NEED) / 1GB, 2) + ' GiB')

Write-Output ''
Write-Output '=== step 2/5: formal download (sections 11-13) ==='
$cmdText = 'hf download ' + $repoId + ' --repo-type dataset --revision "' + $Revision +
           '" --include "mat/*" --local-dir "' + $DataRoot + '" --max-workers 2'
$header = @(
    'ICVL acquisition -- formal download, task doc sections 11-13',
    ('started_at_local : ' + (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')),
    ('repo_id          : ' + $repoId),
    'repo_type        : dataset',
    ('revision         : ' + $Revision + '   (pinned full commit SHA)'),
    'include          : mat/*   (raw/* and preview/* are never requested)',
    ('local-dir        : ' + $DataRoot),
    'max-workers      : 2',
    ('HTTPS_PROXY      : ' + $env:HTTPS_PROXY),
    '',
    'COMMAND:',
    $cmdText,
    '',
    'STDOUT:'
)
($header -join "`r`n") | Set-Content -LiteralPath $stdoutFile -Encoding UTF8

# The exact section 11 command. Run it, then capture stderr separately so that
# resuming after an interruption is a matter of re-running this same script with
# the same revision, the same --include and the same target directory (section 13).
& hf download $repoId --repo-type dataset --revision $Revision `
    --include "mat/*" --local-dir $DataRoot --max-workers 2 1>> $stdoutFile 2> $stderrFile
$dlCode = $LASTEXITCODE
Add-Content -LiteralPath $stdoutFile -Encoding UTF8 ('[exit=' + $dlCode + ']')

# keep download_stderr.txt but strip PowerShell's call-site noise from it
if (Test-Path -LiteralPath $stderrFile) {
    $raw = @(Get-Content -LiteralPath $stderrFile -ErrorAction SilentlyContinue)
    $clean = @($raw | Where-Object {
        $_.Trim() -ne '' -and
        $_ -notmatch 'CategoryInfo' -and
        $_ -notmatch 'FullyQualifiedErrorId' -and
        $_ -notmatch '^\s*\+' -and
        $_ -notmatch '^\S+\.exe\s*:' -and
        $_ -notmatch 'run_download_and_audit'
    })
    # Write BOM-less: Set-Content -Encoding UTF8 emits a BOM, which would leave a
    # stray 3-byte prefix in an otherwise EMPTY stderr file and misreport it as
    # non-empty. An empty stderr is itself evidence worth keeping clean.
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($stderrFile, ($clean -join "`r`n"), $enc)
}

Write-Output ('  download exit code: ' + $dlCode)

Write-Output ''
Write-Output '=== step 3/5: completion check (sections 13-14) ==='
$matDir = Join-Path $DataRoot 'mat'
$local = @()
if (Test-Path -LiteralPath $matDir) {
    $local = @(Get-ChildItem -LiteralPath $matDir -Filter '*.mat' -File -ErrorAction SilentlyContinue)
}
$partial = @()
if (Test-Path -LiteralPath $DataRoot) {
    $partial = @(Get-ChildItem -LiteralPath $DataRoot -Recurse -File -Force -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -match '\.(part|incomplete|tmp|crdownload|partial)$' })
}
Write-Output ('  .mat files present : ' + $local.Count)
Write-Output ('  partial/temp files : ' + $partial.Count)
if ($partial.Count -gt 0) {
    Write-Output '  NOTE: partial files exist -> the download did not finish. Re-run this'
    Write-Output '        script with the same revision to resume (section 13).'
}

if ($dlCode -ne 0 -or $local.Count -eq 0) {
    Write-Output ''
    Write-Output '  Download did not complete cleanly. NOT running the audit as if it passed.'
    Write-Output '  If this was a network interruption: status = network_stopped.'
    Write-Output '  Resume with the SAME revision and the SAME target directory. Nothing was deleted.'
    exit 5
}

Write-Output ''
Write-Output '=== step 4/5: read-only audit (sections 14-18) ==='
# Refresh the pinned tree WITH credentials first. An anonymous capture silently
# returns every lfs.oid as 64 asterisks, which would make the manifest's
# remote-digest comparison vacuously true. fetch_pinned_tree.py refuses to write
# such a tree, so this either produces a trustworthy tree or fails loudly.
& $AuditPy -X utf8 (Join-Path $OutDir 'fetch_pinned_tree.py') `
    --out-dir $OutDir --revision $Revision > $null
$treeCode = $LASTEXITCODE
Write-Output ('  tree refresh exit code: ' + $treeCode)
if ($treeCode -ne 0) {
    Write-Output '  WARNING: the pinned tree refresh did not succeed. If it failed because'
    Write-Output '  digests were redacted, the digest comparison below cannot be trusted.'
}

& $AuditPy -X utf8 (Join-Path $OutDir 'audit_icvl_download.py') all `
    --remote-tree (Join-Path $OutDir 'remote_tree_pinned.json') `
    --data-root $DataRoot --out-dir $OutDir --revision $Revision
$auditCode = $LASTEXITCODE
Write-Output ('  audit exit code: ' + $auditCode)

Write-Output ''
Write-Output '=== step 5/5: verify SHA256 against the repository own LFS oids ==='
# While unauthenticated, huggingface.co redacts these oids on the wire, so this
# step can only succeed once a token is present. It refuses to claim verification
# otherwise (exit 3) rather than reporting a hollow pass.
& $AuditPy -X utf8 (Join-Path $OutDir 'verify_against_remote_oids.py') `
    --out-dir $OutDir --revision $Revision
$oidCode = $LASTEXITCODE
Write-Output ('  oid verification exit code: ' + $oidCode)

Write-Output ''
Write-Output '=== summary ==='
foreach ($p in @($manifest, $auditJson, (Join-Path $OutDir 'remote_oid_verification.json'))) {
    if (Test-Path -LiteralPath $p) {
        try {
            $j = Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json
            Write-Output ('  ' + (Split-Path $p -Leaf) + ' -> status=' + $j.status)
        } catch {
            Write-Output ('  ' + (Split-Path $p -Leaf) + ' -> present but unparsed')
        }
    } else {
        Write-Output ('  ' + (Split-Path $p -Leaf) + ' -> MISSING')
    }
}
exit $auditCode
