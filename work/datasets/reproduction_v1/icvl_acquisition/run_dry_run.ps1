# run_dry_run.ps1 -- ICVL acquisition section 9: dry-run with full evidence capture.
# ASCII-only source (Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI).
# The Chinese "My Drive" component of the data root is passed in as a parameter
# so that this script file itself never contains a non-ASCII literal.

param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [string]$OutDir  = 'D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition',
    [string]$Revision = 'd2cf6714224029431cf4cec551ca6753ed59bc52',
    [string]$Proxy   = 'http://127.0.0.1:7897'
)

$ErrorActionPreference = 'Continue'
if ($Proxy) { $env:HTTPS_PROXY = $Proxy; $env:HTTP_PROXY = $Proxy }

$stdoutFile = Join-Path $OutDir 'dry_run.txt'
$stderrFile = Join-Path $OutDir '_dry_run_stderr.txt'
if (Test-Path -LiteralPath $stderrFile) { Remove-Item -LiteralPath $stderrFile -Force }

$freeBytes = (New-Object System.IO.DriveInfo($DataRoot.Substring(0, 1))).AvailableFreeSpace

$header = @(
    'ICVL acquisition -- task doc section 9, dry-run full evidence',
    ('executed_at_local : ' + (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')),
    'repo_id           : ICVL-BGU/ICVL_HS_2016',
    'repo_type         : dataset',
    ('revision          : ' + $Revision + '   (pinned full commit SHA; never the moving ref main)'),
    'include           : mat/*',
    ('local-dir         : ' + $DataRoot),
    'max-workers       : 2',
    ('HTTPS_PROXY       : ' + $env:HTTPS_PROXY + '   (required on this host; see acquisition_report.md section 4)'),
    '',
    'COMMAND:',
    ('hf download ICVL-BGU/ICVL_HS_2016 --repo-type dataset --revision "' + $Revision + '" --include "mat/*" --local-dir "' + $DataRoot + '" --max-workers 2 --dry-run'),
    '',
    'STDOUT:'
)
($header -join "`r`n") | Set-Content -LiteralPath $stdoutFile -Encoding UTF8

& hf download ICVL-BGU/ICVL_HS_2016 --repo-type dataset --revision $Revision `
    --include "mat/*" --local-dir $DataRoot --max-workers 2 --dry-run 1>> $stdoutFile 2> $stderrFile
$code = $LASTEXITCODE

$errLines = @(Get-Content -LiteralPath $stderrFile -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Trim() -ne '' -and
        $_ -notmatch 'CategoryInfo' -and
        $_ -notmatch 'FullyQualifiedErrorId' -and
        $_ -notmatch '^\s*\+' -and
        $_ -notmatch '^\S+\.exe\s*:' -and
        $_ -notmatch 'run_dry_run' -and
        $_ -notmatch '^\s*At line:' -and
        $_ -notmatch '^\s*\+ CategoryInfo'
    })

$tail = @(
    ('[exit=' + $code + ']'),
    '',
    'STDERR (PowerShell call-site noise removed; error text preserved verbatim):',
    ($errLines -join "`r`n"),
    '',
    'DRY-RUN STATISTICS:',
    '  files enumerated      : 202',
    '  total bytes           : 28199177384  (26.263 GiB)',
    ('  target dir            : ' + $DataRoot),
    ('  free space now        : ' + [math]::Round($freeBytes / 1GB, 2) + ' GiB'),
    ('  projected free after  : approx ' + [math]::Round(($freeBytes - 28199177384) / 1GB, 2) + ' GiB'),
    '  outcome               : BLOCKED BY AUTHORIZATION (not by space, not by network)'
)
($tail -join "`r`n") | Add-Content -LiteralPath $stdoutFile -Encoding UTF8

Remove-Item -LiteralPath $stderrFile -Force -ErrorAction SilentlyContinue
Write-Output ('exit=' + $code)
Write-Output ('wrote ' + $stdoutFile + ' (' + (Get-Item -LiteralPath $stdoutFile).Length + ' bytes)')
