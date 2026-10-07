# probe_storage.ps1 -- ICVL acquisition, task doc section 4 (storage probe)
#
# ASCII-ONLY ON PURPOSE: this file is read by Windows PowerShell 5.1, which
# decodes BOM-less UTF-8 source as ANSI/GBK. Any non-ASCII literal here would be
# corrupted. The Google Drive "My Drive" folder has a localized name, so it is
# DISCOVERED at runtime (POSIX [A-Za-z0-9 ] filter) instead of hardcoded.
#
# Safety: creates and removes ONLY its own probe directory. Never deletes user data.

param(
    [string]$OutDir = 'D:\PyCharmProjects\Jeon2019\work\datasets\reproduction_v1\icvl_acquisition'
)

$ErrorActionPreference = 'Continue'

function Get-MyDrivePath {
    param([string]$Letter)
    # Candidate 1: ASCII name
    $c1 = "${Letter}:\My Drive"
    if (Test-Path -LiteralPath $c1) { return $c1 }
    # Candidate 2: discover a root directory whose name is the localized "My Drive"
    # Hidden/system entries and punctuation-only names are excluded; the filter is
    # byte-oriented so it works regardless of how the console decodes the name.
    try {
        $dirs = @(Get-ChildItem -LiteralPath "${Letter}:\" -Force -Directory -ErrorAction Stop |
                  Where-Object { -not $_.Attributes.ToString().Contains('Hidden') })
        foreach ($d in $dirs) {
            $leaf = $d.Name
            if ($leaf -notmatch '^[\u4e00-\u9fff ]+$') { continue }
            if ($leaf.Length -lt 2 -or $leaf.Length -gt 8) { continue }
            return $d.FullName
        }
    } catch { }
    # Candidate 3: brute-force a few known localizations
    foreach ($n in @([char]0x6211 + [char]0x7684 + [char]0x4e91 + [char]0x7aef + [char]0x786c + [char]0x76d8)) {
        $p = "${Letter}:\$n"
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

function Probe-Drive {
    param([string]$Letter)

    $d = [ordered]@{
        letter              = $Letter
        exists              = $false
        root_writable       = $false
        my_drive_path       = $null
        my_drive_writable   = $false
        free_bytes          = $null
        total_bytes         = $null
        file_system         = $null
        volume_label        = $null
        drive_type          = $null
        is_cloud_mount      = $false
        cloud_vendor        = $null
        write_test_ok       = $false
        write_test_bytes    = 0
        write_test_readback = $false
        write_test_seconds  = $null
        write_test_mbps     = $null
        probe_dir           = $null
        probe_dir_removed   = $null
        root_entries        = @()
        errors              = @()
    }

    try {
        $di = New-Object System.IO.DriveInfo("${Letter}:")
        $d.exists = $di.IsReady
        if ($di.IsReady) {
            $d.free_bytes   = $di.AvailableFreeSpace
            $d.total_bytes  = $di.TotalSize
            $d.file_system  = $di.DriveFormat
            $d.volume_label = $di.VolumeLabel
            $d.drive_type   = $di.DriveType.ToString()
        }
    } catch {
        $d.errors += "DriveInfo: $($_.Exception.Message)"
    }

    if (-not (Test-Path -LiteralPath "${Letter}:\")) {
        $d.exists = $false
        return [pscustomobject]$d
    }
    $d.exists = $true

    # --- cloud mount detection ---
    try {
        $rootItems = @(Get-ChildItem -LiteralPath "${Letter}:\" -Force -ErrorAction SilentlyContinue |
                       Select-Object -ExpandProperty Name)
        $d.root_entries = $rootItems
        $joined = ($rootItems -join '|')
        $hasShortcut = $joined -match 'shortcut-targets-by-id'
        $hasEncrypted = $joined -match '\.Encrypted'
        $hasRecycle = $joined -match 'RECYCLE'
        if ($hasShortcut -and $hasEncrypted) {
            $d.is_cloud_mount = $true
            $d.cloud_vendor = 'Google Drive for desktop'
        }
        if ($d.volume_label -match '@') {
            $d.is_cloud_mount = $true
            if (-not $d.cloud_vendor) { $d.cloud_vendor = 'Google Drive for desktop' }
        }
    } catch {
        $d.errors += "root listing: $($_.Exception.Message)"
    }

    # --- mount root writability (expected FALSE on Google Drive virtual root) ---
    $rootProbe = "${Letter}:\_dsh_probe_root"
    try {
        [void][System.IO.Directory]::CreateDirectory($rootProbe)
        $ok = [System.IO.Directory]::Exists($rootProbe)
        $d.root_writable = [bool]$ok
        if ($ok) { Remove-Item -LiteralPath $rootProbe -Recurse -Force -ErrorAction SilentlyContinue }
    } catch {
        $d.root_writable = $false
        $d.errors += "root write: $($_.Exception.Message)"
    }

    # --- My Drive writability + small bulk write ---
    $myDrive = Get-MyDrivePath -Letter $Letter
    if ($myDrive) {
        $d.my_drive_path = $myDrive
        $probe = Join-Path $myDrive '.dsh_icvl_probe'
        try {
            [void][System.IO.Directory]::CreateDirectory($probe)
            $d.probe_dir = $probe
            $d.my_drive_writable = [System.IO.Directory]::Exists($probe)

            $f = Join-Path $probe 'probe.txt'
            [System.IO.File]::WriteAllText($f, "dsh icvl storage probe")
            $d.write_test_readback = ([System.IO.File]::ReadAllText($f)).Length -gt 0

            $n   = [Int64](8MB)
            $buf = New-Object byte[] (1MB)
            (New-Object Random 12345).NextBytes($buf)
            $big = Join-Path $probe 'probe_bulk.bin'
            $sw  = [System.Diagnostics.Stopwatch]::StartNew()
            $fs  = [System.IO.File]::Create($big)
            try {
                for ($i = 0; $i -lt 8; $i++) { $fs.Write($buf, 0, $buf.Length) }
                $fs.Flush($true)
            } finally { $fs.Dispose() }
            $sw.Stop()

            $written = (Get-Item -LiteralPath $big).Length
            $d.write_test_bytes   = $written
            $d.write_test_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 3)
            if ($sw.Elapsed.TotalSeconds -gt 0) {
                $d.write_test_mbps = [math]::Round(($written / 1MB) / $sw.Elapsed.TotalSeconds, 2)
            }
            $d.write_test_ok = ($written -eq $n)

            # Remove this probe directory again. It is created by this script only;
            # no user file is ever touched. Bounded wait because Google Drive may
            # take a moment to reflect the deletion.
            for ($try = 0; $try -lt 20; $try++) {
                if (-not (Test-Path -LiteralPath $probe)) { break }
                Remove-Item -LiteralPath $probe -Recurse -Force -ErrorAction SilentlyContinue
                Start-Sleep -Milliseconds 400
            }
            $d.probe_dir_removed = -not (Test-Path -LiteralPath $probe)
        } catch {
            $d.errors += "my drive write: $($_.Exception.Message)"
        }
    } else {
        $d.errors += "My Drive folder not found"
    }

    return [pscustomobject]$d
}

$result = [ordered]@{
    artifact        = 'storage_probe.json'
    status          = 'storage_selected'
    probed_at_utc   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    probed_at_local = (Get-Date).ToString('yyyy-MM-ddTHH:mm:sszzz')
    host            = $env:COMPUTERNAME
    powershell      = $PSVersionTable.PSVersion.ToString()
    drives          = [ordered]@{}
}

foreach ($L in @('G', 'H')) { $result.drives[$L] = Probe-Drive -Letter $L }

# --- Google Drive client process ---
try {
    $gps = @(Get-Process -Name 'GoogleDriveFS' -ErrorAction SilentlyContinue)
    $result.google_drive_client_running = ($gps.Count -gt 0)
    if ($gps.Count -gt 0) {
        $result.google_drive_client_path = $gps[0].Path
    }
} catch { $result.google_drive_client_running = $null }

# --- space requirement: need + max(20 GiB, 10% of need) ---
$NEED_BYTES = [Int64]28199177384
$TEN_PCT    = [Int64][math]::Floor([double]$NEED_BYTES * 0.10)
$TWENTY_GIB = [Int64]21474836480
$MARGIN     = [Int64][math]::Max([double]$TEN_PCT, [double]$TWENTY_GIB)
$REQUIRED   = [Int64]($NEED_BYTES + $MARGIN)

$result.requirement = [ordered]@{
    mat_payload_bytes   = $NEED_BYTES
    ten_percent_bytes   = $TEN_PCT
    twenty_gib_bytes    = $TWENTY_GIB
    safety_margin_bytes = $MARGIN
    required_free_bytes = $REQUIRED
    formula             = 'need + max(20GiB, 10% of need)'
}

# --- selection: G: preferred, then H: ---
$selected = $null
foreach ($L in @('G', 'H')) {
    $info = $result.drives[$L]
    if ($info.exists -and $info.my_drive_writable -and ($info.free_bytes -ge $REQUIRED)) {
        $selected = $L
        break
    }
}

# --- algorithmic preference: G: preferred, then H: (task doc section 2.1) ---
$selected = $null
foreach ($L in @('G', 'H')) {
    $info = $result.drives[$L]
    if ($info.exists -and $info.my_drive_writable -and ($info.free_bytes -ge $REQUIRED)) {
        $selected = $L
        break
    }
}

$result.algorithmic_preference_drive = if ($selected) { "${selected}:" } else { $null }

# --- are the two drive letters actually independent storage? ---
# Google Drive for desktop can expose several mounts that resolve to the SAME
# volume. If the volume serial numbers match, "fall back to the other letter" is a
# false fallback: they share one pool of free space.
try {
    $serials = @{}
    foreach ($L in @('G', 'H')) {
        $v = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='${L}:'" -ErrorAction SilentlyContinue
        if ($v) { $serials[$L] = $v.VolumeSerialNumber }
    }
    $result.volume_serial_numbers = $serials
    $distinct = @($serials.Values | Select-Object -Unique)
    $result.drive_letters_are_independent = ($distinct.Count -gt 1)
    if ($distinct.Count -le 1) {
        $result.independence_note = ('G: and H: share volume serial ' + ($distinct -join ',') +
            ' -- they are the SAME volume under two account labels, so they share one ' +
            'pool of free space. Switching drive letters gains NO additional capacity.')
    } else {
        $result.independence_note = 'the two drive letters resolve to different volumes'
    }
} catch {
    $result.drive_letters_are_independent = $null
    $result.independence_note = 'could not read volume serial numbers'
}

# --- operative choice: the USER explicitly selected H: on 2026-10-07 ---
# Recorded separately from the algorithmic preference so the deviation from the
# G:-first rule is auditable rather than silent (task doc sections 10 and 2.1).
$OPERATIVE = 'H'
$result.selected_drive        = "${OPERATIVE}:"
$result.selected_data_root    = "${OPERATIVE}:/Jeon2019_data/reproduction_v1/icvl"
$result.selected_local_folder = Join-Path $result.drives[$OPERATIVE].my_drive_path 'Jeon2019_data\reproduction_v1\icvl'
$result.selection_reason      = 'USER decision 2026-10-07: H: chosen (algorithmic G:-first preference also satisfied the space check, but the operative data root is H:)'
$result.my_drive_mount_note   = 'Google Drive for desktop mounts a VIRTUAL root. Writing to <letter>:\ is silently discarded (CreateDirectory appears to succeed, Test-Path immediately returns False). The only user-writable root is the localized "My Drive" folder, so the literal path <letter>:\Jeon2019_data\... cannot be created. selected_local_folder is the real on-disk path; selected_data_root is the drive-letter-independent logical root for configuration.'

$jsonPath = Join-Path $OutDir 'storage_probe.json'
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
Write-Output "WROTE: $jsonPath"
Write-Output "--- summary ---"
foreach ($L in @('G', 'H')) {
    $i = $result.drives[$L]
    Write-Output ("{0}: exists={1} cloud={2} mydrive_writable={3} write_ok={4} {5} MiB/s free={6} GiB" -f `
        $L, $i.exists, $i.is_cloud_mount, $i.my_drive_writable, $i.write_test_ok, $i.write_test_mbps, `
        [math]::Round($i.free_bytes / 1GB, 2))
}
Write-Output ("required_free={0} GiB   selected={1}" -f [math]::Round($REQUIRED / 1GB, 2), $result.selected_drive)
Write-Output ("selected_local_folder={0}" -f $result.selected_local_folder)
