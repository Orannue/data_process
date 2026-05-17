# moviebench：默认只复制文件夹名形如0001_xxx 且前缀数字在 MoviebenchIdMin..MoviebenchIdMax内的子目录。
# -MoviebenchAll：复制 MoviebenchSource 下所有一级子文件夹（含 0049_、1001_ 等）。
# -FullDataset：另外复制 Annotation / Character / 其余 JSON。
# 请一次跑完；中断会导致 G: 上只有前几个目录。日志：DestRoot\_robocopy_materials.log

param(
  [string]$DestRoot = 'G:\movie_dataset_copy',
  [switch]$FullDataset,
  [string]$MoviebenchSource = 'F:\dataset\movie\moviebench',
  [switch]$MoviebenchAll,
  [int]$MoviebenchIdMin = 1,
  [int]$MoviebenchIdMax = 46
)

$ErrorActionPreference = "Stop"

$base = 'F:\dataset\movie'

if ($MoviebenchAll -and ($MoviebenchIdMin -ne 1 -or $MoviebenchIdMax -ne 46)) {
  Write-Host "Note: -MoviebenchAll set; ID range parameters are ignored."
}

if ([string]::IsNullOrWhiteSpace($MoviebenchSource)) {
  throw "MoviebenchSource is empty. Example: -MoviebenchSource 'F:\dataset\movie\moviebench'"
}

if ([string]::IsNullOrWhiteSpace($DestRoot)) {
  throw "DestRoot is empty."
}

if (-not (Test-Path -LiteralPath $DestRoot)) {
  New-Item -ItemType Directory -Path $DestRoot -Force | Out-Null
}

$robocopyLog = Join-Path $DestRoot "_robocopy_materials.log"
$rcArgs = @(
  "/E", "/COPY:DAT", "/DCOPY:T",
  "/R:10", "/W:30", "/Z",
  "/MT:4",
  "/NFL", "/NDL", "/NP",
  "/LOG+:$robocopyLog"
)

$mbSubs = @()
if (Test-Path -LiteralPath $MoviebenchSource) {
  $allDirs = @(Get-ChildItem -LiteralPath $MoviebenchSource -Directory -ErrorAction SilentlyContinue)
  if ($MoviebenchAll) {
    $mbSubs = $allDirs | Sort-Object Name
    Write-Host "Source: $MoviebenchSource"
    Write-Host "moviebench: ALL subdirs ($($mbSubs.Count) folders)"
  } else {
    $mbSubs = @($allDirs | Where-Object {
        if ($_.Name -match '^(\d{4})_') {
          $n = [int]$Matches[1]
          $n -ge $MoviebenchIdMin -and $n -le $MoviebenchIdMax
        } else { $false }
      } | Sort-Object Name)
    Write-Host "Source: $MoviebenchSource"
    Write-Host "moviebench: prefix $($MoviebenchIdMin.ToString('0000'))-$($MoviebenchIdMax.ToString('0000')) -> $($mbSubs.Count) folders"
  }
  $mbSubs | ForEach-Object { Write-Host "  $($_.Name)" }
} else {
  Write-Host "moviebench not found: $MoviebenchSource"
}

$topDirs = @()
if ($FullDataset) {
  $topDirs = @(
    "$base\Annotation_Shot_Desc_11.15_V2_160movies",
    "$base\Character_Bank_All_11.16"
  ) | Where-Object { Test-Path -LiteralPath $_ }
  foreach ($d in $topDirs) { Write-Host "OK dir (FullDataset): $d" }
}

$skipJsonRoots = @()
foreach ($d in $topDirs) {
  $skipJsonRoots += (Resolve-Path -LiteralPath $d).Path.TrimEnd('\') + '\'
}
foreach ($sub in $mbSubs) {
  $skipJsonRoots += (Resolve-Path -LiteralPath $sub.FullName).Path.TrimEnd('\') + '\'
}

$extraJson = @()
if ($FullDataset) {
  $extraJson = @(Get-ChildItem -LiteralPath $base -Filter *.json -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object {
        $p = $_.FullName
        -not ($skipJsonRoots | Where-Object { $p.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) })
      })
}

Write-Host "Dest: $DestRoot"
Write-Host "Robocopy log (append): $robocopyLog"
if ($FullDataset) {
  Write-Host "Extra JSON (outside copied dirs / selected moviebench): $($extraJson.Count)"
}

if (-not $FullDataset) {
  if ($mbSubs.Count -eq 0) {
    throw "Nothing to copy: no moviebench subdirs under $MoviebenchSource (adjust -MoviebenchAll or ID range)."
  }
} else {
  if ($mbSubs.Count -eq 0 -and $topDirs.Count -eq 0 -and $extraJson.Count -eq 0) {
    throw "Nothing to copy (-FullDataset): no moviebench subdirs, no annotation/character dirs, and no JSON under $base outside those trees."
  }
}

if ($FullDataset) {
  foreach ($d in $topDirs) {
    $name = Split-Path -Leaf $d
    $target = Join-Path $DestRoot $name
    Write-Host "Robocopy (COPY) -> $target"
    robocopy $d $target @rcArgs
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit $LASTEXITCODE for $d (see log)" }
  }
}

$mbOut = Join-Path $DestRoot "moviebench"
$mbTotal = $mbSubs.Count
$mbIdx = 0
foreach ($sub in $mbSubs) {
  $mbIdx++
  $target = Join-Path $mbOut $sub.Name
  Write-Host "----- moviebench [$mbIdx / $mbTotal] $($sub.Name) -----"
  robocopy $sub.FullName $target @rcArgs
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit $LASTEXITCODE for $($sub.FullName) (see log)" }
}

if ($FullDataset) {
  Push-Location -LiteralPath $base
  try {
    $n = 0
    foreach ($j in $extraJson) {
      $rel = Resolve-Path -LiteralPath $j.FullName -Relative
      $rel = $rel -replace '^\.\\', ''
      $out = Join-Path $DestRoot $rel
      $parent = Split-Path -Parent $out
      if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
      }
      Copy-Item -LiteralPath $j.FullName -Destination $out -Force
      $n++
      if (($n % 500) -eq 0) { Write-Host "Copied $n extra JSON files..." }
    }
  } finally {
    Pop-Location
  }
  Write-Host "Done. Extra JSON copied: $($extraJson.Count)"
}

Write-Host "All moviebench robocopy steps finished ($mbTotal folders)."
Get-ChildItem -LiteralPath $DestRoot -ErrorAction SilentlyContinue | Select-Object Name, Mode
