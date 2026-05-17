param(
  [ValidateSet("TarGzip", "Zip")]
  [string]$Format = "TarGzip"
)

$ErrorActionPreference = "Stop"
$base = "F:\dataset\movie"

$dirs = @(
  "$base\Annotation_Shot_Desc_11.15_V2_160movies",
  "$base\Character_Bank_All_11.16"
) | Where-Object { Test-Path -LiteralPath $_ }

foreach ($d in $dirs) { Write-Host "OK dir: $d" }
if ($dirs.Count -eq 0) { throw "No target directories found." }

$dirRoots = $dirs | ForEach-Object { (Resolve-Path -LiteralPath $_).Path.TrimEnd('\') + '\' }

$extraJson = Get-ChildItem -LiteralPath $base -Filter *.json -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object {
    $p = $_.FullName
    -not ($dirRoots | Where-Object { $p.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) })
  }

Write-Host "Extra JSON (outside the two dirs): $($extraJson.Count)"

if ($Format -eq "TarGzip") {
  $dest = "F:\dataset\movie\moviebench_pack.tar.gz"
  if (Test-Path -LiteralPath $dest) { Remove-Item -LiteralPath $dest -Force }
  Push-Location -LiteralPath $base
  try {
    $rel = @()
    foreach ($d in $dirs) { $rel += (Resolve-Path -LiteralPath $d -Relative) }
    foreach ($j in $extraJson) { $rel += (Resolve-Path -LiteralPath $j.FullName -Relative) }
    Write-Host "Creating $dest ..."
    $destAbs = (Resolve-Path -LiteralPath (Split-Path -Parent $dest)).Path + "\" + (Split-Path -Leaf $dest)
    & tar.exe -acf $destAbs @rel
    if ($LASTEXITCODE -ne 0) { throw "tar.exe exited with $LASTEXITCODE" }
  } finally {
    Pop-Location
  }
} else {
  $dest = "F:\dataset\movie\moviebench_pack.zip"
  $all = @($dirs) + @($extraJson.FullName)
  Write-Host "Creating $dest (Compress-Archive, may be slow) ..."
  Compress-Archive -LiteralPath $all -DestinationPath $dest -CompressionLevel Optimal -Force
}

Get-Item -LiteralPath $dest | Select-Object FullName, Length, LastWriteTime
