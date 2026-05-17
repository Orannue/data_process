$ErrorActionPreference = "Stop"
$base = "F:\dataset\movie"
$7z = "C:\Program Files\7-Zip\7z.exe"
if (-not (Test-Path -LiteralPath $7z)) {
  throw "7-Zip not found: $7z"
}

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
$dest = "F:\dataset\movie\moviebench_pack_7z.7z"

$args = @("a", "-t7z", "-m0=LZMA2", "-mx=5", "-mmt=on", "-bb1", $dest) + @($dirs) + @($extraJson.FullName)
& $7z @args

Get-Item -LiteralPath $dest | Select-Object FullName, Length, LastWriteTime
