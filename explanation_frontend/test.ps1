$ErrorActionPreference = 'Stop'
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $modulePython)) { throw 'Run setup.ps1 first.' }
$moduleTemp = Join-Path $PSScriptRoot '.tmp'
New-Item -ItemType Directory -Path $moduleTemp -Force | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.model-cache/matplotlib'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & $modulePython -m pytest (Join-Path $PSScriptRoot 'tests') -q -o "cache_dir=$PSScriptRoot/.pytest_cache" --basetemp (Join-Path $moduleTemp 'test-run')
    if ($LASTEXITCODE -ne 0) { throw 'Frontend tests failed; inspect the output above.' }
    & $modulePython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
} finally { Pop-Location }
