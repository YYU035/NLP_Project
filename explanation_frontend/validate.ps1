param([switch]$CleanSetup)
$ErrorActionPreference = 'Stop'
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $modulePython)) { throw 'Run setup.ps1 followed by setup-pipeline.ps1 first.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.model-cache/matplotlib'
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot '.tmp') -Force | Out-Null
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $arguments = @('-m', 'explanation_frontend.evaluation.validation.run')
    if ($CleanSetup) { $arguments += '--clean-setup' }
    & $modulePython @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Validation failed. Read outputs/validation logs; do not present this as a pass.' }
} finally { Pop-Location }
