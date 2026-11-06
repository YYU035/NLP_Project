param([switch]$SkipDiscoverySetup)
$ErrorActionPreference = 'Stop'
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $modulePython)) { throw 'Run setup.ps1 first.' }
if (-not $SkipDiscoverySetup) {
    & (Join-Path $PSScriptRoot 'setup-discovery.ps1')
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HOME = Join-Path $PSScriptRoot '.model-cache/huggingface'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.model-cache/torch'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & $modulePython -m explanation_frontend.pipeline_setup
    if ($LASTEXITCODE -ne 0) { throw 'NLI setup failed.' }
    & $modulePython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
} finally { Pop-Location }
Write-Host 'Local pipeline is ready. Run run.ps1; use Find papers or Verify evidence > Live.'
