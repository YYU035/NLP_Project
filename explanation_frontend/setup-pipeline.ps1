param([switch]$SkipDiscoverySetup, [switch]$SkipExtras)
$ErrorActionPreference = 'Stop'
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $modulePython)) { throw 'Run setup.ps1 first.' }
if (-not $SkipDiscoverySetup) {
    & (Join-Path $PSScriptRoot 'setup-discovery.ps1')
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.pip-cache'
$env:TEMP = Join-Path $PSScriptRoot '.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$env:HF_HOME = Join-Path $PSScriptRoot '.model-cache/huggingface'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.model-cache/torch'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if (-not $SkipExtras) {
        & $modulePython -m pip install -r (Join-Path $PSScriptRoot 'requirements-attribution.txt') -r (Join-Path $PSScriptRoot 'requirements-local-pdf.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Attribution/PDF dependency setup failed.' }
    }
    & $modulePython -m explanation_frontend.pipeline_setup
    if ($LASTEXITCODE -ne 0) { throw 'NLI setup failed.' }
    & $modulePython -m explanation_frontend.semantic_setup
    if ($LASTEXITCODE -ne 0) { throw 'Person 3 model setup failed.' }
    & $modulePython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
} finally { Pop-Location }
Write-Host 'Local pipeline is ready. Run run.ps1; use Find papers or Verify evidence > Live.'
