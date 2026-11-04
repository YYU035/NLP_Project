param([switch]$SkipModelDownload)
$ErrorActionPreference = 'Stop'
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $modulePython)) { throw 'Run setup.ps1 first.' }
$localTemp = Join-Path $PSScriptRoot '.tmp'
New-Item -ItemType Directory -Force $localTemp | Out-Null
$env:TEMP = $localTemp
$env:TMP = $localTemp
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.pip-cache'
$env:HF_HOME = Join-Path $PSScriptRoot '.model-cache/huggingface'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.model-cache/torch'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $modulePython -m pip install -r (Join-Path $PSScriptRoot 'requirements-local-llm.txt')
if ($LASTEXITCODE -ne 0) { throw 'Local open-source model dependencies could not be installed.' }
if (-not $SkipModelDownload) {
    Push-Location (Split-Path $PSScriptRoot -Parent)
    try {
        & $modulePython -m explanation_frontend.local_model_setup
        if ($LASTEXITCODE -ne 0) { throw 'Public model download failed.' }
    } finally { Pop-Location }
}
Write-Host 'Local explanation setup complete. Inference uses this computer and requires no API account or key.'
