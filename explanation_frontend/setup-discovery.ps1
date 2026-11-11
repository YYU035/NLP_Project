param([switch]$SkipModels)
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
# Per-command Git configuration avoids Windows long-path checkout failures;
# it does not change the user's global Git settings.
$previousCount = $env:GIT_CONFIG_COUNT
$previousKey = $env:GIT_CONFIG_KEY_0
$previousValue = $env:GIT_CONFIG_VALUE_0
try {
    if ($env:GIT_CONFIG_COUNT) { throw 'Run setup from a terminal without GIT_CONFIG_COUNT overrides.' }
    $env:GIT_CONFIG_COUNT = '1'
    $env:GIT_CONFIG_KEY_0 = 'core.longpaths'
    $env:GIT_CONFIG_VALUE_0 = 'true'
    & $modulePython -m pip install -r (Join-Path $PSScriptRoot 'requirements-discovery-lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Optional discovery dependency installation failed.' }
} finally {
    $env:GIT_CONFIG_COUNT = $previousCount
    $env:GIT_CONFIG_KEY_0 = $previousKey
    $env:GIT_CONFIG_VALUE_0 = $previousValue
}
if (-not $SkipModels) {
    & $modulePython -m paper_search.model_setup --model-path (Join-Path $PSScriptRoot '.models/all-MiniLM-L6-v2')
    if ($LASTEXITCODE -ne 0) { throw 'Pinned model setup failed.' }
}
& $modulePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
Write-Host 'Discovery is ready. Launch run.ps1 and choose Find papers > Live search.'
