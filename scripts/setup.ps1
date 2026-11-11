param([string]$Python = 'python', [switch]$SkipModels, [switch]$WithInterpretability)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$version = & $Python -c 'import sys; print(str(sys.version_info.major), str(sys.version_info.minor), sep=chr(46))'
if ($LASTEXITCODE -ne 0 -or $version -notin @('3.12', '3.13', '3.14')) {
    throw 'Use Python 3.12 (tested). Pass -Python with its executable path.'
}
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    & $Python -m venv (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
Push-Location $ProjectRoot
try {
    & $ProjectPython -m pip install -r (Join-Path $ProjectRoot 'requirements/runtime.lock')
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed.' }
    if ($WithInterpretability) {
        & $ProjectPython -m pip install -r (Join-Path $ProjectRoot 'requirements/interpretability.lock')
        if ($LASTEXITCODE -ne 0) { throw 'Interpretability dependency installation failed.' }
    }
    & $ProjectPython -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw 'Local project installation failed.' }
    & $ProjectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    if (-not $SkipModels) {
        & (Join-Path $PSScriptRoot 'setup_models.ps1')
    }
} finally { Pop-Location }
Write-Host 'Setup complete. Run run.cmd. Model setup is explicit; requests do not download weights.'
