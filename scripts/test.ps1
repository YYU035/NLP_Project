$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
Require-ProjectPython
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Push-Location $ProjectRoot
try {
    & $ProjectPython -m pytest tests -q --basetemp (Join-Path $ProjectTemp 'pytest')
    if ($LASTEXITCODE -ne 0) { throw 'Software tests failed.' }
    & $ProjectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    & $ProjectPython scripts/check_integrity.py
    if ($LASTEXITCODE -ne 0) { throw 'Artifact integrity checks failed.' }
} finally { Pop-Location }
