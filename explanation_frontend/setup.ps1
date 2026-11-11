param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$version = & $Python -c 'import sys; print(str(sys.version_info.major), str(sys.version_info.minor), sep=chr(46))'
if ($LASTEXITCODE -ne 0 -or $version -notin @('3.11', '3.12', '3.13', '3.14')) {
    throw 'Use Python 3.12 (tested), or 3.11–3.14. Pass -Python with the full executable path.'
}
$tempPath = Join-Path $PSScriptRoot '.tmp'
New-Item -ItemType Directory -Force $tempPath | Out-Null
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.pip-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $Python -m venv (Join-Path $PSScriptRoot '.venv')
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $modulePython -m pip install -r (Join-Path $PSScriptRoot 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Ready. Run explanation_frontend/run.ps1 from the repository root.'
