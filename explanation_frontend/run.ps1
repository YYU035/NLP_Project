param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $modulePython)) {
    throw 'Run setup.ps1 with a Python 3.12 executable first; see README.md.'
}
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$env:HF_HOME = Join-Path $PSScriptRoot '.model-cache/huggingface'
$env:TORCH_HOME = Join-Path $PSScriptRoot '.model-cache/torch'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.model-cache/matplotlib'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
Push-Location $PSScriptRoot
try {
    & $modulePython -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Streamlit exited with an error.' }
} finally { Pop-Location }
