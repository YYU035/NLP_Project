# Shared process-local paths. Dot-source from the other project scripts.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$ProjectTemp = Join-Path $ProjectRoot '.cache/tmp'
if ($env:RESEARCH_VERIFIER_DATA_DIR) {
    $ProjectData = [IO.Path]::GetFullPath($env:RESEARCH_VERIFIER_DATA_DIR)
} else {
    $ProjectData = Join-Path $ProjectRoot 'data'
}
New-Item -ItemType Directory -Force $ProjectTemp | Out-Null
$env:RESEARCH_VERIFIER_ROOT = $ProjectRoot
$env:RESEARCH_VERIFIER_DATA_DIR = $ProjectData
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = $ProjectTemp
$env:TMP = $ProjectTemp
if (-not $env:PIP_CACHE_DIR) {
    $env:PIP_CACHE_DIR = Join-Path $ProjectRoot '.cache/pip'
}
$env:HF_HOME = Join-Path $ProjectRoot '.cache/huggingface'
$env:TORCH_HOME = Join-Path $ProjectRoot '.cache/torch'
$env:MPLCONFIGDIR = Join-Path $ProjectRoot '.cache/matplotlib'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'

function Require-ProjectPython {
    if (-not (Test-Path -LiteralPath $ProjectPython)) {
        throw 'Run setup.cmd first to create the project environment.'
    }
}
