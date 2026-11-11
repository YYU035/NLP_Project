param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
Require-ProjectPython
Push-Location $ProjectRoot
try {
    & $ProjectPython -m streamlit run (Join-Path $ProjectRoot 'src/research_verifier/ui/app.py') --server.address 127.0.0.1 --server.port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Application exited with an error.' }
} finally { Pop-Location }
