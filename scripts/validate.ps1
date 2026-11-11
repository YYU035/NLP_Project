param([switch]$WithModels, [switch]$Live)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
Require-ProjectPython
& (Join-Path $PSScriptRoot 'test.ps1')
if ($Live -and -not $WithModels) { throw '-Live requires -WithModels.' }
if ($WithModels) {
    Push-Location $ProjectRoot
    try {
        & $ProjectPython -m research_verifier.runtime.smoke
        if ($LASTEXITCODE -ne 0) { throw 'Offline real-model smoke failed.' }
        if ($Live) {
            # Fresh discovery calls public catalogues; all model weights stay local.
            & $ProjectPython -m research_verifier.runtime.smoke --live --limit 3
            if ($LASTEXITCODE -ne 0) { throw 'Live public-catalogue smoke failed.' }
        }
    } finally { Pop-Location }
}
