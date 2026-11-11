param([switch]$IncludeLanguageModel)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
Require-ProjectPython
Push-Location $ProjectRoot
try {
    & $ProjectPython -m paper_search.model_setup --model-path (Join-Path $ProjectData 'models/retrieval_encoder')
    if ($LASTEXITCODE -ne 0) { throw 'Discovery/retrieval model setup failed.' }
    & $ProjectPython -m research_verifier.runtime.setup_verifier
    if ($LASTEXITCODE -ne 0) { throw 'Verifier model setup failed.' }
    & $ProjectPython -m research_verifier.runtime.setup_evidence
    if ($LASTEXITCODE -ne 0) { throw 'Evidence encoder setup failed.' }
    if ($IncludeLanguageModel) {
        & $ProjectPython -m research_verifier.runtime.setup_language_model
        if ($LASTEXITCODE -ne 0) { throw 'Optional local language model setup failed.' }
    }
} finally { Pop-Location }
