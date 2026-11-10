$ErrorActionPreference = 'Stop'
$modulePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$studyOutput = Join-Path $PSScriptRoot 'outputs/context_study_20261111'
if (-not (Test-Path -LiteralPath $modulePython)) { throw 'Run the existing setup scripts first.' }
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'outputs/validation/scifact_full/case_details.jsonl'))) { throw 'Run validate.cmd first to create the original full-corpus baseline and exact passage traces.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.model-cache/matplotlib'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & git cat-file -e ba8cdd6e7766306447913d8087b23f62195fa05a
    if ($LASTEXITCODE -ne 0) {
        & git fetch origin ba8cdd6e7766306447913d8087b23f62195fa05a
        if ($LASTEXITCODE -ne 0) { throw 'Pinned owner experiment is unavailable.' }
    }
    $stages = @(
        @{ Artifact = 'TRAIN_FROZEN.json'; Module = 'freeze_train' },
        @{ Artifact = 'GATE_TRAIN_FROZEN.json'; Module = 'train_gate' },
        @{ Artifact = 'summary.json'; Module = 'run_local' }
    )
    foreach ($stage in $stages) {
        if (Test-Path -LiteralPath (Join-Path $studyOutput $stage.Artifact)) {
            Write-Host "Retaining frozen result: $($stage.Artifact)"
        } else {
            if ($stage.Module -in @('train_gate','run_local')) {
                & $modulePython -m explanation_frontend.evaluation.context_study.provenance start
                if ($LASTEXITCODE -ne 0) { throw 'Provenance capture failed.' }
            }
            & $modulePython -m "explanation_frontend.evaluation.context_study.$($stage.Module)"
            if ($LASTEXITCODE -ne 0) { throw "Study stage failed: $($stage.Module)" }
        }
    }
    & $modulePython -m explanation_frontend.evaluation.context_study.provenance start
    if ($LASTEXITCODE -ne 0) { throw 'Provenance capture failed.' }
    & $modulePython -m explanation_frontend.evaluation.context_study.provenance audit
    if ($LASTEXITCODE -ne 0) { throw 'Split audit failed.' }
    & $modulePython -m pytest (Join-Path $PSScriptRoot 'tests') -q -o "cache_dir=$PSScriptRoot/.pytest_cache" --basetemp (Join-Path $PSScriptRoot '.tmp/context-study-tests') | Tee-Object -FilePath (Join-Path $studyOutput 'tests.log')
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
    foreach ($stage in @('evaluate_gate', 'report')) {
        & $modulePython -m "explanation_frontend.evaluation.context_study.$stage"
        if ($LASTEXITCODE -ne 0) { throw "Study stage failed: $stage" }
    }
    Write-Host 'Review evaluation/context_study/REPORT.md. Production defaults remain unchanged.'
} finally { Pop-Location }
