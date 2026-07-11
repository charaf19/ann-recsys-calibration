param(
    # Remove previous generated results, embeddings, and indexes.
    # Raw datasets and normalized dataset CSVs are preserved.
    [switch]$Fresh,

    # Resume an interrupted/failed run WITHOUT deleting anything. The main grid
    # runs with --write_mode replace --reuse_existing (existing compatible
    # embeddings/indexes are reused after metadata verification; incompatible
    # ones are rebuilt), then downstream analyses continue. Mutually exclusive
    # with -Fresh. Skips dependency install and dataset re-preparation.
    [switch]$Resume,

    # Recreate normalized dataset CSVs even when they already exist.
    [switch]$ReprepareDatasets,

    # Include the optional PyTorch two-tower embedding sensitivity experiment.
    [switch]$IncludeTwoTower,

    # Skip dependency installation when the environment is already ready.
    [switch]$SkipInstall,

    # Skip compileall and pytest.
    [switch]$SkipTests,

    # DEVELOPER/DEBUG ONLY. Skip the synthetic scale-stress experiment. Scale
    # stress is currently CRITICAL paper evidence, so a run with this switch
    # CANNOT produce a validator-complete paper result. Use only for fast
    # iteration; never for the final paper run.
    [switch]$SkipScaleStress,

    # Override the execution log path. Defaults to
    # logs\full_experiment_<timestamp>.log.
    [string]$LogPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Fresh -and $Resume) {
    throw "-Fresh and -Resume are mutually exclusive: -Fresh rebuilds from " +
          "scratch, -Resume preserves and continues. Choose one."
}

# A resumed run must never re-prepare datasets, even if -ReprepareDatasets was
# also passed by habit.
if ($Resume -and $ReprepareDatasets) {
    Write-Warning "-Resume ignores -ReprepareDatasets (datasets are preserved)."
    $ReprepareDatasets = [switch]$false
}

# =====================================================================
# Repository setup
# =====================================================================

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python -PathType Leaf)) {
    throw @"
Python virtual environment was not found:

    $Python

Create it with:

    py -3.10 -m venv .venv
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
"@
}

# Deterministic CPU execution.
$env:PYTHONHASHSEED = "42"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

# Prevent an old environment override from forcing an online Amazon download.
Remove-Item Env:AMAZON_BOOKS_URL -ErrorAction SilentlyContinue

# =====================================================================
# Logging
# =====================================================================

New-Item -ItemType Directory -Path "logs" -Force | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $LogPath = Join-Path $RepoRoot "logs\full_experiment_$Timestamp.log"
}
$LogDirectory = Split-Path -Parent $LogPath
if (-not [string]::IsNullOrWhiteSpace($LogDirectory)) {
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
}

# Tracks the stage currently executing so the failure handler can report it.
$script:CurrentStage = "(startup)"

# Force unbuffered Python child output so progress and tracebacks stream live
# into the transcript instead of arriving in one block at the end.
$env:PYTHONUNBUFFERED = "1"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $script:CurrentStage = $Name
    $commandLine = "$Python $($Arguments -join ' ')"

    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("STAGE   : {0}" -f $Name) -ForegroundColor Cyan
    Write-Host ("STARTED : {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) `
        -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
    # Print the EXACT command before every stage (recorded in the log).
    Write-Host "COMMAND : $commandLine" -ForegroundColor Cyan

    # Run Python with stderr MERGED into stdout (2>&1) so warnings and full
    # tracebacks are captured in execution order. Each line is re-emitted via
    # Write-Host, which Start-Transcript records to the log — guaranteeing the
    # log holds detailed output even for native stderr (which the transcript
    # does not always capture on its own in PowerShell 5.1).
    #
    # ErrorActionPreference is relaxed to Continue only around the call so that
    # Python writing to stderr (warnings, progress) cannot raise a spurious
    # terminating NativeCommandError under Set-StrictMode/Stop. $LASTEXITCODE
    # stays the real Python exit code: the native command sets it regardless of
    # the downstream pipeline, so no Tee-Object/pipeline masks the exit code.
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments 2>&1 | ForEach-Object { Write-Host $_.ToString() }
        $exit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousEap
    }
    if ($null -eq $exit) { $exit = 0 }

    Write-Host ("EXIT    : {0}" -f $exit) -ForegroundColor DarkGray

    if ($exit -ne 0) {
        throw "$Name failed with exit code $exit (see log: $LogPath)."
    }

    Write-Host "$Name completed successfully." -ForegroundColor Green
}

function Assert-FileExists {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Required file does not exist: $Path"
    }
}

function Prepare-DatasetWhenNeeded {
    param(
        [Parameter(Mandatory)]
        [string]$Dataset,

        [Parameter(Mandatory)]
        [string]$Output
    )

    if ((Test-Path $Output -PathType Leaf) -and (-not $ReprepareDatasets)) {
        Write-Host ""
        Write-Host "Reusing prepared dataset: $Output" -ForegroundColor Yellow
        return
    }

    if ($ReprepareDatasets -and (Test-Path $Output -PathType Leaf)) {
        Write-Host "Removing old normalized dataset: $Output" `
            -ForegroundColor Yellow
        Remove-Item $Output -Force
    }

    Invoke-PythonStep `
        -Name "Prepare dataset: $Dataset" `
        -Arguments @(
            "src\prepare_dataset.py",
            "--dataset", $Dataset,
            "--out", $Output
        )

    Assert-FileExists -Path $Output
}

function Resolve-LocalAmazonReviewArchive {
    $RawDirectory = Join-Path $RepoRoot "data\raw"
    $ExpectedPath = Join-Path $RawDirectory "amazon_books_5.json.gz"

    New-Item -ItemType Directory -Path $RawDirectory -Force | Out-Null

    if (Test-Path $ExpectedPath -PathType Leaf) {
        Write-Host ""
        Write-Host "Amazon review archive already uses the expected path:" `
            -ForegroundColor Green
        Write-Host "  $ExpectedPath"
        return $ExpectedPath
    }

    # Common filenames used for the Amazon Books 5-core review archive.
    $CandidateNames = @(
        "reviews_Books_5.json.gz",
        "reviews_books_5.json.gz",
        "Books_5.json.gz",
        "books_5.json.gz",
        "reviews_Books.json.gz",
        "amazon_books.json.gz",
        "amazon-books-5-core.json.gz"
    )

    foreach ($CandidateName in $CandidateNames) {
        $CandidatePath = Join-Path $RawDirectory $CandidateName

        if (Test-Path $CandidatePath -PathType Leaf) {
            Write-Host ""
            Write-Host "Found local Amazon Books review archive:" `
                -ForegroundColor Green
            Write-Host "  $CandidatePath"

            Copy-Item `
                -Path $CandidatePath `
                -Destination $ExpectedPath `
                -Force

            Write-Host "Copied to the path expected by the repository:" `
                -ForegroundColor Green
            Write-Host "  $ExpectedPath"

            return $ExpectedPath
        }
    }

    # Recursive fallback. Metadata archives are intentionally excluded.
    $DetectedFiles = @(
        Get-ChildItem `
            -Path $RawDirectory `
            -Recurse `
            -File `
            -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match "(?i)books" -and
            $_.Name -match "(?i)review" -and
            $_.Name -match "\.json\.gz$" -and
            $_.Name -notmatch "(?i)^meta[_-]"
        } |
        Sort-Object Length -Descending
    )

    if ($DetectedFiles.Count -eq 0) {
        throw @"
No compatible Amazon Books review archive was found under:

    $RawDirectory

The current pipeline requires the review JSONL archive, normally named:

    reviews_Books_5.json.gz

A file such as meta_Books.json.gz is product metadata and cannot replace
the review interaction archive.

Place the review archive inside data\raw and run the script again.
"@
    }

    $SelectedFile = $DetectedFiles[0]

    Write-Host ""
    Write-Host "Automatically selected Amazon review archive:" `
        -ForegroundColor Yellow
    Write-Host "  $($SelectedFile.FullName)"
    Write-Host "  Size: $([math]::Round($SelectedFile.Length / 1MB, 2)) MB"

    Copy-Item `
        -Path $SelectedFile.FullName `
        -Destination $ExpectedPath `
        -Force

    Write-Host "Copied to:" -ForegroundColor Green
    Write-Host "  $ExpectedPath"

    return $ExpectedPath
}

function Assert-MinimumCsvRows {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [int]$MinimumRows
    )

    Assert-FileExists -Path $Path

    $Rows = @(Import-Csv $Path).Count

    if ($Rows -lt $MinimumRows) {
        throw "$Path contains $Rows rows; expected at least $MinimumRows."
    }

    Write-Host ("{0}: {1} rows" -f $Path, $Rows) -ForegroundColor Green
}

# -Append so repeated resume runs to the same -LogPath accumulate a full
# debugging history instead of overwriting it (a default timestamped path is
# unique per run, so nothing is lost there either).
Start-Transcript -Path $LogPath -Append | Out-Null

Write-Host ""
Write-Host ("Execution log (live + captured): {0}" -f $LogPath) `
    -ForegroundColor Green
$InvokedFlags = @()
if ($Fresh) { $InvokedFlags += "-Fresh" }
if ($Resume) { $InvokedFlags += "-Resume" }
if ($ReprepareDatasets) { $InvokedFlags += "-ReprepareDatasets" }
if ($IncludeTwoTower) { $InvokedFlags += "-IncludeTwoTower" }
if ($SkipInstall) { $InvokedFlags += "-SkipInstall" }
if ($SkipTests) { $InvokedFlags += "-SkipTests" }
if ($SkipScaleStress) { $InvokedFlags += "-SkipScaleStress" }
Write-Host ("Switches: {0}" -f ($(if ($InvokedFlags.Count) { $InvokedFlags -join ' ' } else { '(none)' })))
Write-Host ("Started : {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

try {
    # =================================================================
    # Optional cleanup
    # =================================================================

    if ($Resume) {
        Write-Host ""
        Write-Host "RESUME MODE: no results, embeddings, indexes, or datasets " `
            -ForegroundColor Yellow -NoNewline
        Write-Host "will be deleted." -ForegroundColor Yellow
        Write-Host "  Main grid will run with --write_mode replace " `
            "--reuse_existing." -ForegroundColor Yellow
        Write-Host "  Dependency install and dataset re-preparation are " `
            "skipped." -ForegroundColor Yellow
    }

    if ($Fresh) {
        Write-Host ""
        Write-Host "Cleaning generated results, embeddings, and indexes..." `
            -ForegroundColor Yellow

        # Delete generated result files while preserving .gitkeep files.
        if (Test-Path "results") {
            Get-ChildItem "results" -Recurse -File |
                Where-Object { $_.Name -ne ".gitkeep" } |
                Remove-Item -Force
        }

        # Delete generated embeddings and indexes.
        # Do not delete data\raw or normalized CSVs.
        if (Test-Path "data") {
            Get-ChildItem "data" -Directory |
                Where-Object {
                    $_.Name -like "emb_*" -or
                    $_.Name -like "index_*"
                } |
                Remove-Item -Recurse -Force
        }

        Write-Host "Generated artifacts cleaned." -ForegroundColor Green
        Write-Host "Raw datasets and normalized CSVs were preserved." `
            -ForegroundColor Green
    }

    # =================================================================
    # Dependency installation
    # =================================================================

    # On a resumed run, never reinstall dependencies unless the caller
    # explicitly did not ask to skip AND is not resuming.
    if ((-not $SkipInstall) -and (-not $Resume)) {
        Invoke-PythonStep `
            -Name "Upgrade pip" `
            -Arguments @(
                "-m", "pip", "install", "--upgrade", "pip"
            )

        Invoke-PythonStep `
            -Name "Install canonical CPU requirements" `
            -Arguments @(
                "-m", "pip", "install",
                "-r", "requirements-cpu.txt"
            )

        # pytest is used by the repository tests but is not part of the
        # scientific runtime requirements file.
        Invoke-PythonStep `
            -Name "Install test dependency" `
            -Arguments @(
                "-m", "pip", "install", "pytest"
            )

        if ($IncludeTwoTower) {
            Invoke-PythonStep `
                -Name "Install optional dependencies" `
                -Arguments @(
                    "-m", "pip", "install",
                    "-r", "requirements-optional.txt"
                )
        }
    }

    # =================================================================
    # Lightweight verification
    # =================================================================

    if (-not $SkipTests) {
        Invoke-PythonStep `
            -Name "Compile source and test files" `
            -Arguments @(
                "-m", "compileall", "src", "tests"
            )

        Invoke-PythonStep `
            -Name "Run regression tests" `
            -Arguments @(
                "-m", "pytest", "tests", "-q"
            )
    }

    # =================================================================
    # Hardware provenance
    # =================================================================

    Invoke-PythonStep `
        -Name "Capture CPU hardware and Python environment" `
        -Arguments @(
            "src\capture_hardware.py"
        )

    # =================================================================
    # Resolve the existing local Amazon Books archive
    # =================================================================

    $AmazonReviewArchive = Resolve-LocalAmazonReviewArchive
    Assert-FileExists -Path $AmazonReviewArchive

    Write-Host ""
    Write-Host "Amazon Books will be prepared from the local archive:" `
        -ForegroundColor Green
    Write-Host "  $AmazonReviewArchive"
    Write-Host "No Amazon Books download will be attempted." `
        -ForegroundColor Green

    # =================================================================
    # Dataset preparation
    # =================================================================

    Prepare-DatasetWhenNeeded `
        -Dataset "ml-1m" `
        -Output "data\ml1m.csv"

    Prepare-DatasetWhenNeeded `
        -Dataset "ml-20m" `
        -Output "data\ml20m.csv"

    Prepare-DatasetWhenNeeded `
        -Dataset "goodbooks" `
        -Output "data\goodbooks.csv"

    Prepare-DatasetWhenNeeded `
        -Dataset "amazon-books" `
        -Output "data\amazon_books.csv"

    $DatasetFiles = @(
        "data\ml1m.csv",
        "data\ml20m.csv",
        "data\goodbooks.csv",
        "data\amazon_books.csv"
    )

    foreach ($DatasetFile in $DatasetFiles) {
        Assert-FileExists -Path $DatasetFile
    }

    # =================================================================
    # Dataset statistics
    # =================================================================

    Invoke-PythonStep `
        -Name "Generate dataset statistics" `
        -Arguments @(
            "src\dataset_stats.py",
            "--datasets",
            "ml-1m:data/ml1m.csv",
            "ml-20m:data/ml20m.csv",
            "goodbooks:data/goodbooks.csv",
            "amazon-books:data/amazon_books.csv",
            "--min_user_interactions", "5",
            "--out_dir", "results\paper\tables"
        )

    # =================================================================
    # Main four-dataset experiment
    #
    # 4 datasets × 2 modalities × 5 methods = 40 rows
    # =================================================================

    $MainArguments = @(
        "src\run_revision_experiments.py",
        "--config", "configs\main_cpu.yml"
    )
    if ($Resume) {
        # Reuse compatible embeddings/indexes (metadata-verified) and overwrite
        # any partial results; incompatible artifacts are rebuilt honestly.
        $MainArguments += @("--write_mode", "replace", "--reuse_existing")
    }
    else {
        $MainArguments += @("--write_mode", "fail_if_exists")
    }

    Invoke-PythonStep `
        -Name "Run complete main ANN recommendation experiment" `
        -Arguments $MainArguments

    Assert-FileExists -Path "results\main\summary_main.csv"
    Assert-FileExists -Path "results\main\run_config.json"
    Assert-FileExists -Path "results\_meta\run_manifest.json"

    Assert-MinimumCsvRows `
        -Path "results\main\summary_main.csv" `
        -MinimumRows 40

    # =================================================================
    # Calibration-target sensitivity
    #
    # 4 datasets × 3 tunable methods × 3 targets = 36 rows
    # =================================================================

    Invoke-PythonStep `
        -Name "Run calibration sensitivity" `
        -Arguments @(
            "src\run_calibration_sensitivity.py",
            "--config", "configs\main_cpu.yml",
            "--write_mode", "fail_if_exists"
        )

    Assert-MinimumCsvRows `
        -Path "results\analyses\calibration_sensitivity\calibration_sensitivity.csv" `
        -MinimumRows 36

    # =================================================================
    # Statistical analysis
    # =================================================================

    Invoke-PythonStep `
        -Name "Run bootstrap confidence intervals and paired tests" `
        -Arguments @(
            "src\bootstrap_significance.py",
            "--config", "configs\main_cpu.yml",
            "--n_boot", "2000",
            "--seed", "42",
            "--write_mode", "fail_if_exists"
        )

    Invoke-PythonStep `
        -Name "Run paired effect-size analysis" `
        -Arguments @(
            "src\effect_size_tables.py",
            "--config", "configs\main_cpu.yml",
            "--seed", "42",
            "--write_mode", "fail_if_exists"
        )

    # =================================================================
    # Embedding-backbone sensitivity
    #
    # Required: BM25-SVD, TF-IDF-SVD, unweighted SVD, BPR-MF
    # Optional: two-tower MLP
    # =================================================================

    $EmbeddingArguments = @(
        "src\run_embedding_backbone_sensitivity.py",
        "--config", "configs\analyses.yml",
        "--write_mode", "fail_if_exists"
    )

    if ($IncludeTwoTower) {
        $EmbeddingArguments += "--include_optional_backbones"
    }

    Invoke-PythonStep `
        -Name "Run embedding-backbone sensitivity" `
        -Arguments $EmbeddingArguments

    Assert-MinimumCsvRows `
        -Path "results\analyses\embedding_sensitivity\embedding_backbone_sensitivity_all.csv" `
        -MinimumRows 20

    # =================================================================
    # Exposure and popularity analysis
    # =================================================================

    Invoke-PythonStep `
        -Name "Run exposure and popularity-proxy analysis" `
        -Arguments @(
            "src\run_exposure_analysis.py",
            "--tail_frac", "0.2",
            "--head_frac", "0.1",
            "--write_mode", "fail_if_exists"
        )

    # =================================================================
    # Product-quantization diagnostics
    # =================================================================

    Invoke-PythonStep `
        -Name "Run PQ diagnostics" `
        -Arguments @(
            "src\run_pq_diagnostics.py",
            "--config", "configs\main_cpu.yml",
            "--sample_vectors", "5000",
            "--sample_queries", "1000",
            "--tail_frac", "0.2",
            "--write_mode", "fail_if_exists"
        )

    # =================================================================
    # Synthetic cost-only scale stress
    #
    # 5 catalog sizes × 3 dimensions × 5 methods = 75 rows
    # =================================================================

    if ($SkipScaleStress) {
        Write-Host ""
        Write-Warning ("SKIPPING scale-stress (developer/debug -SkipScaleStress). " +
            "Scale stress is CRITICAL paper evidence: this run CANNOT be " +
            "validator-complete and MUST NOT be used as the final paper run.")
    }
    else {
        Invoke-PythonStep `
            -Name "Run complete scale-stress experiment" `
            -Arguments @(
                "src\run_scale_stress.py",
                "--config", "configs\analyses.yml",
                "--write_mode", "fail_if_exists"
            )

        Assert-MinimumCsvRows `
            -Path "results\analyses\scale_stress\scale_stress_all.csv" `
            -MinimumRows 75
    }

    # =================================================================
    # ANN selection decision framework
    # =================================================================

    Invoke-PythonStep `
        -Name "Generate ANN decision framework" `
        -Arguments @(
            "src\ann_decision_framework.py",
            "--config", "configs\analyses.yml",
            "--write_mode", "fail_if_exists"
        )

    # =================================================================
    # Strict evidence validation
    # =================================================================

    Invoke-PythonStep `
        -Name "Validate all critical paper evidence" `
        -Arguments @(
            "src\validate_paper_evidence.py"
        )

    # =================================================================
    # Claim-support audit
    # =================================================================

    Invoke-PythonStep `
        -Name "Generate claim-support audit" `
        -Arguments @(
            "src\claim_support_audit.py"
        )

    # =================================================================
    # Paper tables and figures
    # =================================================================

    Invoke-PythonStep `
        -Name "Generate paper tables" `
        -Arguments @(
            "src\tables_paper.py",
            "--write_mode", "replace"
        )

    Invoke-PythonStep `
        -Name "Generate paper figures" `
        -Arguments @(
            "src\figures_paper.py",
            "--write_mode", "replace"
        )

    # Validate again after creating the optional claim-support evidence.
    Invoke-PythonStep `
        -Name "Run final paper-evidence validation" `
        -Arguments @(
            "src\validate_paper_evidence.py"
        )

    # =================================================================
    # Required output verification
    # =================================================================

    $RequiredOutputs = @(
        "results\main\summary_main.csv",
        "results\main\run_config.json",
        "results\analyses\calibration_sensitivity\calibration_sensitivity.csv",
        "results\analyses\bootstrap\bootstrap_cis.csv",
        "results\analyses\bootstrap\paired_tests.csv",
        "results\analyses\effect_sizes\effect_sizes.csv",
        "results\analyses\embedding_sensitivity\embedding_backbone_sensitivity_all.csv",
        "results\analyses\exposure\exposure_analysis_all.csv",
        "results\analyses\pq_diagnostics\pq_diagnostics_all.csv",
        "results\analyses\pq_diagnostics\pq_diagnostics_summary.csv",
        "results\analyses\decision_framework\ann_decision_framework_scores.csv",
        "results\paper\tables\dataset_stats.csv",
        "results\paper\tables\claim_support_audit.csv",
        "results\_meta\hardware.json",
        "results\_meta\environment.txt",
        "results\_meta\run_manifest.json",
        "results\_meta\validation_report.csv",
        "results\_meta\validation_report.json",
        "results\_meta\validation_report.md"
    )

    if (-not $SkipScaleStress) {
        $RequiredOutputs += "results\analyses\scale_stress\scale_stress_all.csv"
    }

    foreach ($Output in $RequiredOutputs) {
        Assert-FileExists -Path $Output
    }

    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Green
    Write-Host "FULL EXPERIMENT WORKFLOW COMPLETED SUCCESSFULLY" `
        -ForegroundColor Green
    Write-Host ("=" * 78) -ForegroundColor Green

    Write-Host ""
    Write-Host "Amazon source used:"
    Write-Host "  $AmazonReviewArchive"

    Write-Host ""
    Write-Host "Main results:"
    Write-Host "  results\main\summary_main.csv"

    Write-Host ""
    Write-Host "Validation report:"
    Write-Host "  results\_meta\validation_report.md"

    Write-Host ""
    Write-Host "Paper tables:"
    Write-Host "  results\paper\tables"

    Write-Host ""
    Write-Host "Paper figures:"
    Write-Host "  results\paper\figures"

    Write-Host ""
    Write-Host "Execution log:"
    Write-Host "  $LogPath"
}
catch {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Red
    Write-Host "FULL EXPERIMENT WORKFLOW FAILED" -ForegroundColor Red
    Write-Host ("=" * 78) -ForegroundColor Red
    Write-Host ("Failed stage : {0}" -f $script:CurrentStage) -ForegroundColor Red
    if ($null -ne $LASTEXITCODE) {
        Write-Host ("Exit code    : {0}" -f $LASTEXITCODE) -ForegroundColor Red
    }
    Write-Host ("Error        : {0}" -f $_.Exception.Message) -ForegroundColor Red

    Write-Host ""
    Write-Host "Diagnostics:" -ForegroundColor Red
    Write-Host "  Execution log : $LogPath"
    Write-Host "  Run manifest  : results\_meta\run_manifest.json"
    Write-Host "  Step statuses : results\main\status"

    # Print the last useful output lines from the transcript, if available.
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
    if (Test-Path $LogPath -PathType Leaf) {
        Write-Host ""
        Write-Host "Last 40 log lines:" -ForegroundColor Red
        Get-Content -Path $LogPath -Tail 40 | ForEach-Object {
            Write-Host "  $_"
        }
    }

    exit 1
}
finally {
    try {
        Stop-Transcript
    }
    catch {
        # Ignore transcript shutdown errors.
    }
}