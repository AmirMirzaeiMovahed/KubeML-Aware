[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
        throw "Python executable '$Python' was not found. Install Python 3.11 or pass -Python with its full path."
    }

    & $Python -c "import sys; assert (3,11) <= sys.version_info[:2] < (3,13), sys.version; print(sys.version)"
    if ($LASTEXITCODE -ne 0) { throw "Python version check failed." }

    & $Python -m compileall -q scheduler workload k8s results sim experiments
    if ($LASTEXITCODE -ne 0) { throw "Python syntax validation failed." }

    & $Python -c "import jsonschema,kubernetes,kubernetes_validate,matplotlib,numpy,pandas,prometheus_client,yaml; print('runtime imports: OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime dependencies are missing or incompatible. Run: $Python -m pip install -r requirements-dev.txt"
    }

    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Installed dependency validation failed." }

    & $Python -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw "Ruff quality gate failed." }

    & $Python -m experiments.run_cluster `
        --plan-out experiments/locks/article-70.json
    if ($LASTEXITCODE -ne 0) { throw "Article plan lock validation failed." }
    & $Python -m experiments.run_cluster `
        --include-adaptive `
        --plan-out experiments/locks/extended-90.json
    if ($LASTEXITCODE -ne 0) { throw "Extended plan lock validation failed." }

    if (-not $SkipTests) {
        & $Python -m pytest
        if ($LASTEXITCODE -ne 0) { throw "Test suite failed." }
    }

    if (Get-Command helm -ErrorAction SilentlyContinue) {
        & helm lint deploy/helm/ml-ai-scheduler
        if ($LASTEXITCODE -ne 0) { throw "Default Helm lint failed." }
        & helm lint deploy/helm/ml-ai-scheduler --values deploy/helm/ml-ai-scheduler/values-production.yaml
        if ($LASTEXITCODE -ne 0) { throw "Production Helm lint failed." }
        foreach ($Values in @("values-reproduction.yaml", "values-reproduction-matrix.yaml")) {
            & helm lint deploy/helm/ml-ai-scheduler `
                --values "deploy/helm/ml-ai-scheduler/$Values" `
                --set-string scheduler.targetNode=preflight-node
            if ($LASTEXITCODE -ne 0) { throw "Helm lint failed for $Values." }
        }
        $Profiles = @(
            @{ Name = "production"; Values = "values-production.yaml"; Target = $false },
            @{ Name = "reproduction"; Values = "values-reproduction.yaml"; Target = $true },
            @{ Name = "matrix"; Values = "values-reproduction-matrix.yaml"; Target = $true }
        )
        foreach ($Profile in $Profiles) {
            $TempFile = Join-Path ([IO.Path]::GetTempPath()) (
                "ml-ai-scheduler-$($Profile.Name)-$([guid]::NewGuid().ToString('N')).yaml"
            )
            try {
                $HelmArgs = @(
                    "template", "ml-ai-scheduler", "deploy/helm/ml-ai-scheduler",
                    "--namespace", "ai-scheduler", "--values",
                    "deploy/helm/ml-ai-scheduler/$($Profile.Values)"
                )
                if ($Profile.Target) {
                    $HelmArgs += @("--set-string", "scheduler.targetNode=preflight-node")
                }
                $Rendered = & helm @HelmArgs
                if ($LASTEXITCODE -ne 0) { throw "Helm render failed for $($Profile.Name)." }
                [IO.File]::WriteAllText(
                    $TempFile,
                    ($Rendered -join [Environment]::NewLine),
                    [Text.UTF8Encoding]::new($false)
                )
                & $Python scripts/validate_manifests.py $TempFile --kubernetes-version 1.36.0
                if ($LASTEXITCODE -ne 0) {
                    throw "Kubernetes schema validation failed for $($Profile.Name)."
                }
            } finally {
                if (Test-Path -LiteralPath $TempFile) {
                    Remove-Item -LiteralPath $TempFile -Force
                }
            }
        }
    } else {
        Write-Warning "Helm is not installed locally; Helm validation remains a server-side check."
    }

    Write-Host "Local preflight completed successfully." -ForegroundColor Green
} finally {
    Pop-Location
}
