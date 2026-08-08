[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Registry,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')]
    [string]$Tag,

    [string]$PythonBaseImage = "python:3.11.13-slim-bookworm",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
if ($Tag -ieq "latest") {
    throw "The mutable tag 'latest' is forbidden. Use an immutable release tag."
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found."
}

$Root = Split-Path -Parent $PSScriptRoot
$Prefix = $Registry.TrimEnd('/')
$SchedulerImage = "${Prefix}/ml-aware-scheduler:${Tag}"
$TrainerImage = "${Prefix}/ml-sim-job:${Tag}"
$InferenceImage = "${Prefix}/ml-inference-service:${Tag}"

Push-Location $Root
try {
    docker build --pull `
        --build-arg "PYTHON_BASE_IMAGE=$PythonBaseImage" `
        --build-arg "APP_VERSION=$Tag" `
        --file scheduler/Dockerfile `
        --tag $SchedulerImage .
    if ($LASTEXITCODE -ne 0) { throw "Scheduler image build failed." }

    docker build --pull `
        --build-arg "PYTHON_BASE_IMAGE=$PythonBaseImage" `
        --build-arg "APP_VERSION=$Tag" `
        --file k8s/Dockerfile `
        --tag $TrainerImage .
    if ($LASTEXITCODE -ne 0) { throw "Trainer image build failed." }

    docker build --pull `
        --build-arg "PYTHON_BASE_IMAGE=$PythonBaseImage" `
        --build-arg "APP_VERSION=$Tag" `
        --file inference/Dockerfile `
        --tag $InferenceImage .
    if ($LASTEXITCODE -ne 0) { throw "Inference image build failed." }

    docker run --rm --entrypoint python $SchedulerImage -c "import kubernetes,prometheus_client,yaml; import scheduler.rank; print('scheduler image: OK')"
    if ($LASTEXITCODE -ne 0) { throw "Scheduler image smoke test failed." }

    docker run --rm --entrypoint python $TrainerImage -c "import numpy; print('trainer image: OK', numpy.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "Trainer image smoke test failed." }

    docker run --rm --entrypoint python $InferenceImage -c "from inference.service import InferenceModel; print(InferenceModel(4, 2, 1).predict([[0,0,0,0]]))"
    if ($LASTEXITCODE -ne 0) { throw "Inference image smoke test failed." }

    if ($Push) {
        docker push $SchedulerImage
        if ($LASTEXITCODE -ne 0) { throw "Scheduler image push failed." }
        docker push $TrainerImage
        if ($LASTEXITCODE -ne 0) { throw "Trainer image push failed." }
        docker push $InferenceImage
        if ($LASTEXITCODE -ne 0) { throw "Inference image push failed." }
    }

    Write-Host "Scheduler image: $SchedulerImage"
    Write-Host "Trainer image:   $TrainerImage"
    Write-Host "Inference image: $InferenceImage"
    Write-Host "Record the registry-provided sha256 digests before deployment."
} finally {
    Pop-Location
}
