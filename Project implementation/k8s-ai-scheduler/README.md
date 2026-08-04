# ML-aware Kubernetes Scheduler

This repository reproduces the scheduling design described by Abdelshaheed and
Ashour (2025) and keeps the proposed resource-feedback adaptive pacing (RFAP)
extension in a separate evaluation track. Simulator output is exploratory;
only completed, validated Kubernetes runs may be reported as article
reproduction results.

## Execution profiles

| Profile | Purpose | Placement mechanism | Intended environment |
|---|---|---|---|
| `reproduction` | Faithful single-node article experiment | Custom scheduler manually creates Pod bindings | Dedicated disposable single-node cluster only |
| `reproduction-matrix` | Dynamic controller for the locked 70/90-run protocol | Same manual binding, with each run contract read from Pod annotations | Dedicated disposable single-node cluster only |
| `production` | Safer integration for real workloads | Controller removes `ml.scheduler/release`; default kube-scheduler performs filtering, scoring, and binding | Kubernetes 1.30+ |

The reproduction profile intentionally retains the article's no-resource-limit,
single-node contention experiment. It must not be used as a general-purpose
multi-node scheduler. The production profile preserves Kubernetes placement
safety through stable Pod Scheduling Readiness gates.

The packaged Helm release requires Kubernetes 1.30 or newer for all profiles.

## Repository layout

```text
scheduler/                 Ranking, reproduction scheduler, gated controller
workload/                  Deterministic paired workload generation
k8s/                       Synthetic trainer and its container image
results/                   Strict collection code and local generated outputs
sim/                       Proxy simulation and statistical plots
experiments/               Locked plans, result schema, orchestration, analysis
deploy/helm/               Reproduction and production Helm profiles
scripts/                   Local build/preflight and Linux server operations
docs/                      Compliance and pre-cluster acceptance checklists
```

## Pinned toolchain

- Python: `>=3.11,<3.13`
- Kubernetes Python client: `36.0.2`
- NumPy: `2.3.5`
- pandas: `3.0.3`
- Matplotlib: `3.11.0`
- PyYAML: `6.0.3`
- prometheus-client: `0.25.0`
- pytest: `9.1.1`
- kubernetes-validate schemas: `1.36.0`
- Ruff: `0.16.1`
- Python build frontend: `1.5.0`

Direct dependencies are exactly pinned in `requirements.txt`; transitive
resolution is frozen in `constraints.txt`. The two container images use
smaller role-specific requirement files against the same constraints. The default
base image is versioned (`python:3.11.13-slim-bookworm`); before a controlled
release, resolve it to a registry-verified `tag@sha256:digest` and pass that
complete reference through `PYTHON_BASE_IMAGE`.

## Windows PowerShell setup and local validation

Run from the repository root:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
& .\scripts\preflight.ps1 -Python ".\.venv\Scripts\python.exe"
```

The preflight compiles the packages, checks the locked environment, runs Ruff
and pytest, validates both immutable experiment plans, and (when Helm is
available) lints and schema-validates every deployment profile. The same gates
run on Python 3.11 and 3.12 in GitHub Actions.

Build a wheel, source distribution, and checksum sidecar without committing
generated archives:

```powershell
python .\scripts\build_release.py
Get-Content .\dist\SHA256SUMS.txt
```

Build and optionally push immutable, non-`latest` images:

```powershell
$Registry = "registry.example.com/your-project"
$Tag = "0.1.0"
docker login ($Registry.Split('/')[0])
& .\scripts\build-images.ps1 -Registry $Registry -Tag $Tag -Push
```

Replace the registry, project, and credentials with values from the target
environment. Never commit a registry password, kubeconfig, pull secret, or
local values file.

## Tests and offline simulation

```powershell
python -m pytest
python -m sim.calibration --output artifacts/calibration.json
python -m sim.run_experiments --results-dir artifacts/sim `
  --calibration artifacts/calibration.json --require-calibration
python -m sim.plot_results --results-dir artifacts/sim
```

The simulator validates ranking and analysis logic but does not prove
Kubernetes behavior or reproduce the article's wall-clock measurements. Its
JSON records always remain `exploratory-only`; a content-addressed calibration
records the measured machine and trainer primitive samples, while an omitted
calibration is explicitly labeled `uncalibrated-assumptions`.

Workload generation, the trainer, collection validation, and the simulator
share the versioned model in `k8s/work_model.py`. `T` is derived from
`R/M/G/C/P`; the trainer performs matrix work over disjoint row partitions,
updates a real `G`-MiB gradient buffer, copies it for partition synchronization,
and writes plus `fsync`s bounded checkpoints every `C` steps. Every Pod and
terminal log records the model version, and strict collection rejects stale or
incomplete work-model evidence.

The ambiguous half-load profile is operationalized as 50% of aggregate
estimated physical work. A deterministic burst-level search selects one global
`M` scale (rather than blindly halving a cubic dimension), recomputes `T`, and
stores the target, achieved ratio, tolerance, and selected scale in `jobs.json`.

## Kubernetes packaging

The Helm chart includes:

- Deployment, Service, ConfigMap, ServiceAccount, namespace-scoped RBAC;
- conditional Node/metrics ClusterRole permissions;
- non-root UID/GID 10001, read-only root filesystem, dropped capabilities,
  RuntimeDefault seccomp, resource requests/limits, and bounded `emptyDir`s;
- results PVC/existing-claim support, enabled by the production profile so
  in-progress state survives Pod replacement;
- startup/liveness `/livez`, readiness `/readyz`, and `/metrics` on port 8080;
- ingress plus Kubernetes-API-port-only egress NetworkPolicy (optionally CIDR-bound);
- PDB enabled by the production profile and an optional ServiceMonitor;
- image pull-secret references and digest-based image selection.
- a JSON Schema for Helm values and a dedicated dynamic matrix values file.

Ingress and HPA are intentionally absent: the controller has no public API,
and multiple replicas are unsafe until leader election has been implemented
and tested. Rollouts use `maxSurge: 0` to prevent old/new controller overlap,
which deliberately introduces a short control-plane gap during upgrades.
The production controller resumes a versioned atomic run record after restart,
matches every Pod by name and UID, reconciles already-removed gates without a
second release, and repeats an interrupted pacing delay conservatively. Gated
manifests receive CPU/memory bounds and an explicit execution-container
annotation. A stranded run makes `/readyz` fail while `/livez` and metrics stay
reachable through the Service for diagnosis.

```powershell
helm lint .\deploy\helm\ml-ai-scheduler
helm template ml-ai-scheduler .\deploy\helm\ml-ai-scheduler `
  --namespace ai-scheduler `
  --values .\deploy\helm\ml-ai-scheduler\values-production.yaml
```

The full Linux server procedure—including context guards, image push, RBAC
checks, server-side dry run, smoke tests, diagnostics, and rollback—is in
[`run_on_cluster.md`](run_on_cluster.md).

## Experiment matrices

The article-only matrix contains **70 independent runs**:

- pacing study: one scenario × five configurations × five repetitions = 25;
- main study: three scenarios × three configurations × five repetitions = 45.

The extended matrix contains **90 runs** by adding 20 RFAP runs: five in the
pacing study and 15 across the three main scenarios. Exact definitions and
labels are pre-registered in `experiments/scenarios.yaml` and documented in
`docs/ARTICLE_COMPLIANCE.md`; the Persian status checklist is
`docs/ARTICLE_MATCH_CHECKLIST_FA.md`. `python -m experiments.run_cluster` validates,
materializes, and—only with explicit `--execute`—runs the plan; see the server
runbook before enabling execution.

Cluster execution never applies a category-named manifest directory with
`kubectl`. The runner submits every Pod through a seed-controlled concurrent
Kubernetes API barrier, records the actual server creation timestamps, and
rejects a burst whose creation spread exceeds the configured limit. The
default `article-exact` environment profile also rejects multi-node, non-
Minikube, non-4-CPU/8-GiB, pressured, or shared target nodes. Use
`--environment-profile record-only` only for smoke tests; those results are
explicitly ineligible for an article reproduction claim.

Create and review the immutable plan locks locally:

```powershell
python -m experiments.run_cluster --plan-out experiments/locks/article-70.json
python -m experiments.run_cluster --include-adaptive `
  --plan-out experiments/locks/extended-90.json
```

Cluster execution additionally requires an explicit kubeconfig context,
target node, digest-pinned trainer image, and the live dynamic reproduction
Deployment. The runner validates the complete observed Pod contract, reuses
reviewed manifests only when they still match exactly, archives scheduler
order/pacing records before cleanup, captures image and cluster metadata, and
refuses a scheduling gate on default baselines. It now also attests the running
Minikube Docker profile, prewarms
the exact trainer digest on the target node, proves the actual BLAS pool is
single-threaded, and enforces a registered 30-second clean cooldown before
every run. These control records are covered by the per-run snapshot hash;
strict resume and analysis reject missing or altered evidence.
Exact Linux commands are in
[`run_on_cluster.md`](run_on_cluster.md).

## Result acceptance rules

A run is valid only when all expected Pods belong to one unique `run-id`, all
complete successfully, every required timestamp is present and ordered, the
rank/order record is complete, and cluster/image/version metadata is stored.
Partial runs must fail closed; they must not contribute to aggregates.

## Current acceptance boundary

Local source checks can be completed on a workstation. Image builds, Helm
server-side validation, RBAC authorization, live scheduling order, probes,
metrics freshness, and the 70/90-run matrices are server-only validations and
remain pending until executed against the target cluster. See
`docs/VALIDATION_REPORT.md` for captured local evidence and
`docs/PRE_CLUSTER_CHECKLIST.md` before deployment.
