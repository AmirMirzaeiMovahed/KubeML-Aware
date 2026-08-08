# Pre-cluster validation report

Date: 2026-08-04

This report records only validations completed without a reachable target
Kubernetes cluster or local Docker daemon. It is not evidence of a successful
image build, registry push, server-side dry run, or live deployment.

## Release candidate

- Project version: `0.2.1`
- Local Python: `3.11.9`
- Helm used for validation: `v4.2.0`
- Offline Kubernetes schema target: `1.36.0`
- Article plan lock: `experiments/locks/article-70.json`
  - schema 1.2; 70 runs; 25 pacing + 45 main
  - canonical plan SHA-256:
    `6076e7863c7b2507baf856c3f78750f594f834590847a0d1f403758b8bac4566`
- Extended plan lock: `experiments/locks/extended-90.json`
  - schema 1.2; 90 runs; 30 pacing + 60 main
  - canonical plan SHA-256:
    `a383708fc9cecaec48d573716687a51d9ef9d379d5e9a932029e2487f688b52c`
- Release builder: `python scripts/build_release.py`
  - creates the standard wheel and sdist under the ignored `dist/` directory;
  - writes their current SHA-256 values to `dist/SHA256SUMS.txt` after build;
  - avoids a stale checksum in version-controlled documentation.

## Completed local gates

| Gate | Result |
|---|---|
| Exact dependency imports and `pip check` | Passed; no broken requirements |
| Ruff quality gate | Passed repository-wide |
| Python compile check | Passed for scheduler, workload, trainer, results, simulation, and experiments |
| Automated tests | **126 passed**, zero skipped/failing |
| PowerShell parser validation | Passed |
| Bash `-n` validation | Passed |
| Helm values JSON Schema | Passed |
| Helm lint | Passed for default, production, fixed reproduction, and dynamic matrix values |
| Helm rendering | Passed for production, fixed reproduction, and dynamic matrix profiles |
| Strict Kubernetes 1.36 schema validation | Passed for 27 rendered Helm objects |
| Conditional RBAC inspection | Matrix includes Node Metrics read; fixed reproduction does not |
| Helm negative cases | Missing target node, multiple replicas, `latest`, and invalid digest all rejected |
| Dynamic Deployment contract | Digest rendering, custom scheduler module, target node, results template, and absence of fixed run filter verified |
| Runtime drift contract | Live Deployment timing/API/adaptive arguments must equal computed Helm values; the same versioned contract is embedded in every scheduler record |
| Deterministic workload materialization | A 48-Pod run was generated, reused after exact comparison, and strict-schema validated |
| Workload Pod security | Non-root, read-only root, dropped capabilities, RuntimeDefault seccomp, disabled API token/service links, bounded `/tmp` verified |
| Private-registry wiring | Workload manifests and `jobs.json` carry the configured pull Secret reference |
| Production restart recovery | Pod UID-bound state resumes a simulated mid-release interruption without duplicate release |
| Production workload controls | Gated manifests carry CPU/memory bounds and an explicit execution-container contract |
| Production isolation | PVC, PDB, diagnostic Service, ingress and API-port egress policies render and validate |
| Plan invariants | Exact 70/90 counts, unique run IDs, `0..4` repetitions, paired seeds, and canonical hashes verified |
| Result/evidence analysis | Complete 70-run synthetic Kubernetes-shaped evidence set passed strict end-to-end analysis |
| Simulator smoke | Run/CSV/mean-ECDF/IQR plot generation passed; output remains proxy evidence only |
| Simulator calibration | Content-addressed hardware benchmark CLI and tamper/stale-model rejection passed |
| Paired inference | Default-minus-custom improvements and reversed-minus-intended degradations use paired Student-t 95% CIs by scenario/repetition/seed |
| Published reference | Table I, Figures 2-4, and Section V-C aggregate claims are versioned and produce 54 diagnostic observed-delta rows without an invented threshold |
| Wheel build/install | Built, contents inspected, installed outside the source directory, resources and 70/90 plans imported successfully |
| Release archives | Standard wheel and sdist built; checksum sidecar generated |
| CI contract | Locked Python 3.11/3.12 tests plus Helm/package job committed |

The reproducible local command is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\preflight.ps1 `
  -Python .\.venv\Scripts\python.exe
```

## Fail-closed execution controls implemented

- `--execute` requires an explicit kubeconfig context and target node.
- Trainer and live scheduler images must be digest-pinned.
- The active Minikube profile must attest a running Docker driver and healthy
  host, kubelet, API server, and kubeconfig state.
- The exact trainer digest is prewarmed on the target node; its runtime image
  ID and actual single-thread BLAS pools are content-addressed in the evidence.
- Every registered run starts after a 30-second clean cooldown with at least
  three workload-free, pressure-free, scheduler-continuity polls.
- The expanded plan lock must already exist and match exactly.
- The live Helm release must be deployed in dynamic reproduction mode.
- The live Deployment must be Ready, single-replica, restart-free, use the
  expected scheduler/target/results contract, and have the required RBAC.
- Default baselines cannot be contaminated with a scheduling gate.
- Reviewed manifests are reused only after exact deterministic regeneration
  and semantic comparison; drift aborts execution.
- The collector observes and validates seed, expected count, pacing, delay,
  reverse, load profile, scheduler name, gate state, image, pull Secret,
  trainer environment, runtime image ID, labels, features, and timestamps.
- Scheduler Pod replacement/restart or a failed scheduler record aborts a run
  before the one-hour workload timeout.
- Every custom run must provide a complete scheduler record with exact job set,
  ranks, order, bind/execution timestamps, pacing waits, and adaptive samples.
- Results embed plan/artifact hashes and cluster, Helm, image, node, context,
  and scheduler evidence before label-scoped Pod cleanup.
- Resume and final analysis revalidate results against the exact RunSpec and
  plan hash; partial or mismatched evidence is rejected.

## Server-only gates still required

The following cannot be marked complete on this workstation:

- Docker build and local container smoke tests for both images;
- registry authentication, push, registry-provided digests, and node pulls;
- target Kubernetes version/distribution/context and API readiness;
- `kubectl apply --dry-run=server` against the target API;
- live ServiceAccount authorization, CNI NetworkPolicy behavior, probes,
  scheduler metrics, and Metrics API cadence;
- three-job reproduction and production-gate smoke tests;
- the one-run matrix pilot, complete 70/90 execution, rollback, and recovery
  tests.

These are intentionally unchecked in `docs/PRE_CLUSTER_CHECKLIST.md`. Follow
`run_on_cluster.md` in order and preserve every command output before moving
from one gate to the next.
