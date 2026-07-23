# Pre-cluster validation report

Date: 2026-07-11

This report records only validations completed without a reachable target
Kubernetes cluster or local Docker daemon. It is not evidence of a successful
image build, registry push, server-side dry run, or live deployment.

## Release candidate

- Project version: `0.1.0`
- Local Python: `3.12.13`
- Helm used for validation: `v4.2.0`
- Offline Kubernetes schema target: `1.36.0`
- Article plan lock: `experiments/locks/article-70.json`
  - 70 runs; 25 pacing + 45 main
  - canonical plan SHA-256:
    `d1fd13b06d446e1360fcee0e37199811c64ec49f0fb687bba6d71bdda0817ae8`
- Extended plan lock: `experiments/locks/extended-90.json`
  - 90 runs; 30 pacing + 60 main
  - canonical plan SHA-256:
    `c0a1f72dd626cfa8c83f096c78fcfd03c42a6787a141e04eb88e310d96866337`
- Wheel: `dist/k8s_ai_scheduler-0.1.0-py3-none-any.whl`
  - file SHA-256:
    `39c3ed7c4566134e4c4d53afade5c6423e40f78f5ab4dea6f668eeac21de8017`
- Clean source archive: `dist/k8s-ai-scheduler-0.1.0-source.zip`
  - excludes `.venv`, caches, build directories, and validation scratch data;
  - archive integrity and required-content checks passed;
  - final checksum is recorded in `dist/SHA256SUMS.txt`.

## Completed local gates

| Gate | Result |
|---|---|
| Exact dependency imports and `pip check` | Passed; no broken requirements |
| Python compile check | Passed for scheduler, workload, trainer, results, simulation, and experiments |
| Automated tests | **88 passed**, zero skipped/failing |
| PowerShell parser validation | Passed |
| Bash `-n` validation | Passed |
| Helm values JSON Schema | Passed |
| Helm lint | Passed for default, production, fixed reproduction, and dynamic matrix values |
| Helm rendering | Passed for production, fixed reproduction, and dynamic matrix profiles |
| Strict Kubernetes 1.36 schema validation | Passed for 25 rendered Helm objects |
| Conditional RBAC inspection | Matrix includes Node Metrics read; fixed reproduction does not |
| Helm negative cases | Missing target node, multiple replicas, `latest`, and invalid digest all rejected |
| Dynamic Deployment contract | Digest rendering, custom scheduler module, target node, results template, and absence of fixed run filter verified |
| Deterministic workload materialization | A 48-Pod run was generated, reused after exact comparison, and strict-schema validated |
| Workload Pod security | Non-root, read-only root, dropped capabilities, RuntimeDefault seccomp, disabled API token/service links, bounded `/tmp` verified |
| Private-registry wiring | Workload manifests and `jobs.json` carry the configured pull Secret reference |
| Plan invariants | Exact 70/90 counts, unique run IDs, `0..4` repetitions, paired seeds, and canonical hashes verified |
| Result/evidence analysis | Complete 70-run synthetic Kubernetes-shaped evidence set passed strict end-to-end analysis |
| Simulator smoke | Run/CSV/mean-ECDF/IQR plot generation passed; output remains proxy evidence only |
| Wheel build/install | Built, contents inspected, installed outside the source directory, resources and 70/90 plans imported successfully |
| Clean source backup | Archive contents/integrity verified; checksum sidecar supplied |

The reproducible local command is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\preflight.ps1 `
  -Python .\.venv\Scripts\python.exe
```

## Fail-closed execution controls implemented

- `--execute` requires an explicit kubeconfig context and target node.
- Trainer and live scheduler images must be digest-pinned.
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
