# Article compliance matrix

Article: Abdelshaheed and Ashour, *Intelligent Scheduling of AI Tasks in
Kubernetes* (2025).

This document is an implementation traceability record, not a claim that the
published measurements have already been reproduced.

## Status legend

- **Implemented/local**: source and local validation path exist.
- **Server pending**: implementation exists but requires the target Kubernetes
  server before it can be accepted.
- **Article deviation**: intentionally differs and must be disclosed.
- **Extension**: proposed work not present in the article.

## Architecture and algorithm

| Article requirement | Implementation | Status before cluster execution | Acceptance evidence |
|---|---|---|---|
| Burst-generated annotated ML jobs | `workload/generate_workload.py` | Implemented/local | Deterministic manifest and exact-count tests |
| Features `T,R,M,G,C,P` | generator, trainer, `scheduler/rank.py` | Implemented/local | Feature parsing/range tests |
| Per-burst Min-Max normalization | `scheduler/rank.py` | Implemented/local | Unit vectors including equal-value case |
| Weighted rank 0.40/0.35/0.20/0.15/0.10/0.05 | `scheduler/rank.py` | Implemented/local | Equation parity test |
| Smaller `T,M,G,P`; larger `R,C` preferred | `scheduler/rank.py` | Implemented/local | Directionality tests |
| Descending rank order | reproduction scheduler | Implemented/local | Ordered schedule record; server pending |
| Reversed-order ablation | `--reverse` | Implemented/local | Reversed rank order in a completed cluster run |
| Brief burst collection interval | burst collector and `--quiet-period` | Implemented/local | Expected-count and timeout tests; server pending |
| Category-neutral simultaneous submission | seeded concurrent API submitter with one worker per Pod | Implemented/local | Submission order, UID/timestamps and API-server creation spread are embedded and strictly validated |
| Priority queue behavior | deterministic sorted burst | Implemented/local | Complete order contains every expected Pod once |
| Manual Pod binding | reproduction scheduler | Server pending | RBAC `create pods/binding`, successful API record, Pod node assignment |
| Single-node placement | explicit `--target-node` | Server pending | Ready/uncordoned target-node preflight |
| Execution-start confirmation | trainer marker and log watcher | Server pending | Every bound Pod yields a valid marker within timeout |
| Pacing delay | none/fixed pacing | Implemented/local | ILT and fixed-delay tests; cluster timing pending |
| In-cluster standalone scheduler | scheduler image + Helm reproduction profile | Server pending | Pod Ready, probes green, in-cluster auth, completed burst |

## Workload simulator

| Article requirement | Implementation | Status before cluster execution | Notes |
|---|---|---|---|
| Four synthetic workload categories | workload generator | Implemented/local | Article does not publish all distribution parameters; repository choices must be reported |
| Matrix multiplication | `k8s/train.py` | Implemented/local | Actual CPU/BLAS behavior is server dependent |
| Gradient/update effect | real `G`-MiB NumPy buffer update | Implemented/local | Memory bandwidth remains hardware dependent |
| Exponential loss decay using `R` | trainer | Implemented/local | Validate logged terminal state |
| Checkpoint interval `C` | bounded file write plus `fsync` | Implemented/local | Payload size and bandwidth assumptions are versioned |
| Partition overhead `P` | disjoint matrix row partitions plus `G`-MiB peer copies | Implemented/local | Physical synchronization proxy, not multi-node distributed training |
| `T` is derived estimate, not termination | shared work model | Implemented/local | Generator, trainer evidence, and simulator share model version `2.0` |
| No workload requests/limits | reproduction manifests only | Server pending | Deliberately isolated profile; not a production default |
| Controlled BLAS concurrency | trainer image environment | Implemented/local | One thread by default for repeatability |

## Experiment protocol

| Article requirement | Concrete matrix/implementation | Status |
|---|---|---|
| Single-node Minikube, 4 CPU, 8 GiB | fail-closed `article-exact` environment policy | Implemented/local; server pending |
| Clean isolated runs | target-node inventory rejects every active non-system/non-scheduler Pod before each run | Implemented/local; server pending |
| Identical default/custom workload | deterministic paired generation | Implemented/local; server pairing pending |
| 12 jobs normal | `12-normal` | Server pending |
| 48 jobs normal | `48-normal` | Server pending |
| 48 jobs half | `48-half` | Server pending |
| Fixed pacing 0, 1, 2, 5 seconds | pacing block | Server pending |
| Default baseline | `default` | Server pending |
| Intended custom order | `custom-baseline` (plus fixed-delay variants) | Server pending |
| Reversed ablation | `reversed` | Server pending |
| Five independent repetitions | canonical repetition labels `0..4` | Server pending |
| Full article-only run plan | 25 pacing + 45 main = **70** | Server pending |

### Exact 70-run article-only matrix

The pacing block uses `48-half-pacing` with `default`, `custom-baseline` (the
zero-second custom case),
`custom-delay-1s`, `custom-delay-2s`, and `custom-delay-5s`, each repeated five
times: 25 runs.

The main block uses `12-normal`, `48-normal`, and `48-half` with `default`,
`custom-baseline`, and `reversed`, each repeated five times: 45 runs.

The blocks intentionally remain separate even where a scenario/config pair
appears in both, because they answer different experiment questions and must
retain independent run IDs.

## Measurement and statistics

| Requirement | Implementation expectation | Acceptance rule |
|---|---|---|
| JCT per job | creation-to-completion timestamp | Reject missing or negative intervals |
| Average/min/max JCT | aggregate within each valid run | Never pool partial runs |
| p95 JCT | explicit NumPy percentile method | Record method and library version |
| Makespan | earliest creation to latest completion | All expected jobs must be complete |
| ILT | consecutive execution-start timestamps | Sort timestamps; require complete marker set |
| Mean ECDF over repetitions | interpolate each run ECDF on common grid | Do not pool jobs across repetitions |
| IQR band | 25th/75th percentiles across run ECDFs | Plot and retain underlying numeric data |
| Reproducibility metadata | schema-versioned result records | Include seed, scenario, config, repetition, image digests, Kubernetes/node/tool versions |
| Failure handling | strict collector | Any failed/missing/duplicate Pod invalidates the run |

Checked-in simulator CSV/JSON/PNG files are reference artifacts only. They are
not cluster measurements and do not satisfy the article's evaluation evidence.

## Adaptive RFAP extension—explicitly not part of the article

Adaptive pacing is an extension motivated by the fixed-delay sensitivity. It
must be reported separately and must never be described as article behavior.

The extended **90-run** plan adds:

- five adaptive runs in the `48-half` pacing block;
- five adaptive runs for each of the three main scenarios (15 runs).

Total extension: 20 runs; article-only 70 + extension 20 = 90.

Adaptive acceptance additionally requires fresh metrics timestamps, bounded
wall-clock waits, hysteresis, explicit unavailable/stale-metric failure, and a
record of every pacing decision. Missing metrics must never be interpreted as
zero utilization.

The dynamic matrix Helm profile grants the separate Node Metrics read
capability without changing the per-run fallback mode. The runner measures an
advancing Metrics API timestamp before the 90-run plan, and scheduler records
persist distinct metrics samples plus every pacing wait start/completion.

## Production-safe profile—also not an article reproduction

The `production` Helm values run `scheduler.gate_controller`, which removes the
stable `ml.scheduler/release` Pod scheduling gate in ranked order. The normal
`default-scheduler` then performs feasibility filters, scoring, volume/taint/
affinity handling, and binding. This profile improves Kubernetes compatibility
but is not numerically comparable to manual-binding article runs unless it is
reported as a separate experimental configuration.

## Infrastructure compliance

| Control | Repository implementation | Acceptance state |
|---|---|---|
| Non-root images | UID/GID 10001 in both Dockerfiles | Image build/server pending |
| Pinned Python packages | role-specific exact requirements | Local install/import/`pip check` complete; image inspection server pending |
| Immutable release image | Helm `image.digest` support | Registry digest/server pending |
| Least-privilege namespace RBAC | Role/RoleBinding; mode-conditional verbs | Server authorization pending |
| Minimal cluster RBAC | Node/metrics ClusterRole only when required | Server authorization pending |
| Secret hygiene | pull-secret references only | Server secret mechanism pending |
| Read-only filesystem | container security context plus bounded `/tmp` and `/results` | Server pending |
| Probes/metrics | `/livez`, `/readyz`, `/metrics` on 8080 | Server pending |
| NetworkPolicy | ingress limited to namespace/additional selectors | CNI enforcement pending |
| Resource controls | scheduler requests/limits | Server pending |
| Rollout/rollback | atomic Helm install and revision rollback | Server pending |
| HPA | intentionally omitted | Multiple active controllers are unsafe without leader election |
| Ingress | intentionally omitted | No public service is required |
| PDB | supplied but disabled | Enable only after replica/leader-election design changes |
| ServiceMonitor | supplied but disabled | Requires Prometheus Operator CRD |

The automated evidence chain additionally locks the expanded plan by SHA-256,
verifies reviewed workload/manifests before reuse, validates the live dynamic
Deployment and RBAC, records requested and runtime image digests, embeds the
authoritative scheduler record in every custom result, and refuses cleanup
until strict collection and evidence validation succeed.

## Known article ambiguities that cannot be invented

Exact original feature ranges/proportions, random seeds, calibration method for
rank weights, training constants, image/cache warm-up policy, burst duration,
software versions, host hardware, percentile interpolation, and complete node
filtering logic are not fully specified by the article. Repository choices for
these items must be recorded as reproduction assumptions, not attributed to
the authors.
