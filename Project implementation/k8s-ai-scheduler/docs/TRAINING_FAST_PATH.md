# Train-only FastPath production extension

## Scope

FastPath is a production scheduling-gate extension for explicit ML training
bursts. It does not change the locked article-reproduction scheduler and it is
not enabled for inference or generic Kubernetes workloads.

Eligibility is fail-closed. A burst must provide:

- `ml.scheduler/workload-kind: training` and all six `T/R/M/G/C/P` features;
- a CPU request or limit for every application and init container;
- a fresh target-node Metrics API sample with allocatable CPU capacity;
- no fixed/adaptive pacing and no reversed ablation order.

Missing annotations, unbounded demand, stale/unavailable metrics or a fully
consumed headroom window retain the conservative ranked release path.

## Scheduling behavior

1. Rank the complete burst with the unchanged six-feature equation.
2. Read current target-node CPU utilization and compute bounded Pod peak CPU.
3. If the full burst fits under the configured threshold, remove all release
   gates immediately.
4. If at least one Pod fits but the burst is larger than current headroom,
   prefill kube-scheduler's queue in ranked order. Gate removal does not bind a
   Pod or allocate CPU; `default-scheduler` still performs feasibility,
   scoring and binding against normal requests, taints, affinity and volumes.
5. Preserve the high-priority prefix and exhaustively evaluate permutations of
   at most four lowest-ranked training jobs using predicted `T`. Choose the
   suffix with the lowest predicted list-schedule makespan, then the lowest
   predicted completion-time sum.
6. Persist the admission decision, queue order, predicted tail evidence and
   execution-start records so controller restart cannot duplicate a release.

The production Helm profile enables FastPath with an 85% target-node CPU
threshold and the RBAC needed to read Node and `metrics.k8s.io` data.

## Paired France pilot

Date: 2026-08-08

The final pilot used one shared four-core France K3s node. OpenVPN and the
Marzban node remained Ready with unchanged restart counts before and after all
runs. The workload used the versioned NumPy matrix/gradient/checkpoint trainer,
12 heavy training Pods per arm, one CPU per Pod, seeds 8100 through 8104 and
five paired repetitions. Arm order alternated by repetition and the harness
required two cool-node samples before each arm.

Both arms used a synchronized launch barrier to remove sequential Kubernetes
API creation spread from the comparison. The baseline barrier was removed in
manifest/FIFO order and every Pod still used `default-scheduler`; it had no
custom rank or binding. KubeML removed the same barrier in its six-feature
ranked/tail-balanced order.

| Metric | Kubernetes baseline | KubeML FastPath | Mean paired improvement |
|---|---:|---:|---:|
| Average JCT | 31.047 s | 28.755 s | **+7.31%** |
| p95 JCT | 52.896 s | 53.270 s | **-0.75%** |
| Makespan | 55.369 s | 54.005 s | **+2.47%** |

Average JCT improved in all five paired repetitions. Makespan also improved in
all five. p95 improved in two repetitions and regressed in three, so the small
aggregate tail regression is a disclosed trade-off rather than a claimed win.

## Ablations retained

- The earlier mixed profile was dominated by sub-second jobs; controller and
  Kubernetes startup overhead exceeded useful training work. Across five
  repetitions it was 2.75% worse in average JCT and 7.54% worse in makespan.
- A heavy-workload version with only a small speculative queue and tail
  balancing was effectively tied on average JCT (-0.08%) and 4.35% worse in
  makespan. That negative result motivated complete ranked queue prefill.
- A raw, unsynchronized baseline is retained outside the repository evidence.
  It exposes the real first-arrival advantage of sequential API creation but
  confounds burst scheduling with submission order; it is not used for the
  algorithm-only comparison above.

## Claim boundary

This pilot validates the production extension on a real cluster, not the
article's registered 70/90-run matrix. The node was shared with VPN services,
the trainer was CPU-only and the environment was not a dedicated Minikube
article-exact host. GPU behavior, multi-node placement and confidence intervals
from the complete registered matrix remain future validation work.

The sanitized aggregate evidence is
[`docs/evidence/TRAINING_FAST_PATH_FRANCE_20260808.json`](evidence/TRAINING_FAST_PATH_FRANCE_20260808.json).
