# France deployment evidence

Validation date: 2026-08-08 UTC

This is a production-path smoke validation of the inference/profiling extension.
It is not a substitute for executing and analysing the locked 70-run article
matrix or the 90-run extended matrix.

## Environment

- Single-node K3s: `v1.36.2+k3s1`
- Knative Serving and Kourier manifests: `knative-v1.23.0`
- Knative Service: `kubeml-inference-00002`, Ready
- External TLS endpoint:
  `https://kubeml-inference.kubeml-inference.167-104-216-211.nip.io`
- Inference image manifest:
  `sha256:fa983f544ff804eefa2b37b457f179452b3bc3a5920f174180c2ce1ac52c12f1`
- Scheduler image manifest:
  `sha256:a75773b63a6520b1cbddf191a01bd239150721573e13cb6b1174ce62ee6722d2`

The Kourier Service uses `ClusterIP`; the existing ingress-nginx controller is
the only host listener on ports 80/443. This avoids a ServiceLB collision with
the server's existing networking workloads.

## Inference and autoscaling result

- The Knative revision reached zero Pods after 77 seconds without requests.
- An external HTTPS request from scale zero returned in 5,058.74 ms.
- The service reported 0.3993 ms model execution latency and 30.7974 ms process
  cold-start initialization for that request.
- The returned probability vector summed to exactly `1.0`.
- A separate 20-request measured profile produced p95 latency 0.4663928645 ms,
  observed throughput 3.4672546183 requests/s, peak RSS 38.17578125 MiB, and
  cold start 29.8931910656 ms.

Evidence hashes:

| Artifact | SHA-256 |
|---|---|
| `inference-samples-0.2.1.jsonl` | `c9df264b0cff56cac884fd3a4cae7d5647c35300d39146d4beccd9d4f6a51ca1` |
| `inference-profile-0.2.1.json` | `02938e5c5d05072b4cb05d18f56252bdfb0db5226192275c4f4463372adcb414` |

## Scheduling-gate result

The long-lived controller remained a normal Kubernetes Deployment. A two-Pod
inference burst used identical run metadata and different measured/SLO inputs:

1. `inference-urgent-003`: rank `0.95`, release order 1
2. `inference-background-003`: rank `0.05`, release order 2

Both Pods emitted the versioned `EXECUTION_STARTED` marker, completed, and the
controller persisted a schema-v3 record with status `completed`. Its readiness
endpoint then reported `ready: true` with all observed runs reconciled.

The live test exposed a Kubernetes Python-client behavior where a JSON log line
was coerced into a Python dictionary string. Commit `a3722bd` fixes this by
reading raw log bytes and keeps strict JSON marker validation. The final result
record SHA-256 is
`6a3326759776aa4f52b78bb662747fc2238d33988526bd9e0157793c4a962923`.

## Local release gates

- dependency check: passed
- Ruff: passed
- locked article plan: 70 runs validated
- locked extended plan: 90 runs validated
- tests: 145 passed
- release artifacts: wheel and source distribution for `0.2.1` built
- Helm lint/render and Kubernetes server-side dry-run: passed on the deployment
  path used by this cluster
