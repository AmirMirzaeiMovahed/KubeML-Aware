# Inference and profiling extension

The proposal covers AI-task characteristics and scheduling outcomes; it is not
limited to training. The original article reproduction remains unchanged and
uses `T/R/M/G/C/P`. Inference and profiling are an explicitly separate project
extension and must not be reported as article behavior.

## Inference scheduling contract

Set `ml.scheduler/workload-kind: inference` and provide every annotation below:

| Annotation | Meaning | Rank direction |
|---|---|---|
| `ml.scheduler/latency-slo-ms` | End-to-end latency objective | Used to derive SLO pressure |
| `ml.scheduler/predicted-latency-ms` | Profiled p95 service latency | Smaller is more efficient |
| `ml.scheduler/request-rate-rps` | Observed demand | Larger is more urgent |
| `ml.scheduler/memory-mib` | Peak profiled memory | Smaller is more efficient |
| `ml.scheduler/cold-start-ms` | Maximum observed startup cost | Larger is more urgent |
| `ml.scheduler/priority` | Positive operator/business priority | Larger is more urgent |

The inference score is burst-relative. Its weights are versioned in
`scheduler/rank.py`: SLO pressure `0.30`, demand `0.25`, predicted latency
`0.15`, peak memory `0.10`, cold start `0.10`, and priority `0.10`. A burst that
mixes training and inference is rejected because the two policies do not share
comparable semantics.

## Produce annotations from measurements

Each input line is one JSON object:

```json
{"latency_ms":12.4,"requests":10,"duration_seconds":1.0,"memory_mib":96.2,"cold_start_ms":310.0}
```

Generate an auditable profile and ready-to-copy annotations:

```bash
python -m profiling.cli \
  --input samples.jsonl \
  --job-id classifier-v1 \
  --latency-slo-ms 100 \
  --priority 2 \
  --output classifier-profile.json
```

The profile uses linearly interpolated p95 latency, total observed request rate,
peak memory, and maximum cold-start cost. Invalid, missing, non-finite, zero, or
negative measurements fail closed.

## Knative serving path

`inference/service.py` exposes:

- `POST /v1/predict` with `{"instances": [[...]]}`;
- `GET /v1/profile` for rolling p50/p95, throughput, memory, and cold-start data;
- `GET /healthz` and `GET /readyz`.

Build `inference/Dockerfile`, load or push the immutable image, then apply
`deploy/knative/namespace.yaml` and `deploy/knative/inference-service.yaml`.
For the documented air-gapped/local-image manifest, import the image as
`registry.local/kubeml/ml-inference-service:0.2.1` and add `registry.local` to
Knative's `registries-skipping-tag-resolving`; production registries should use
an immutable digest instead.
The France deployment keeps Kourier as a `ClusterIP` to avoid competing with
the existing ingress controller and VPN listeners. Its reviewed ingress bridge
is `deploy/knative/france-ingress.yaml`.
The Knative Service scales from zero to at most two replicas by default. Keep
the scheduler controller as a normal Kubernetes Deployment; it is a long-lived
reconciler, while Knative is used for the request-driven inference workload.

The measured France deployment, image digests, autoscaling result, and live
scheduling-gate output are recorded in `FRANCE_DEPLOYMENT_EVIDENCE.md`.
