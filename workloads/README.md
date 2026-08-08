# KubeML-Aware Workloads

This directory contains diverse synthetic ML training workloads for testing and evaluating the KubeML-Aware scheduler on single-node (Minikube) clusters.

## Directory Structure

```
workloads/
├── run-small/              # 3 pods - Fast convergence, quick experiments
├── run-medium/             # 5 pods - Varied characteristics, standard test
├── run-large/              # 8 pods - Stress test, scheduling pressure
├── run-heterogeneous/      # 5 pods - Mixed partition counts (P=1,2,4)
└── run-ablation/           # 4 pods - For reverse-order ablation study
```

## Workload Characteristics

Each workload pod is annotated with the 6 ML features used by the ranking algorithm:

| Feature | Annotation Key | Description |
|---------|----------------|-------------|
| **T** | `ml.scheduler/estimated-training-time` | Estimated training time (seconds) |
| **R** | `ml.scheduler/loss-reduction-rate` | Loss reduction rate per step |
| **M** | `ml.scheduler/matrix-size` | Matrix dimension (rows/cols) |
| **G** | `ml.scheduler/gradient-update-size` | Gradient size (MiB) |
| **C** | `ml.scheduler/checkpoint-interval` | Checkpoint frequency (steps) |
| **P** | `ml.scheduler/model-partitions` | Model partition count |

### Ranking Weights (from `scheduler/rank.py`)
- **T**: 0.40 (smaller is better - shorter training preferred)
- **R**: 0.35 (larger is better - faster convergence preferred)
- **M**: 0.20 (smaller is better - less compute preferred)
- **G**: 0.15 (smaller is better - less communication preferred)
- **C**: 0.10 (smaller is better - fewer checkpoints preferred)
- **P**: 0.05 (smaller is better - fewer partitions preferred)

## Run Scenarios

### 1. run-small (3 pods)
Quick validation runs with fast-converging workloads.
- Expected scheduling order: workload-1 > workload-3 > workload-2
- Best for: CI/CD, quick iteration

### 2. run-medium (5 pods)
Standard test with diverse characteristics including partitioned models.
- Mix of P=1 and P=2 workloads
- Tests ranking with varying matrix sizes and loss rates
- Best for: Standard benchmarking

### 3. run-large (8 pods)
Stress test with maximum scheduling pressure.
- 8 concurrent gated pods
- Wide range of training times (150-400s)
- Tests burst collection quiet period
- Best for: Performance evaluation

### 4. run-heterogeneous (5 pods)
Tests scheduler behavior with mixed partition counts.
- P=1, P=2, and P=4 workloads in same burst
- Validates partition-aware ranking (P weight = 0.05)
- Best for: Distributed training simulation

### 5. run-ablation (4 pods)
Designed for ablation study with `--reverse` flag.
- Identical to run-medium but smaller
- Run twice: normal and `--reverse` to verify ordering flip
- Best for: Paper ablation experiments

## Usage

### Deploy Scheduler (Production Mode)
```bash
cd Project\ implementation/k8s-ai-scheduler/deploy/helm/ml-ai-scheduler
helm install ml-ai-scheduler . \
  --namespace default \
  --create-namespace \
  -f values.yaml \
  --set mode=production \
  --set scheduler.name=default-scheduler \
  --set scheduler.gateName=ml.scheduler/release \
  --set scheduler.expectedCount=N \
  --set-string experiment.runId=RUN_ID \
  --set-string experiment.scenario=test \
  --set-string experiment.config=test \
  --set-string experiment.repetition=1 \
  --set image.repository=ml-ai-scheduler \
  --set image.tag=local \
  --set image.pullPolicy=Never
```

### Apply Workloads
```bash
# For run-small (3 pods)
kubectl apply -f workloads/run-small/

# For run-medium (5 pods)
kubectl apply -f workloads/run-medium/

# For run-large (8 pods)
kubectl apply -f workloads/run-large/

# For run-heterogeneous (5 pods)
kubectl apply -f workloads/run-heterogeneous/

# For run-ablation (4 pods)
kubectl apply -f workloads/run-ablation/
```

### Run Controller Once (CLI)
```bash
cd Project\ implementation/k8s-ai-scheduler
python -m scheduler.single_node_gate_controller \
  --scheduler-name default-scheduler \
  --namespace default \
  --gate-name ml.scheduler/release \
  --target-node minikube \
  --run-id RUN_ID \
  --expected-count N \
  --pacing-mode none \
  --once \
  --results ../results/single_node_gate_schedule_run.json
```

### With Adaptive Pacing
```bash
python -m scheduler.single_node_gate_controller \
  --scheduler-name default-scheduler \
  --namespace default \
  --gate-name ml.scheduler/release \
  --target-node minikube \
  --run-id RUN_ID \
  --expected-count N \
  --pacing-mode adaptive \
  --cpu-threshold 0.85 \
  --adaptive-hysteresis 0.05 \
  --max-wait 30 \
  --once
```

### Ablation Study (Reverse Order)
```bash
python -m scheduler.single_node_gate_controller \
  --scheduler-name default-scheduler \
  --namespace default \
  --gate-name ml.scheduler/release \
  --target-node minikube \
  --run-id run-ablation \
  --expected-count 4 \
  --pacing-mode none \
  --reverse \
  --once
```

## Expected Results

The controller produces JSON schedule records at:
- Default: `../results/single_node_gate_schedule_run.json`
- Custom: via `--results` flag

Each record contains:
```json
{
  "job_id": "workload-1",
  "order": 1,
  "rank": 0.723,
  "pod_uid": "...",
  "status": "execution_started",
  "release_time": 1234567890.123,
  "exec_start_time": 1234567895.456
}
```

## Metrics Endpoint
```bash
kubectl port-forward -n default svc/ml-ai-scheduler-ml-ai-scheduler 8080:8080
curl http://localhost:8080/metrics
```

Key metrics:
- `ml_scheduler_bursts_total` - Completed bursts
- `ml_scheduler_releases_total` - Gates removed
- `ml_scheduler_burst_jobs` - Pods per burst
- `ml_scheduler_cpu_utilization_ratio` - Current CPU (adaptive mode)
- `ml_scheduler_metrics_age_seconds` - Metrics freshness

## Requirements

- Minikube running (`minikube start`)
- Metrics server for adaptive pacing (`minikube addons enable metrics-server`)
- Docker image loaded: `minikube image load ml-ai-scheduler:local`
- RBAC permissions for pod status reads (included in Helm chart)

## Notes

- All workloads use `nginx` as placeholder - replace with actual trainer image
- The `EXECUTION_STARTED` marker must be emitted by trainer container logs
- For real experiments, use the synthetic trainer from `k8s/work_model.py`
- Quiet period default: 1.5s (adjust with `--quiet-period`)
- Burst timeout default: 120s (adjust with `--burst-timeout`)