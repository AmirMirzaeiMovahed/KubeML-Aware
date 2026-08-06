# KubeML-Aware

KubeML-Aware is a reproducible implementation of the ML-aware Kubernetes
scheduling

The maintained implementation, setup instructions, experiment contracts, and
validation reports live in
[`Project implementation/k8s-ai-scheduler`](Project%20implementation/k8s-ai-scheduler/README.md).

## Start here

1. Read the [implementation README](Project%20implementation/k8s-ai-scheduler/README.md).
2. Run the [local preflight](Project%20implementation/k8s-ai-scheduler/scripts/preflight.ps1).
3. Complete the [pre-cluster checklist](Project%20implementation/k8s-ai-scheduler/docs/PRE_CLUSTER_CHECKLIST.md).
4. Follow the [cluster runbook](Project%20implementation/k8s-ai-scheduler/run_on_cluster.md) only on a disposable target cluster.

Generated CSV, JSON, and plot files are intentionally not committed as source
evidence. A result is reportable only after the implementation's strict
Kubernetes evidence validation succeeds.
