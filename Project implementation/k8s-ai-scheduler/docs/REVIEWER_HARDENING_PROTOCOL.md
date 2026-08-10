# Reviewer-hardening experiment

This protocol answers the main causal questions that the five-pair pilot could
not answer. It is deliberately committed as **registered, not executed**. A
paper may describe these arms as results only after the generated archive is
complete and validated.

## Arms

| Arm | Rank | FastPath | Tail | Purpose |
|---|---|---:|---:|---|
| `baseline` | FIFO | off | off | synchronized Kubernetes baseline |
| `duration-only` | T only | on | off | SPT/weighted-SJF baseline |
| `kubeml` | six features | on | on | deployed policy |
| `six-feature-no-tail` | six features | on | off | tail ablation |
| `six-feature-no-fastpath` | six features | off | off | FastPath ablation |
| `reversed` | reverse six-feature | off | off | ordering ablation |

The controller reads rank, tail, and FastPath policy from per-run annotations;
all arms still use `default-scheduler` for placement. Per-run settings are
validated for consistency across every Pod in a burst and are persisted in the
gate record.

## Command

Run on a dedicated Kubernetes node after deploying this branch's controller:

```bash
python scripts/run_training_fastpath_pilot.py \
  --reviewer-matrix \
  --output-root /root/kubeml-reviewer-matrix
```

The registered matrix is 30 repetitions × 6 arms = 180 completed bursts. The
driver deterministically shuffles arm order inside every repetition, reuses the
same workload seed across arms, waits for the node to cool, and aborts if a
protected service becomes unhealthy. Based on the archived five-pair pilot,
budget roughly 4–6 hours on the same class of node; this is an operational
estimate, not a measured outcome for the new matrix.

## Current evidence boundary

The repository already includes five real paired FIFO-vs-deployed-policy runs.
Those runs support the reported pilot result only. They do not contain a live
duration-only, no-tail, no-FastPath, or reversed arm. The structural diagnostic
under `paper/data/` quantifies order overlap and records the FastPath branch,
but it is not a substitute for this live matrix.
