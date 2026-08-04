# Kubernetes build, deployment, smoke test, and rollback

This runbook starts at the point where local source/tests are complete. It does
not claim deployment success until every server-only gate in
`docs/PRE_CLUSTER_CHECKLIST.md` is checked with captured output.

## 1. Required environment facts

Record these before changing the cluster:

```text
Server OS and shell:
Kubernetes distribution and version:
kubectl context:
Container build tool and version:
Registry host/project:
Registry authentication method:
Namespace:
Single reproduction target node:
Storage location for raw experiment results:
```

The packaged chart supports Kubernetes 1.30+ for every profile (the production
Scheduling Readiness API establishes the chart-wide lower bound).
Reproduction manual binding should run only in a dedicated, single-node
experimental cluster.

## 2. Read-only server preflight

Prepare the pinned Python runtime on the Linux server from the repository
root. This changes only the repository-local virtual environment:

```bash
python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), sys.version; print(sys.version)'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest
```

Do not continue unless the full suite passes. Then run the read-only cluster
preflight:

```bash
export EXPECTED_CONTEXT='replace-with-exact-context'
chmod +x scripts/*.sh
./scripts/server-preflight.sh | tee preflight-output.txt
```

Review `kubectl get nodes -o wide`. For reproduction, select a Ready worker and
record its exact name:

```bash
export TARGET_NODE='replace-with-ready-node-name'
kubectl get node "$TARGET_NODE" -o jsonpath='{.metadata.name}{" ready="}{.status.conditions[?(@.type=="Ready")].status}{" unschedulable="}{.spec.unschedulable}{"\n"}'
kubectl describe node "$TARGET_NODE"
```

Stop if the node is not Ready, is cordoned, has unexplained taints, or cannot
provide the article's isolated 4 CPU / 8 GiB setup.

For a disposable Minikube reproduction cluster, creation is an explicit
state-changing operation:

```bash
minikube start --driver=docker --cpus=4 --memory=8192
minikube addons enable metrics-server  # only needed by adaptive runs
```

Do not run this command against an existing shared Minikube profile without
first confirming its impact.

## 3. Build, smoke-test, and push images

Use immutable version tags—never `latest`. A digest-pinned base image is
preferred; obtain the digest from the trusted registry rather than copying an
unverified value from documentation.

```bash
export REGISTRY='registry.example.com/your-project'
export REGISTRY_HOST='registry.example.com'
export TAG='0.1.0'
export PYTHON_BASE_IMAGE='python:3.11.13-slim-bookworm'

docker login "$REGISTRY_HOST"
docker build --pull \
  --build-arg "PYTHON_BASE_IMAGE=$PYTHON_BASE_IMAGE" \
  --build-arg "APP_VERSION=$TAG" \
  -f scheduler/Dockerfile -t "$REGISTRY/ml-aware-scheduler:$TAG" .
docker build --pull \
  --build-arg "PYTHON_BASE_IMAGE=$PYTHON_BASE_IMAGE" \
  --build-arg "APP_VERSION=$TAG" \
  -f k8s/Dockerfile -t "$REGISTRY/ml-sim-job:$TAG" .

docker run --rm --entrypoint python "$REGISTRY/ml-aware-scheduler:$TAG" \
  -c "import kubernetes,prometheus_client,yaml; import scheduler.rank; print('scheduler image OK')"
docker run --rm --entrypoint python "$REGISTRY/ml-sim-job:$TAG" \
  -c "import numpy; print('trainer image OK', numpy.__version__)"

docker push "$REGISTRY/ml-aware-scheduler:$TAG"
docker push "$REGISTRY/ml-sim-job:$TAG"
docker buildx imagetools inspect "$REGISTRY/ml-aware-scheduler:$TAG"
docker buildx imagetools inspect "$REGISTRY/ml-sim-job:$TAG"
```

Record both registry digests. The scheduler digest is passed to Helm; the
trainer digest/reference is passed to workload generation.

```bash
export IMAGE_DIGEST='sha256:replace-with-registry-scheduler-digest'
export TRAINER_DIGEST='sha256:replace-with-registry-trainer-digest'
```

## 4. Registry pull credentials

Prefer node/workload identity or an externally managed pull secret. If the
cluster requires a Kubernetes pull secret, create it from an authenticated
Docker config without putting a password in source or Helm values:

```bash
export NAMESPACE='ai-scheduler'
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create secret generic registry-credentials \
  --type=kubernetes.io/dockerconfigjson \
  --from-file=.dockerconfigjson="$HOME/.docker/config.json" \
  --dry-run=client -o yaml | kubectl apply -f -
```

This copies the Docker config into the cluster. Confirm that this complies with
your registry policy and that the file contains usable auth rather than only a
desktop credential-helper reference. Do not commit the generated Secret.
The remaining examples assume this private-registry Secret exists; for a
public registry, omit every `--image-pull-secret registry-credentials` and
`imagePullSecrets[0]` argument.

## 5. Render and validate before applying

Production profile:

```bash
helm lint deploy/helm/ml-ai-scheduler
helm template ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace "$NAMESPACE" \
  --values deploy/helm/ml-ai-scheduler/values-production.yaml \
  --set-string "image.repository=$REGISTRY/ml-aware-scheduler" \
  --set-string "image.tag=$TAG" \
  --set-string "image.digest=$IMAGE_DIGEST" \
  --set-string 'imagePullSecrets[0].name=registry-credentials' \
  > rendered-production.yaml
kubectl apply --dry-run=server -f rendered-production.yaml
```

Reproduction profile requires a validated target node and a unique run ID:

```bash
export RUN_ID='paper-48-half-custom-baseline-r1'
helm template ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace "$NAMESPACE" \
  --values deploy/helm/ml-ai-scheduler/values-reproduction.yaml \
  --set-string "image.repository=$REGISTRY/ml-aware-scheduler" \
  --set-string "image.tag=$TAG" \
  --set-string "image.digest=$IMAGE_DIGEST" \
  --set-string 'imagePullSecrets[0].name=registry-credentials' \
  --set-string "scheduler.targetNode=$TARGET_NODE" \
  --set-string "experiment.runId=$RUN_ID" \
  > rendered-reproduction.yaml
kubectl apply --dry-run=server -f rendered-reproduction.yaml
```

Dynamic matrix profile (the target node and immutable digest are mandatory):

```bash
helm template ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace "$NAMESPACE" \
  --values deploy/helm/ml-ai-scheduler/values-reproduction-matrix.yaml \
  --set-string "image.repository=$REGISTRY/ml-aware-scheduler" \
  --set-string "image.digest=$IMAGE_DIGEST" \
  --set-string 'imagePullSecrets[0].name=registry-credentials' \
  --set-string "scheduler.targetNode=$TARGET_NODE" \
  --set-string 'imagePullSecrets[0].name=registry-credentials' \
  > rendered-reproduction-matrix.yaml
kubectl apply --dry-run=server -f rendered-reproduction-matrix.yaml
```

Run the repository's strict offline schema check as well:

```bash
python scripts/validate_manifests.py \
  rendered-production.yaml rendered-reproduction.yaml \
  rendered-reproduction-matrix.yaml \
  --kubernetes-version 1.36.0
```

Inspect rendered files for unexpected namespaces, cluster-wide permissions,
mutable image tags, and secret values. They contain references only and may be
deleted after review.

## 6. Install production profile

```bash
export MODE='production'
export IMAGE_DIGEST='sha256:replace-with-registry-scheduler-digest'
./scripts/deploy.sh
```

If a pull secret is required, add it with a reviewed local values file (ignored
by Git):

```bash
cp deploy/helm/ml-ai-scheduler/local-values.example.yaml values.local.yaml
```

```yaml
imagePullSecrets:
  - name: registry-credentials
# Production enables persistence because restart recovery requires order state
# across Pod replacement. Confirm a default dynamic StorageClass exists, or set
# persistence.existingClaim before installing.
persistence:
  enabled: true
  size: 1Gi
# Recommended: restrict the API-port egress rule to the observed API endpoint.
networkPolicy:
  egress:
    apiServerCIDRs:
      - 10.96.0.1/32  # replace; do not copy this example blindly
```

Then install directly:

```bash
helm upgrade --install ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace "$NAMESPACE" --create-namespace \
  --values deploy/helm/ml-ai-scheduler/values-production.yaml \
  --values values.local.yaml \
  --set-string "image.repository=$REGISTRY/ml-aware-scheduler" \
  --set-string "image.tag=$TAG" \
  --set-string "image.digest=$IMAGE_DIGEST" \
  --atomic --wait --timeout 5m
```

## 7. Install reproduction profile

Start the scheduler before submitting workload Pods:

```bash
export MODE='reproduction'
export RUN_ID='paper-48-half-custom-baseline-r1'
helm upgrade --install ml-ai-scheduler deploy/helm/ml-ai-scheduler \
  --namespace "$NAMESPACE" --create-namespace \
  --values deploy/helm/ml-ai-scheduler/values-reproduction.yaml \
  --set-string "image.repository=$REGISTRY/ml-aware-scheduler" \
  --set-string "image.tag=$TAG" \
  --set-string "image.digest=$IMAGE_DIGEST" \
  --set-string "scheduler.targetNode=$TARGET_NODE" \
  --set-string "experiment.runId=$RUN_ID" \
  --set 'scheduler.expectedCount=48' \
  --atomic --wait --timeout 5m
```

The chart starts a long-running process around one fixed run contract. After
the burst completes it remains available but only watches that contract. Use a
unique run label, preserve `/results/schedule-<run-id>.json`, and never submit a
second burst with the same run ID.

For the automated 70/90-run matrix, use the dedicated dynamic profile. It has
no fixed run filter, enables the optional Node Metrics RBAC capability, and
stores one scheduler record per run:

```bash
export MODE='reproduction-matrix'
export IMAGE_PULL_SECRET='registry-credentials'  # unset for a public registry
./scripts/deploy.sh
```

The runner verifies that this live Deployment is Ready, single-replica,
digest-pinned, uses `scheduler.custom_scheduler`, has no fixed `run-id` or
expected count, targets `$TARGET_NODE`, exposes `/readyz`, and has the required
RBAC before it creates any workload Pod.

## 8. RBAC verification

Determine the generated ServiceAccount and test effective authorization:

```bash
export SA='ml-ai-scheduler-ml-ai-scheduler'
kubectl auth can-i get pods -n "$NAMESPACE" \
  --as="system:serviceaccount:$NAMESPACE:$SA"
kubectl auth can-i list nodes \
  --as="system:serviceaccount:$NAMESPACE:$SA"
```

For reproduction the following must return `yes`:

```bash
kubectl auth can-i create pods/binding -n "$NAMESPACE" \
  --as="system:serviceaccount:$NAMESPACE:$SA"
kubectl auth can-i get pods/log -n "$NAMESPACE" \
  --as="system:serviceaccount:$NAMESPACE:$SA"
```

For production, patching Pods must return `yes`; creating bindings should
return `no`:

```bash
kubectl auth can-i patch pods -n "$NAMESPACE" \
  --as="system:serviceaccount:$NAMESPACE:$SA"
kubectl auth can-i create pods/binding -n "$NAMESPACE" \
  --as="system:serviceaccount:$NAMESPACE:$SA"
```

Adaptive mode additionally requires `get` on `nodes.metrics.k8s.io`.

The user running the automated matrix must be able to read scheduler evidence
through `pods/exec` (this permission is not granted to the scheduler
ServiceAccount):

```bash
kubectl auth can-i create pods/exec -n "$NAMESPACE"
```

The command must return `yes` before `experiments.run_cluster --execute`.

## 9. Rollout and smoke tests

```bash
kubectl -n "$NAMESPACE" rollout status \
  deployment/ml-ai-scheduler-ml-ai-scheduler --timeout=180s
kubectl -n "$NAMESPACE" get pods -o wide
kubectl -n "$NAMESPACE" logs \
  deployment/ml-ai-scheduler-ml-ai-scheduler --all-containers --tail=200
kubectl -n "$NAMESPACE" get events --sort-by=.metadata.creationTimestamp

export SERVICE='ml-ai-scheduler-ml-ai-scheduler'
./scripts/smoke-test.sh
```

Expected outcomes: one Ready scheduler Pod, HTTP 200 from `/livez` and
`/readyz`, Prometheus text from `/metrics`, no repeated authorization errors,
and no unresolved Warning events.

## 10. Workload submission and run acceptance

Generate each run into a new, empty directory. Use all four labels exactly:

- `ml.scheduler/run-id`
- `ml.scheduler/scenario`
- `ml.scheduler/config`
- `ml.scheduler/repetition`

Production workloads also carry the scheduling gate
`ml.scheduler/release`. Reproduction workloads use
`schedulerName: ml-aware-scheduler`; default baselines use
`schedulerName: default-scheduler` without a gate.

Example reproduction custom run (the output path must not already contain
files):

```bash
export TRAINER_DIGEST='sha256:replace-with-registry-trainer-digest'
export SEED='1001'
export RUN_ID='paper-48-half-custom-baseline-r1'
export RUN_DIR="workload/runs/$RUN_ID"
python -m workload.generate_workload \
  --n 48 --load half --seed "$SEED" \
  --out "$RUN_DIR" --namespace "$NAMESPACE" \
  --run-id "$RUN_ID" --scenario 48-half --repetition 1 \
  --custom-config custom-baseline \
  --scheduler-name ml-aware-scheduler \
  --image-pull-secret registry-credentials \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST"
kubectl -n "$NAMESPACE" apply -f "$RUN_DIR/pods_custom"
```

For its paired default baseline, use the same seed and feature profile but a
new output directory and unique default run ID, then submit `pods_default`.
This avoids result-key collisions while preserving deterministic features:

```bash
export RUN_ID='paper-48-half-default-r1'
export RUN_DIR="workload/runs/$RUN_ID"
python -m workload.generate_workload \
  --n 48 --load half --seed "$SEED" \
  --out "$RUN_DIR" --namespace "$NAMESPACE" \
  --run-id "$RUN_ID" --scenario 48-half --repetition 1 \
  --custom-config custom-baseline \
  --scheduler-name ml-aware-scheduler \
  --image-pull-secret registry-credentials \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST"
kubectl -n "$NAMESPACE" apply -f "$RUN_DIR/pods_default"
```

For the production gate profile, generate the custom set with
`--scheduler-name default-scheduler --scheduling-gate ml.scheduler/release`.
This automatically adds the execution-container annotation and production
CPU/memory bounds. Never add the gate to an unpaired default baseline.

Before submission:

```bash
kubectl -n "$NAMESPACE" get pods -l "ml.scheduler/run-id=$RUN_ID"
```

This must return no Pods. After applying the generated manifests:

```bash
kubectl -n "$NAMESPACE" get pods -l "ml.scheduler/run-id=$RUN_ID" -w
kubectl -n "$NAMESPACE" get pods -l "ml.scheduler/run-id=$RUN_ID" \
  -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,NODE:.spec.nodeName
kubectl -n "$NAMESPACE" logs \
  deployment/ml-ai-scheduler-ml-ai-scheduler --tail=500
```

Reject the run if the Pod count differs from the expected count, any Pod fails,
timestamps are missing/out of order, the scheduler reports an incomplete
burst, or metadata differs within the run.

After all expected Pods succeed, collect strict run metrics (set `CONFIG` to
the exact manifest label, for example `custom-baseline` or `default`):

```bash
export CONFIG='custom-baseline'
mkdir -p results/runs
python -m results.metrics_collector \
  --namespace "$NAMESPACE" \
  --label-selector "ml.scheduler/run-id=$RUN_ID" \
  --expected-jobs 48 --run-id "$RUN_ID" \
  --scenario 48-half --config "$CONFIG" \
  --repetition 1 --seed "$SEED" \
  --context "$EXPECTED_CONTEXT" \
  --out "results/runs/$RUN_ID.json"
```

The command must exit nonzero for a partial or inconsistent run. Archive its
JSON plus `jobs.json`, scheduler record, manifests, and server metadata as one
immutable result bundle.

Preserve the scheduler order record before upgrading or deleting its Pod:

```bash
export SCHEDULER_POD="$(kubectl -n "$NAMESPACE" get pod \
  -l app.kubernetes.io/component=scheduler \
  -o jsonpath='{.items[0].metadata.name}')"
kubectl -n "$NAMESPACE" exec "$SCHEDULER_POD" -- \
  cat "/results/schedule-$RUN_ID.json" > "$RUN_ID-schedule-run.json"
```

The shell receives no secret. Check that the resulting JSON is non-empty and
schema-valid before any cleanup.

## 11. Paper and extended matrices

Run in randomized block order while retaining deterministic workload pairing.
Never execute two experimental runs concurrently on the same reproduction
node.

The runner itself submits each workload as a seed-controlled concurrent API
burst. Keep the default `--environment-profile article-exact`: it validates a
single Minikube node, four CPU cores, approximately 8 GiB memory, healthy node
conditions, and the absence of active non-system workloads before every run.
`--environment-profile record-only` is reserved for development smoke tests
and makes the resulting evidence ineligible for article claims. A run also
fails when its API-server Pod creation spread exceeds five seconds; tighten
`--max-submission-spread-seconds` only after observing a stable pilot.

**Article-only matrix: 70 runs**

| Block | Scenarios | Configurations | Repetitions | Runs |
|---|---:|---:|---:|---:|
| Pacing | `48-half-pacing` | default, custom-baseline (0s), custom-delay-1s, custom-delay-2s, custom-delay-5s | 5 | 25 |
| Main | `12-normal`, `48-normal`, `48-half` | default, custom-baseline, reversed | 5 | 45 |
| Total | | | | 70 |

**Extended matrix: 90 runs**

Add adaptive to the pacing block (5 runs) and to each main scenario (15
runs). These 20 runs are RFAP extension evidence and must not be attributed to
the article.

Validate both plans without contacting Kubernetes:

```bash
python -m experiments.run_cluster \
  --plan-out experiments/locks/article-70.json
python -m experiments.run_cluster --include-adaptive \
  --plan-out experiments/locks/extended-90.json
```

The first command must report 70 runs; the second must report 90. These JSON
files are immutable plan locks: execution refuses to start if the registered
YAML or expanded order differs. Repetitions are canonically labelled `0..4`
(five independent repetitions).

Materialize all 70 workloads for review without contacting the cluster:

```bash
python -m experiments.run_cluster --materialize \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST" \
  --image-pull-secret registry-credentials \
  --namespace "$NAMESPACE" \
  --work-dir workload/cluster_runs/article \
  --plan-out experiments/locks/article-70.json
```

After the three-job smoke tests and all pre-cluster gates pass, choose exactly
one matrix. Start with `--limit 1`, inspect/accept that result, then resume the
same output set:

```bash
python -m experiments.run_cluster --execute --limit 1 \
  --context "$EXPECTED_CONTEXT" --namespace "$NAMESPACE" \
  --target-node "$TARGET_NODE" \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST" \
  --image-pull-secret registry-credentials \
  --work-dir workload/cluster_runs/article \
  --results-dir results/cluster/article \
  --plan-out experiments/locks/article-70.json

python -m experiments.run_cluster --execute --resume \
  --context "$EXPECTED_CONTEXT" --namespace "$NAMESPACE" \
  --target-node "$TARGET_NODE" \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST" \
  --image-pull-secret registry-credentials \
  --work-dir workload/cluster_runs/article \
  --results-dir results/cluster/article \
  --plan-out experiments/locks/article-70.json
```

For the 90-run extended plan, add `--include-adaptive` to both execution
commands, use `experiments/locks/extended-90.json`, and use separate `extended`
work/results directories. Before creating a Pod, the runner verifies Metrics
API RBAC, freshness, and an advancing sample timestamp within the configured
adaptive deadline. Do not run the 70-run plan and then blindly run all 90
again; the extended plan already contains the article's 70 runs.

Exact extended execution commands:

```bash
python -m experiments.run_cluster --include-adaptive --materialize \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST" \
  --image-pull-secret registry-credentials \
  --namespace "$NAMESPACE" \
  --work-dir workload/cluster_runs/extended \
  --plan-out experiments/locks/extended-90.json

python -m experiments.run_cluster --include-adaptive --execute --limit 1 \
  --context "$EXPECTED_CONTEXT" --namespace "$NAMESPACE" \
  --target-node "$TARGET_NODE" \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST" \
  --image-pull-secret registry-credentials \
  --work-dir workload/cluster_runs/extended \
  --results-dir results/cluster/extended \
  --plan-out experiments/locks/extended-90.json

python -m experiments.run_cluster --include-adaptive --execute --resume \
  --context "$EXPECTED_CONTEXT" --namespace "$NAMESPACE" \
  --target-node "$TARGET_NODE" \
  --image "$REGISTRY/ml-sim-job@$TRAINER_DIGEST" \
  --image-pull-secret registry-credentials \
  --work-dir workload/cluster_runs/extended \
  --results-dir results/cluster/extended \
  --plan-out experiments/locks/extended-90.json
```

Do not pass `--scheduling-gate` to this registered matrix. The runner rejects
it because it would turn the configurations labelled `default` into gated,
non-baseline workloads. Existing reviewed manifests are reused only after an
exact workload/manifest contract comparison; any drift fails closed. Every
custom run archives `results/cluster/<track>/schedules/<run-id>.json` and
embeds the validated scheduler record in its run result before Pod cleanup.

Strict final analysis (do not use `--allow-partial` for report results):

```bash
python -m experiments.analyze \
  --runs-dir results/cluster/article/runs \
  --output-dir results/cluster/article/analysis

python -m experiments.analyze --include-adaptive \
  --runs-dir results/cluster/extended/runs \
  --output-dir results/cluster/extended/analysis
```

## 12. Diagnostics

```bash
kubectl -n "$NAMESPACE" describe deployment ml-ai-scheduler-ml-ai-scheduler
kubectl -n "$NAMESPACE" describe pods -l app.kubernetes.io/component=scheduler
kubectl -n "$NAMESPACE" logs \
  deployment/ml-ai-scheduler-ml-ai-scheduler --previous --tail=300
kubectl -n "$NAMESPACE" get events --sort-by=.metadata.creationTimestamp
kubectl get apiservice v1beta1.metrics.k8s.io
kubectl get --raw='/apis/metrics.k8s.io/v1beta1/nodes'
kubectl -n "$NAMESPACE" get role,rolebinding
kubectl get clusterrole,clusterrolebinding \
  -l app.kubernetes.io/instance=ml-ai-scheduler
```

Interpret common failures:

| Symptom | Root-cause check | Smallest safe response |
|---|---|---|
| `ImagePullBackOff` | Pod Events and image reference/digest | Correct registry auth/reference; do not change pull policy to hide it |
| `Forbidden` | Exact verb/resource in logs and `kubectl auth can-i` | Add only the missing scoped permission after review |
| Probe failures | Port-forward and scheduler logs | Fix process/endpoint; do not disable probes as a first response |
| Pending reproduction Pods | scheduler logs, expected run labels, target node | Correct run contract or node; do not manually bind individual Pods |
| Adaptive metric stale/unavailable | APIService and raw metrics response | Reject adaptive run or use fixed/none; never treat missing metrics as zero |
| Incomplete results | expected vs observed Pod IDs and logs | Reject and rerun from a clean, unique run ID |

## 13. Rollback and scoped cleanup

Inspect revision history before rollback:

```bash
helm history ml-ai-scheduler -n "$NAMESPACE"
helm rollback ml-ai-scheduler <known-good-revision> \
  -n "$NAMESPACE" --wait --timeout 5m
kubectl -n "$NAMESPACE" rollout status \
  deployment/ml-ai-scheduler-ml-ai-scheduler --timeout=180s
```

Rollback changes the controller only; it does not undo already released or
bound workloads. Preserve raw logs/results first. To remove one failed
experiment safely, preview and then delete by its unique run ID:

```bash
kubectl -n "$NAMESPACE" get pods -l "ml.scheduler/run-id=$RUN_ID" -o name
kubectl -n "$NAMESPACE" delete pods -l "ml.scheduler/run-id=$RUN_ID" \
  --wait=true --timeout=120s
```

Do not delete the namespace or all Pods as routine cleanup. Uninstalling the
release removes scheduler objects but does not remove independently created
workload Pods. A chart-created results PVC carries
`helm.sh/resource-policy: keep` and is deliberately retained; delete it only
after verifying an external backup:

```bash
helm uninstall ml-ai-scheduler -n "$NAMESPACE"
kubectl -n "$NAMESPACE" get pvc ml-ai-scheduler-ml-ai-scheduler
# Only after the retained data is verified elsewhere:
kubectl -n "$NAMESPACE" delete pvc ml-ai-scheduler-ml-ai-scheduler
```
