# Pre-cluster release checklist

This is the go/no-go checklist immediately before touching the target
Kubernetes cluster. Checked items are limited to validations that can be
performed locally; server-only items remain unchecked until their command
output is captured.

## A. Repository and local quality gates

- [x] `python scripts/build_release.py` produces a clean wheel, standard sdist,
      and `dist/SHA256SUMS.txt`; the archives are ignored by Git and were
      verified locally. Copy them off-host before deployment.
- [x] Direct dependencies are exactly pinned and transitive resolution is
      frozen in `constraints.txt` for the release candidate.
- [x] Python version contract is `>=3.11,<3.13`.
- [x] Trainer and scheduler Dockerfiles run as UID/GID 10001.
- [x] Generated trainer Pods disable API tokens/service links, enforce
      non-root/read-only/no-capabilities/RuntimeDefault security, and mount a
      bounded writable `/tmp`.
- [x] Dockerfiles use versioned application dependencies and no `latest` tag.
- [x] Helm provides production, fixed-run reproduction, and dynamic-matrix
      reproduction values plus a values JSON Schema.
- [x] Chart contains ConfigMap, ServiceAccount, RBAC, Deployment, Service,
      probes, resource settings, security contexts, NetworkPolicy, optional PDB,
      optional result PVC, and optional ServiceMonitor.
- [x] Chart intentionally omits Ingress and HPA.
- [x] Repository contains PowerShell local and Linux server runbooks/scripts.
- [x] Linux scripts pass `bash -n`; PowerShell scripts pass parser validation.
- [x] `python -m compileall -q scheduler workload k8s results sim experiments`
      passes using the release Python.
- [x] Exact dependency versions import successfully and `python -m pip check`
      reports no broken requirements in the prepared local environment.
- [x] `pyproject.toml`, `Chart.yaml`, and all four plain values files parse.
- [x] `python -m pytest` passes: 121 passed, no skipped tests (local release
      candidate validation).
- [x] `python -m ruff check .` passes with the repository-owned rule set.
- [x] CI runs locked installs, lint, compile, tests, and plan validation on
      Python 3.11 and 3.12, then builds and installs the release wheel.
- [x] Published Table I/Figures 2-4 values are transcribed in a versioned
      package resource and the analyzer emits diagnostic observed deltas.
- [x] `helm lint deploy/helm/ml-ai-scheduler` passes with Helm 4.2.0.
- [x] Production, fixed reproduction, and dynamic matrix Helm profiles render
      without errors and pass strict Kubernetes 1.36 offline schema validation.
- [x] Rendered YAML has been inspected locally for mutable images, unexpected
      namespaces, cluster permissions, and secrets.
- [ ] Both images build successfully from the repository root.
- [ ] Local image import/smoke commands pass.
- [ ] Image vulnerability/SBOM policy checks pass, if required by the target.

## B. Required user/environment facts

- [ ] Target server operating system and shell are recorded.
- [ ] Kubernetes distribution/version is recorded and supported.
- [ ] Exact `kubectl` context is recorded.
- [ ] Registry host/project and authentication method are confirmed.
- [ ] Namespace is approved.
- [ ] Result retention location and backup owner are confirmed.
- [ ] Reproduction target node is dedicated, Ready, uncordoned, and reviewed.
- [ ] Production, fixed reproduction, or reproduction-matrix profile is
      explicitly selected.
- [ ] Article-only 70 runs or extended 90 runs is explicitly selected.

## C. Server-only read-only validation

- [ ] `scripts/server-preflight.sh` succeeds against `EXPECTED_CONTEXT`.
- [ ] Server Python 3.11/3.12 venv installs pinned dependencies; imports,
      `pip check`, tests, and 70/90 plan expansion succeed.
- [ ] Kubernetes API `/readyz?verbose` succeeds.
- [ ] Node capacity, readiness, taints, pressure, and architecture are recorded.
- [ ] Helm and container tooling versions are captured.
- [ ] `metrics.k8s.io` is available and fresh if adaptive pacing is selected.
- [ ] `helm lint` succeeds on the server.
- [ ] Production profile passes `helm template` on the server.
- [ ] Reproduction profile passes `helm template` with a real target node.
- [ ] Dynamic matrix profile passes `helm template` with a real target node.
- [ ] All selected rendered profiles pass `kubectl apply --dry-run=server`.

## D. Registry and image validation

- [ ] Non-`latest` scheduler and trainer tags are pushed.
- [ ] Registry-provided sha256 digests are captured.
- [ ] Base image reference/digest and application image digests are recorded in
      experiment metadata.
- [ ] Target nodes can pull both images.
- [ ] Pull-secret or workload-identity method is tested without committing
      credentials.
- [ ] Generated trainer Pods reference the reviewed pull Secret when the
      registry is private.
- [ ] Scheduler image imports its runtime packages as non-root.
- [ ] Trainer image imports NumPy and respects controlled BLAS thread variables.
- [ ] Trainer prewarm attestation reports exactly one actual BLAS thread for
      every detected BLAS library on the target node.

## E. Helm installation and security validation

- [ ] `helm upgrade --install --atomic --wait` completes.
- [ ] Exactly one scheduler replica is configured.
- [ ] Scheduler Pod is Running and Ready with no restart loop.
- [ ] Scheduler container runs as UID/GID 10001.
- [ ] Root filesystem is read-only and all Linux capabilities are dropped.
- [ ] CPU/memory requests and limits are present for the scheduler.
- [ ] Gated production workload Pods have requests/limits and an explicit
      execution-container contract when more than one container exists.
- [ ] `/livez` returns HTTP 200.
- [ ] `/readyz` returns HTTP 200 only when the Kubernetes client is ready.
- [ ] `/metrics` returns Prometheus text.
- [ ] NetworkPolicy ingress and API-port-only egress are enforced by the cluster CNI;
      production API-server CIDRs are configured when the network is known.
- [ ] Production uses a Bound results PVC and a PDB; the Service publishes
      diagnostic endpoints even while controller readiness is false.
- [ ] Reproduction ServiceAccount can create `pods/binding`, get `pods/log`,
      and read the selected Node.
- [ ] Production ServiceAccount can patch namespace Pods but cannot create
      `pods/binding`.
- [ ] Dynamic adaptive capability grants read-only Node Metrics access; the
      base fixed profile does not receive it.
- [ ] Matrix operator can create `pods/exec` to archive scheduler records.
- [ ] No unresolved Warning events or repeated 401/403/409/5xx errors exist.

## F. Minimal functional smoke tests

- [ ] A three-job deterministic reproduction burst is collected exactly once.
- [ ] All three ranks match the offline expected order.
- [ ] All three Pods bind only to the explicit target node.
- [ ] Every Pod emits a valid execution-start marker.
- [ ] The result record contains all expected Pod IDs and ordered timestamps.
- [ ] Authoritative scheduler rank/order/bind/pacing evidence is archived and
      embedded before workload cleanup.
- [ ] A malformed annotation fails without binding the malformed Pod.
- [ ] A missing Pod causes burst timeout/incomplete-run failure.
- [ ] An image-pull failure invalidates the run.
- [ ] Scheduler Pod replacement during a partial production run resumes from
      the PVC, matches Pod UIDs, and does not duplicate a release.
- [ ] Corrupt/failed persisted state makes `/readyz` fail and is not replayed.
- [ ] A three-job production-gate burst removes gates in ranked order and the
      default scheduler selects nodes.
- [ ] Missing/stale metrics fail closed in adaptive mode.
- [ ] Adaptive Metrics API timestamps advance within the configured max-wait
      safety margin.

## G. Experiment readiness

- [ ] Seeds and paired manifests are archived before submission.
- [ ] Expanded 70/90 plan SHA-256 lock is reviewed and archived.
- [ ] Each workload directory is generator-owned and either newly created or
      passes exact RunSpec/jobs/manifest validation before reuse.
- [ ] Every run has one unique `ml.scheduler/run-id`.
- [ ] Scenario, config, and repetition labels match the matrix.
- [ ] No two reproduction runs execute concurrently on the target node.
- [ ] Minikube profile/status evidence proves the registered Docker driver.
- [ ] The exact trainer digest is prewarmed and its runtime image ID archived.
- [ ] Every run has at least the registered 30-second clean cooldown, three
      clean polls, clear Node pressure conditions, and scheduler UID continuity.
- [ ] Cluster/image/tool metadata is captured for each run.
- [ ] Requested trainer digest, runtime image IDs, scheduler digest, target
      node, context, artifact hashes, and cluster snapshot are present.
- [ ] Collector rejects missing, failed, duplicate, or timestamp-invalid jobs.
- [ ] Mean ECDF and IQR use per-run distributions, not pooled jobs.
- [ ] Article-only and adaptive-extension aggregates are separate.
- [ ] Raw results are backed up before cleanup or Helm rollback.

## H. Rollback readiness

- [ ] A known-good Helm revision/image digest exists.
- [ ] Operator has permission to run `helm rollback`.
- [ ] Label-scoped workload preview/cleanup commands are reviewed.
- [ ] Raw result preservation is tested before Pod deletion.
- [ ] It is understood that controller rollback cannot undo already bound or
      released workloads.

## Go/no-go rule

Do not start the 70-run or 90-run matrix until every applicable item in A–F is
checked. Any incomplete burst, failed Pod, stale adaptive metric, missing
timestamp, scheduler restart race, or environment drift is a **no-go** for
that run. Preserve evidence, identify the cause, and rerun with a new run ID.
