# Paper artifact map

`main.tex` is the source of the reviewer-hardened manuscript and `main.pdf` is
the compiled seven-page IEEE Computer Society layout.

## Evidence used by the manuscript

- `data/paired_metrics.csv` and `data/results_summary.json`: metrics from five
  real paired Kubernetes repetitions.
- `data/timeline_r0_*.csv`: application-level intervals for the first paired
  repetition shown in Fig. 4.
- `data/pilot_feature_matrix.csv`: sanitized feature and rank matrix extracted
  from the 60 archived real-cluster manifests.
- `data/rank_policy_diagnostics.csv`: per-burst six-feature versus
  duration-only structural comparison.
- `data/fastpath_branch_evidence.csv`: sanitized controller decisions proving
  which FastPath branch ran.
- `data/structural_diagnostics.json`: machine-readable aggregate of the two
  preceding diagnostics.
- `data/optimistic_projection_summary.csv` and
  `data/optimistic_projection_assumptions.json`: explicitly synthetic
  200-workload what-if inputs and summary. They are not cluster evidence.
- `figures/fig_optimistic_improvements.pdf`: the compact projection figure used
  in the manuscript; its caption preserves the non-measured boundary.
  `figures/fig_optimistic_metrics.pdf` and
  `figures/fig_optimistic_distribution.pdf` are retained as supplemental plots
  rather than occupying additional manuscript pages.

The structural diagnostics do **not** claim a live duration-only result. The
five-pair pilot did not execute that arm. The prospective 30-block protocol is
registered at
`../Project implementation/k8s-ai-scheduler/experiments/reviewer_hardening.yaml`.

## Rebuild diagnostics

From `Project implementation/k8s-ai-scheduler`, point the extractor at the raw
pilot archive:

```bash
python scripts/analyze_pilot_structure.py \
  --pilot-dir /path/to/fastpath-heavy-pilot \
  --output-dir ../../paper/data
```

## Rebuild manuscript

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The bibliography is resolved through `references.bib`. Generated auxiliary
LaTeX files are not part of the scientific artifact.
