# KubeML-Aware paper package

This local package contains the complete manuscript source, final PDF, vector
figures, PNG previews, supporting data, and the 200-workload scenario bundle.

## Contents

- `KubeML-Aware-paper.pdf`: final seven-page manuscript.
- `main.tex` and `references.bib`: ready-to-build manuscript source.
- `source/`: an additional source copy and the paper README.
- `figures/`: publication-quality PDF figures and their available LaTeX sources.
- `figures/png/`: PNG previews of every PDF figure.
- `data/`: pilot measurements, diagnostics, timelines, and projection inputs.
- `experiments/200-workload/`: workload definitions, scenario generators, CSV/JSON outputs, and standalone charts.

The five-pair Kubernetes pilot is measured evidence. The 200-workload values
remain a declared synthetic scenario and are identified as such in the paper
caption and supporting files.

## Build

From the package root, run:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
