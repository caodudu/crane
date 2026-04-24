# CRANE

CRANE is a Python package for analyzing perturbation responses in single-cell expression data.

It is designed for experiments such as CRISPR screens, Perturb-seq, CROP-seq, drug perturbation assays, or other single-cell perturbation studies where users want to ask:

- Which genes respond to this perturbation?
- Which cells show a stronger perturbation response?
- Which pathways or gene sets are affected?

![CRANE algorithm overview](assets/crane_algorithm_overview_fig1c.png)

## What You Need

CRANE works with an `AnnData` object, usually stored as an `.h5ad` file.

Your data should contain:

- an expression matrix in `adata.X` or an expression layer
- a column in `adata.obs` that stores perturbation labels
- one label for control cells
- one label for the perturbation you want to analyze

Example:

```text
adata.obs["perturbation"]
control label: "control"
case label: "KRAS-G12D"
```

## Installation

```bash
git clone https://github.com/caodudu/crane.git
cd crane
pip install -e .
```

For optional function or extension backends:

```bash
pip install -e ".[extensions]"
```

CRANE requires Python 3.10 or later.

## Quick Start

```python
import scanpy as sc
import crane

adata = sc.read_h5ad("your_data.h5ad")

result = crane.tl.gene_response(
    adata,
    perturbation_key="perturbation",
    control_value="control",
    case_value="KRAS-G12D",
    layer="log_norm",      # optional; omit if using adata.X
    inplace=False,
)

result.summary().head()
```

The output table ranks genes by perturbation response.

Common columns include:

- `response_score`: response strength
- `response_identity`: whether CRANE calls the gene responsive
- `gene_self_cor`: local expression consistency
- `gene_label_cor`: alignment with perturbation labels

## Cell Response

Use `cell_response` when you want to inspect which labeled perturbed cells look more response-like.

```python
cell_result = crane.tl.cell_response(
    adata,
    perturbation_key="perturbation",
    control_value="control",
    case_value="KRAS-G12D",
    layer="log_norm",
    inplace=False,
)

cell_result.summary().head()
```

## Function and Gene-Set Response

After running `gene_response`, CRANE can score pathways or custom gene sets on the learned response graph.

![CRANE function response overview](assets/crane_function_response_fig5a.png)

```python
gene_sets = {
    "MAPK_program": ["KRAS", "RAF1", "MAPK1", "MAPK3", "DUSP6"],
}

function_result = crane.tl.function_response(
    adata,
    result=result,
    gene_set=gene_sets,
    layer="log_norm",
)

function_result.summary().head()
```

## Command Line Usage

Run gene response from an `.h5ad` file:

```bash
python -m crane run \
  --input-h5ad your_data.h5ad \
  --perturbation-key perturbation \
  --control-value control \
  --case-value KRAS-G12D \
  --layer log_norm \
  --output-dir crane_output
```

Other commands:

```bash
python -m crane cell-response --help
python -m crane function-response --help
python -m crane extension-response --help
```

## Demo Data

The example datasets used during development are real `.h5ad` perturbation datasets and are larger than 100 MB, so they should not be stored directly in the git repository.

Recommended local layout:

```text
crane/
  data/
    demo_gsc.h5ad
    demo_mixl1.h5ad
  examples/
    quickstart.py
```

Recommended use:

```bash
python examples/quickstart.py \
  --input-h5ad data/demo_gsc.h5ad \
  --perturbation-key perturbation_targets \
  --control-value control \
  --case-value GSC
```

For public distribution, place demo `.h5ad` files in GitHub Releases or Zenodo, and keep only scripts and instructions in this repository. This keeps `git clone` fast while still giving users a reproducible local test.

## Minimal Example Script

See:

```text
examples/quickstart.py
```

This script runs gene response and optionally cell response on any compatible `.h5ad` file.
