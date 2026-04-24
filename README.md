# CRANE

CRANE is a Python package for analyzing perturbation responses in single-cell expression data.

Unlike a simple perturbed-vs-control expression comparison, CRANE evaluates whether expression changes are aligned with perturbation labels across local cell neighborhoods. This is useful when some labeled cells do not fully respond, or when batch, cell cycle, library size, and other structured variation obscure the perturbation signal.

CRANE helps answer:

- Which genes respond to this perturbation?
- Which cells respond to this perturbation?
- Which functions (gene sets) respond to this perturbation?

![CRANE algorithm overview](assets/crane_algorithm_overview_fig1c.png)

## What You Need

CRANE uses an `AnnData` object, usually stored as an `.h5ad` file.

Your data should contain:

- a cells-by-genes expression matrix in `adata.X` or an expression layer
- gene names in `adata.var_names`
- a column in `adata.obs` that labels control and perturbed cells
- enough cells in both groups for balanced sampling

The demo GSC CRISPR data uses:

```text
expression layer: adata.layers["log1p_norm"]
perturbation labels: adata.obs["perturbation_targets"]
control label: "control"
case label: "GSC"
```

## Installation

```bash
git clone https://github.com/caodudu/crane.git
cd crane
pip install -e .
python examples/rebuild_demo_data.py
python examples/scanpy_warmup.py
```

These commands create a local `demo_workspace/` directory, rebuild the demo `.h5ad` files, and run a standard Scanpy HVG/PCA/neighbors/UMAP workflow once before CRANE.

The Scanpy step is included for two reasons:

- it warms up Scanpy/UMAP runtime dependencies before timing CRANE
- it gives a familiar baseline view of the demo data

The HVG-filtered Scanpy object is not used as CRANE input.

## Standard Scanpy baseline

The warm-up command writes baseline UMAP plots to:

```text
demo_workspace/scanpy/
```

This step uses the standard Scanpy HVG workflow for visualization only. CRANE is then run on the expression layer from the rebuilt AnnData object, not on the HVG-subsetted Scanpy object.

## Quick Start (Gene response)

Run the GSC CRISPR demo:

```bash
python examples/quickstart.py
```

Equivalent Python API:

```python
import scanpy as sc
import crane

adata = sc.read_h5ad("demo_workspace/data/demo_gsc.h5ad")

result = crane.tl.gene_response(
    adata,
    perturbation_key="perturbation_targets",
    control_value="control",
    case_value="GSC",
    layer="log1p_norm",
    inplace=False,
)

result.summary().head()
```

Example output:

| gene | response_identity | response_score | gene_self_cor | gene_label_cor |
| --- | --- | --- | --- | --- |
| MFAP4 | 1 | 1.249 | 0.886 | 0.880 |
| POU5F1 | 1 | 1.245 | 0.913 | 0.846 |
| CXCR4 | 1 | 1.230 | 0.906 | -0.833 |
| LEFTY2 | 1 | 1.195 | 0.864 | -0.826 |
| TMEM14B | 1 | 1.189 | 0.834 | -0.847 |

The result object also supports response-gene relationship and module summaries:

```python
pair = result.gene_pair()
module = result.gene_module(method="correlation_components")
```

Example pairwise gene-response output:

| gene | GSC | TMEM14B | CXCR4 | POU5F1 | LEFTY2 |
| --- | --- | --- | --- | --- | --- |
| GSC | 0.467 | 0.484 | 0.512 | -0.488 | 0.484 |
| TMEM14B | 0.484 | 0.535 | 0.552 | -0.547 | 0.474 |
| CXCR4 | 0.512 | 0.552 | 0.773 | -0.597 | 0.618 |
| POU5F1 | -0.488 | -0.547 | -0.597 | 0.769 | -0.456 |
| LEFTY2 | 0.484 | 0.474 | 0.618 | -0.456 | 0.701 |

Example module output:

| gene | module_label |
| --- | --- |
| GSC | M1 |
| TMEM14B | M1 |
| CXCR4 | M1 |
| POU5F1 | M1 |
| LEFTY2 | M1 |

## Quick Start (Cell response)

Run cell response on the same GSC CRISPR demo:

```bash
python examples/quickstart.py --cell-response
```

Python API:

```python
cell_result = crane.tl.cell_response(
    adata,
    perturbation_key="perturbation_targets",
    control_value="control",
    case_value="GSC",
    layer="log1p_norm",
    inplace=False,
)

cell_result.summary().head()
```

Example output:

| cell | cell_score |
| --- | --- |
| AAACGGGAGGCCCGTT-1 | -0.513 |
| AAACGGGGTGCAACTT-1 | -0.137 |
| AAACGGGTCGCGGATC-1 | 0.505 |
| AAAGCAATCGCCTGTT-1 | 0.422 |
| AACACGTTCTAACTCT-1 | -0.705 |

## Quick Start (Extension response)

CRANE can also evaluate cell-level covariates or other vectors on the same response graph. This is useful for checking whether common covariates remain strongly structured after CRANE graph refinement.

Run the GSC CRISPR covariate demo:

```bash
python examples/extension_response_covariates.py
```

The demo evaluates library size, cell-cycle scores, phase, and batch labels.

Example output:

| feature | response_score | gene_self_cor | gene_label_cor |
| --- | --- | --- | --- |
| phase_G1 | 0.128 | -0.001 | 0.128 |
| S_score | 0.116 | -0.024 | 0.114 |
| phase_G2M | 0.103 | -0.001 | -0.103 |
| n_genes | 0.073 | -0.028 | -0.067 |
| G2M_score | 0.038 | -0.021 | -0.032 |

## Quick Start (Functional response)

The functional demo uses Erlotinib drug perturbation data and two test gene sets, `MAPK` and `EGFR`, which are directly relevant to the drug-response example.

![CRANE function response overview](assets/crane_function_response_fig5a.png)

Run:

```bash
python examples/function_response_erlotinib.py
```

Python API:

```python
import json
import scanpy as sc
import crane

adata = sc.read_h5ad("demo_workspace/data/demo_erlotinib_drug.h5ad")
gene_sets = json.load(open("demo_data/erlotinib_mapk_egfr_genesets.json"))["gene_sets"]

gene_result = crane.tl.gene_response(
    adata,
    perturbation_key="perturbation_targets",
    control_value="nc",
    case_value="sensi",
    layer="log1p_norm",
    inplace=False,
)

function_result = crane.tl.function_response(
    adata,
    result=gene_result,
    gene_set=gene_sets,
    layer="log1p_norm",
    set_min_genes_count=10,
)

function_result.summary().head()
```

Example output:

| feature | response_score | gene_self_cor | gene_label_cor | gene_call |
| --- | --- | --- | --- | --- |
| EGFR_mode1 | 0.768 | 0.556 | 0.529 | 28 |
| MAPK_mode2 | 0.626 | 0.448 | 0.438 | 34 |
| MAPK_mode1 | 0.518 | 0.335 | 0.394 | 41 |
| EGFR_mode3 | 0.278 | 0.125 | 0.249 | 32 |
| MAPK_mode4 | 0.239 | 0.112 | 0.211 | 27 |

## Demo Workspace

The demo data in this repository is stored as compressed AnnData components, not as the original large `.h5ad` files.

Create the local demo workspace:

```bash
python examples/rebuild_demo_data.py
python examples/scanpy_warmup.py
```

This writes:

```text
demo_workspace/data/demo_gsc.h5ad
demo_workspace/data/demo_erlotinib_drug.h5ad
demo_workspace/scanpy/
```

After that, the demo commands in this README can be copied and run without editing paths or parameters.
