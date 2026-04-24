# CRANE

CRANE(under review) is a Python package for analyzing perturbation responses in single-cell expression data.

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

The demo GSC CRISPR data uses a perturbation targeting the `GSC` gene:

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

conda create -n crane-demo python=3.10 -y
conda activate crane-demo

pip install -e .
```

## Demo Workspace

Create a local test workspace inside the cloned repository and rebuild the demo `.h5ad` files. If you just ran the installation commands above, you are already inside this directory.

```bash
mkdir demo_workspace
python examples/rebuild_demo_data.py
```

This writes:

```text
demo_workspace/data/demo_gsc.h5ad
demo_workspace/data/demo_erlotinib_drug.h5ad
```

The demo scripts first run a standard Scanpy HVG/PCA/neighbors/UMAP view for exploratory analysis, then run CRANE on the rebuilt AnnData expression layer. The HVG-subsetted Scanpy object is only used for visualization and is not used as CRANE input.

Scanpy baseline plots are written to:

```text
demo_workspace/scanpy/
```

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

# Standard Scanpy exploratory view. CRANE below still uses the original adata.
view = adata.copy()
sc.pp.highly_variable_genes(
    view,
    layer="log1p_norm",
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
)
view_hvg = view[:, view.var["highly_variable"]].copy()
view_hvg.X = view_hvg.layers["log1p_norm"].copy()
sc.pp.pca(view_hvg)
sc.pp.neighbors(view_hvg)
sc.tl.umap(view_hvg)

result = crane.tl.gene_response(
    adata,
    perturbation_key="perturbation_targets",
    control_value="control",
    case_value="GSC",
    layer="log1p_norm",
    inplace=False,
)

result.summary(normalized=True).head()
```

Example output:

| gene | response_identity | response_score | gene_self_cor | gene_label_cor |
| --- | --- | --- | --- | --- |
| MFAP4 | 1 | 0.883 | 0.886 | 0.880 |
| POU5F1 | 1 | 0.880 | 0.913 | 0.846 |
| CXCR4 | 1 | 0.870 | 0.906 | -0.833 |
| LEFTY2 | 1 | 0.845 | 0.864 | -0.826 |
| TMEM14B | 1 | 0.841 | 0.834 | -0.847 |

The result object also supports response-gene relationship and module summaries:

```python
pair = result.gene_pair()
module = result.gene_module()
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
| GSC | M2 |
| TMEM14B | M2 |
| CXCR4 | M2 |
| POU5F1 | M3 |
| LEFTY2 | M2 |

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

The GSC CRISPR demo contains common cell-level covariates such as library size, cell-cycle scores, phase, and batch labels. The extension demo compares how strongly these covariates are structured on a standard Scanpy HVG graph versus the CRANE response graph.

Run:

```bash
python examples/extension_response_covariates.py
```

Example output:

| feature | Scanpy HVG graph | CRANE graph | reduction |
| --- | --- | --- | --- |
| G2M_score | 0.939 | 0.021 | 97.8% |
| phase_G2M | 0.777 | 0.001 | 99.9% |
| phase_G1 | 0.723 | 0.001 | 99.8% |
| S_score | 0.564 | 0.024 | 95.8% |
| n_genes | 0.541 | 0.028 | 94.9% |

This is the intended reading: covariates that are strongly structured in a standard Scanpy graph become much less structured on the CRANE response graph.

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

function_result.summary(normalized=True).head()
```

Example output, showing only the highest-response component for each gene set:

| gene set | top component | response_score | gene_self_cor | gene_label_cor |
| --- | --- | --- | --- | --- |
| MAPK | MAPK_pc2 | 0.662 | 0.708 | 0.612 |
| EGFR | EGFR_pc1 | 0.584 | 0.567 | -0.599 |

After the demo workspace is created, all commands above can be copied and run without editing paths or parameters.

## Demo Data Sources

The GSC CRISPR demo is derived from:

Genga, R. M. J. et al. Single-cell RNA-sequencing-based CRISPRi screening resolves molecular drivers of early human endoderm development. *Cell Reports* 27, 708-718.e10 (2019).

The Erlotinib drug demo is derived from:

Chang, M. T. et al. Identifying transcriptional programs underlying cancer drug response with TraCe-seq. *Nature Biotechnology* 40, 86-93 (2022).
