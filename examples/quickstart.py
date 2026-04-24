"""User-facing CRANE quickstart demo."""

from pathlib import Path

import scanpy as sc

import crane


input_h5ad = Path("path/to/your_input.h5ad")

if not input_h5ad.exists():
    raise SystemExit(
        "Please replace 'path/to/your_input.h5ad' with a real input file path before running this demo."
    )

adata = sc.read_h5ad(input_h5ad)

result = crane.tl.gene_response(
    adata,
    perturbation_key="perturbation",
    control_value="control",
    inplace=False,
)

print(result)
print(result.summary().head(20))

# Optional follow-up:
#
# cell_result = crane.tl.cell_response(
#     adata,
#     perturbation_key="perturbation",
#     control_value="control",
#     inplace=False,
# )
#
# gene_sets = {"example_pathway": ["GENE1", "GENE2", "GENE3"]}
# function_result = crane.tl.function_response(
#     adata,
#     result=result,
#     gene_set=gene_sets,
# )
