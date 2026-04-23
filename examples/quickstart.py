"""User-facing CRANE quickstart demo.

This demo shows the recommended scverse-style entry points:
- `crane.tl.gene_response(...)` for the main workflow
- `crane.tl.cell_response(...)` for the optional Step 1 tendency view
- `crane.tl.function_response(...)` for graph-backed functional follow-up
"""

from pathlib import Path

import scanpy as sc

import crane


# Replace this with the user's own h5ad file path.
input_h5ad = Path("path/to/your_input.h5ad")

# Optional future save path for a CRANE result object or exported AnnData.
output_dir = Path("path/to/output_dir")


if not input_h5ad.exists():
    raise SystemExit(
        "Please replace 'path/to/your_input.h5ad' with a real input file path "
        "before running this demo."
    )


# 1. Read single-cell input data.
adata = sc.read_h5ad(input_h5ad)


# 2. Run the main CRANE gene-response workflow.
result = crane.tl.gene_response(
    adata,
    perturbation_key="perturbation",
    control_value="control",
    inplace=False,
)


# 3. Inspect the main outputs exposed by the public API.
print(result)
print(result.summary().head(20))


# 4. Optional input-space summary write-back for downstream scverse use.
# This exports broadcastable summaries only. It does not reconstruct the
# internal CRANE result-space graph on the original AnnData axes.
# adata_out = result.to_anndata(adata)
# adata_out.write_h5ad(output_dir / "crane_output.h5ad")


# 5. Optional Step 1 cell-response view.
# cell_result = crane.tl.cell_response(
#     adata,
#     perturbation_key="perturbation",
#     control_value="control",
#     inplace=False,
# )
# print(cell_result.summary().head(20))
#
# If you prefer scverse-style input-space write-back:
# crane.tl.cell_response(
#     adata,
#     perturbation_key="perturbation",
#     control_value="control",
#     key_added="crane",
#     inplace=True,
# )
# print(adata.obs["crane_cell_score"].head())


# 6. Optional function-response follow-up from the gene-response result.
# Function response uses adata expression plus result.result_ad graph/labels.
# It does not rerun gene_response.
# gene_sets = {
#     "example_pathway": ["GENE1", "GENE2", "GENE3"],
# }
# func_result = crane.tl.function_response(
#     adata,
#     result=result,
#     gene_set=gene_sets,
# )
# print(func_result.summary().head(20))


# 7. Optional compatibility path.
# legacy_result = crane.run_crane(
#     adata,
#     perturbation_key="perturbation",
#     control_value="control",
# )
