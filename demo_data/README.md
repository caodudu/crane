# Demo Data Components

This directory stores compressed components for rebuilding small CRANE demo `.h5ad` files.

The original development `.h5ad` files contain repeated expression layers and many metadata columns. For the GitHub demo, only the necessary pieces are kept:

- compressed CSR expression matrix
- minimal `obs`
- gene names
- dataset manifest
- MAPK and EGFR test gene-set JSON

Rebuild local demo files:

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

Recommended demos:

```bash
python examples/quickstart.py \
  --cell-response
```

```bash
python examples/extension_response_covariates.py
```

```bash
python examples/function_response_erlotinib.py
```
