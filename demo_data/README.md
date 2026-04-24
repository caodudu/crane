# Demo Data Components

This directory stores compressed components for rebuilding small CRANE demo `.h5ad` files.

The original development `.h5ad` files contain repeated expression layers and many metadata columns. For the GitHub demo, only the necessary pieces are kept:

- compressed CSR expression matrix
- minimal `obs`
- gene names
- dataset manifest
- PROGENy top-200 gene-set JSON

Rebuild local demo files:

```bash
python examples/rebuild_demo_data.py --dataset all
```

This writes:

```text
data/demo_gsc.h5ad
data/demo_drug_trace_progeny.h5ad
```

Recommended demos:

```bash
python examples/quickstart.py \
  --input-h5ad data/demo_gsc.h5ad \
  --perturbation-key perturbation_targets \
  --control-value control \
  --case-value GSC \
  --layer log1p_norm \
  --cell-response
```

```bash
python examples/function_response_progeny.py \
  --input-h5ad data/demo_drug_trace_progeny.h5ad \
  --pathways MAPK EGFR
```
