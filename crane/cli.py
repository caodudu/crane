"""Command-line interface for the public CRANE APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from . import tl
from .api import run_crane


DEFAULT_OUTPUT_DIR_NAME = "crane_cli_output"
SUMMARY_HEAD_ROWS = 20


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _command_slug(command_name: str) -> str:
    return str(command_name).strip().replace("-", "_")


def _summary_filename(command_name: str) -> str:
    return f"crane_{_command_slug(command_name)}_summary.json"


def _result_pickle_filename(command_name: str) -> str:
    return f"crane_{_command_slug(command_name)}_result.pkl"


def _result_h5ad_filename(command_name: str) -> str:
    return f"crane_{_command_slug(command_name)}_result.h5ad"


def _input_writeback_filename(command_name: str) -> str:
    return f"crane_{_command_slug(command_name)}_input_writeback.h5ad"


def _save_flags_requested(args: argparse.Namespace) -> bool:
    return any(
        bool(getattr(args, attr, False))
        for attr in ("write_input_anndata", "save_result_pkl", "save_result_h5ad")
    )


def _resolve_output_dir(args: argparse.Namespace) -> Path | None:
    if getattr(args, "output_dir", None):
        return Path(args.output_dir).expanduser().resolve()
    if _save_flags_requested(args):
        return (Path.cwd() / DEFAULT_OUTPUT_DIR_NAME).resolve()
    return None


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path).expanduser().resolve()
    if not table_path.exists():
        raise FileNotFoundError(f"Table file not found: {table_path}")
    return pd.read_csv(table_path, sep=None, engine="python", index_col=0)


def _load_gene_set_json(path: str | Path) -> dict[str, list[str]]:
    json_path = Path(path).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Gene-set JSON file not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("gene_sets"), dict):
        payload = payload["gene_sets"]
    if not isinstance(payload, dict):
        raise TypeError("Gene-set JSON must be a mapping or a {'gene_sets': ...} wrapper.")
    return {
        str(key): [str(gene) for gene in value]
        for key, value in payload.items()
    }


def _load_extension_payload(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = [
        bool(getattr(args, "gene_set_json", None)),
        bool(getattr(args, "gene_vector_csv", None)),
        bool(getattr(args, "cell_vector_csv", None)),
    ]
    if sum(supplied) != 1:
        raise ValueError(
            "extension-response requires exactly one of --gene-set-json, --gene-vector-csv, or --cell-vector-csv."
        )

    if getattr(args, "gene_set_json", None):
        gene_set_path = Path(args.gene_set_json).expanduser().resolve()
        return (
            {
                "gene_set": _load_gene_set_json(gene_set_path),
                "set_min_genes_count": args.set_min_genes_count,
                "set_loading_threshold": args.set_loading_threshold,
                "set_embedding_threshold": args.set_embedding_threshold,
                "vector_min_genes_count": args.vector_min_genes_count,
            },
            {
                "input_mode": "gene_set",
                "gene_set_json": str(gene_set_path),
            },
        )
    if getattr(args, "gene_vector_csv", None):
        gene_vector_path = Path(args.gene_vector_csv).expanduser().resolve()
        return (
            {
                "gene_vector": _read_table(gene_vector_path),
                "vector_min_genes_count": args.vector_min_genes_count,
            },
            {
                "input_mode": "gene_vector",
                "gene_vector_csv": str(gene_vector_path),
            },
        )
    cell_vector_path = Path(args.cell_vector_csv).expanduser().resolve()
    return (
        {
            "cell_vector": _read_table(cell_vector_path),
            "vector_min_genes_count": args.vector_min_genes_count,
        },
        {
            "input_mode": "cell_vector",
            "cell_vector_csv": str(cell_vector_path),
        },
    )


def _result_metadata_summary(result: Any) -> dict[str, Any]:
    metadata = dict(getattr(result, "metadata", {}) or {})
    return {
        key: _jsonable(metadata[key])
        for key in ("mode", "step1_summary", "step2_summary", "cell_response", "function_class")
        if key in metadata
    }


def _build_summary_payload(
    *,
    args: argparse.Namespace,
    input_h5ad: Path,
    output_dir: Path | None,
    result: Any,
    generated_files: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result_ad = getattr(result, "result_ad", None)
    payload: dict[str, Any] = {
        "command": args.command,
        "package": "crane",
        "input_h5ad": str(input_h5ad),
        "output_dir": None if output_dir is None else str(output_dir),
        "generated_files": generated_files,
        "result": {
            "result_ad_shape": None
            if result_ad is None
            else [int(result_ad.n_obs), int(result_ad.n_vars)],
            "summary_head": result.summary().head(SUMMARY_HEAD_ROWS).reset_index().to_dict(orient="records"),
            "metadata_summary": _result_metadata_summary(result),
        },
    }
    parameters = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if key
        not in {
            "handler",
            "command",
            "output_dir",
            "save_result_pkl",
            "save_result_h5ad",
            "write_input_anndata",
        }
        and value is not None
        and value is not False
    }
    payload["parameters"] = parameters
    if extra:
        payload["extra"] = _jsonable(extra)
    return payload


def _write_summary_json(
    *,
    command_name: str,
    output_dir: Path | None,
    payload: dict[str, Any],
    generated_files: dict[str, str],
) -> None:
    if output_dir is None:
        return
    summary_path = output_dir / _summary_filename(command_name)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    generated_files["run_summary_json"] = str(summary_path)


def _print_result(result: Any) -> None:
    print(result)
    print()
    print("Top summary:")
    print(result.summary().head(SUMMARY_HEAD_ROWS).to_string())


def _print_generated_files(generated_files: dict[str, str]) -> None:
    if not generated_files:
        return
    print()
    print("Generated files:")
    for label, path in generated_files.items():
        print(f"- {label}: {path}")


def _save_gene_like_result(
    *,
    command_name: str,
    args: argparse.Namespace,
    adata: Any,
    result: Any,
    output_dir: Path | None,
    allow_input_writeback: bool,
    allow_result_space_h5ad: bool,
) -> dict[str, str]:
    generated_files: dict[str, str] = {}

    if getattr(args, "write_input_anndata", False):
        if not allow_input_writeback:
            raise ValueError(f"{command_name} does not support --write-input-anndata.")
        if output_dir is None:
            raise RuntimeError("Internal CLI error: output_dir was not resolved for input-space writeback.")
        input_writeback_path = output_dir / _input_writeback_filename(command_name)
        result.to_anndata(adata, key_added=args.key_added).write_h5ad(input_writeback_path)
        generated_files["input_writeback_h5ad"] = str(input_writeback_path)

    if getattr(args, "save_result_pkl", False):
        if output_dir is None:
            raise RuntimeError("Internal CLI error: output_dir was not resolved for result pickle output.")
        result_pickle_path = output_dir / _result_pickle_filename(command_name)
        result.save(result_pickle_path)
        generated_files["result_pickle"] = str(result_pickle_path)

    if getattr(args, "save_result_h5ad", False):
        if not allow_result_space_h5ad:
            raise ValueError(f"{command_name} does not support --save-result-h5ad.")
        if output_dir is None:
            raise RuntimeError("Internal CLI error: output_dir was not resolved for result-space AnnData output.")
        if getattr(result, "result_ad", None) is None:
            raise ValueError(f"{command_name} did not return result-space AnnData; cannot save --save-result-h5ad.")
        result_space_path = output_dir / _result_h5ad_filename(command_name)
        result.result_ad.write_h5ad(result_space_path)
        generated_files["result_space_h5ad"] = str(result_space_path)

    return generated_files


def _save_extension_result(
    *,
    command_name: str,
    args: argparse.Namespace,
    result: Any,
    output_dir: Path | None,
) -> dict[str, str]:
    generated_files: dict[str, str] = {}
    if getattr(args, "save_result_h5ad", False):
        if output_dir is None:
            raise RuntimeError("Internal CLI error: output_dir was not resolved for extension/function output.")
        result_h5ad_path = output_dir / _result_h5ad_filename(command_name)
        result.to_anndata().write_h5ad(result_h5ad_path)
        generated_files["result_h5ad"] = str(result_h5ad_path)
    return generated_files


def _load_input_h5ad(path: str | Path) -> Any:
    import scanpy as sc

    input_h5ad = Path(path).expanduser().resolve()
    if not input_h5ad.exists():
        raise FileNotFoundError(f"Input .h5ad file not found: {input_h5ad}")
    return input_h5ad, sc.read_h5ad(input_h5ad)


def _run_gene_response_command(args: argparse.Namespace) -> int:
    input_h5ad, adata = _load_input_h5ad(args.input_h5ad)
    output_dir = _resolve_output_dir(args)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    result = run_crane(
        adata,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
        case_value=args.case_value,
        expression_layer=args.layer,
        key_added=args.key_added,
    )
    generated_files = _save_gene_like_result(
        command_name=args.command,
        args=args,
        adata=adata,
        result=result,
        output_dir=output_dir,
        allow_input_writeback=True,
        allow_result_space_h5ad=True,
    )
    payload = _build_summary_payload(
        args=args,
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        result=result,
        generated_files=generated_files,
    )
    _write_summary_json(
        command_name=args.command,
        output_dir=output_dir,
        payload=payload,
        generated_files=generated_files,
    )
    _print_result(result)
    _print_generated_files(generated_files)
    return 0


def _run_cell_response_command(args: argparse.Namespace) -> int:
    input_h5ad, adata = _load_input_h5ad(args.input_h5ad)
    output_dir = _resolve_output_dir(args)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    result = tl.cell_response(
        adata,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
        case_value=args.case_value,
        layer=args.layer,
        key_added=args.key_added,
        graph_method=args.graph_method,
        n_neighbors=args.n_neighbors,
        inplace=False,
    )
    generated_files = _save_gene_like_result(
        command_name=args.command,
        args=args,
        adata=adata,
        result=result,
        output_dir=output_dir,
        allow_input_writeback=True,
        allow_result_space_h5ad=False,
    )
    payload = _build_summary_payload(
        args=args,
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        result=result,
        generated_files=generated_files,
    )
    _write_summary_json(
        command_name=args.command,
        output_dir=output_dir,
        payload=payload,
        generated_files=generated_files,
    )
    _print_result(result)
    _print_generated_files(generated_files)
    return 0


def _run_function_response_command(args: argparse.Namespace) -> int:
    input_h5ad, adata = _load_input_h5ad(args.input_h5ad)
    output_dir = _resolve_output_dir(args)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    gene_result = run_crane(
        adata,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
        case_value=args.case_value,
        expression_layer=args.layer,
    )
    gene_set_path = Path(args.gene_set_json).expanduser().resolve()
    gene_set = _load_gene_set_json(gene_set_path)
    result = tl.function_response(
        adata,
        result=gene_result,
        gene_set=gene_set,
        layer=args.layer,
        set_min_genes_count=args.set_min_genes_count,
        set_loading_threshold=args.set_loading_threshold,
        set_embedding_threshold=args.set_embedding_threshold,
    )
    generated_files = _save_extension_result(
        command_name=args.command,
        args=args,
        result=result,
        output_dir=output_dir,
    )
    payload = _build_summary_payload(
        args=args,
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        result=result,
        generated_files=generated_files,
        extra={
            "source_gene_result_shape": [int(gene_result.result_ad.n_obs), int(gene_result.result_ad.n_vars)],
            "gene_set_json": str(gene_set_path),
        },
    )
    _write_summary_json(
        command_name=args.command,
        output_dir=output_dir,
        payload=payload,
        generated_files=generated_files,
    )
    _print_result(result)
    _print_generated_files(generated_files)
    return 0


def _run_extension_response_command(args: argparse.Namespace) -> int:
    input_h5ad, adata = _load_input_h5ad(args.input_h5ad)
    output_dir = _resolve_output_dir(args)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    gene_result = run_crane(
        adata,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
        case_value=args.case_value,
        expression_layer=args.layer,
    )
    payload_kwargs, payload_meta = _load_extension_payload(args)
    result = tl.extension_response(
        adata,
        result=gene_result,
        layer=args.layer,
        **payload_kwargs,
    )
    generated_files = _save_extension_result(
        command_name=args.command,
        args=args,
        result=result,
        output_dir=output_dir,
    )
    payload = _build_summary_payload(
        args=args,
        input_h5ad=input_h5ad,
        output_dir=output_dir,
        result=result,
        generated_files=generated_files,
        extra={
            "source_gene_result_shape": [int(gene_result.result_ad.n_obs), int(gene_result.result_ad.n_vars)],
            **payload_meta,
        },
    )
    _write_summary_json(
        command_name=args.command,
        output_dir=output_dir,
        payload=payload,
        generated_files=generated_files,
    )
    _print_result(result)
    _print_generated_files(generated_files)
    return 0


def _add_core_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-h5ad", required=True, help="Path to input .h5ad")
    parser.add_argument(
        "--perturbation-key",
        required=True,
        help="obs column containing perturbation labels",
    )
    parser.add_argument(
        "--control-value",
        required=True,
        help="Label used as the control condition.",
    )
    parser.add_argument(
        "--case-value",
        help="Optional case label. If omitted, CRANE keeps its current public default case-selection behavior.",
    )
    parser.add_argument(
        "--layer",
        help="Optional AnnData layer to use as the expression matrix.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for generated files. If omitted but any save flag is enabled, "
            f"the CLI writes into ./{DEFAULT_OUTPUT_DIR_NAME}."
        ),
    )


def _add_writeback_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--key-added",
        default="crane",
        help="Prefix used when writing input-space CRANE summaries back onto AnnData.",
    )
    parser.add_argument(
        "--write-input-anndata",
        action="store_true",
        help="Write an input-space AnnData copy with CRANE summaries expanded onto obs/var.",
    )


def _add_gene_result_save_arguments(parser: argparse.ArgumentParser, *, allow_result_space_h5ad: bool) -> None:
    parser.add_argument(
        "--save-result-pkl",
        action="store_true",
        help="Save the returned CRANE result object as a pickle file.",
    )
    if allow_result_space_h5ad:
        parser.add_argument(
            "--save-result-h5ad",
            action="store_true",
            help="Save result-space AnnData as an .h5ad file.",
        )


def _add_extension_save_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-result-h5ad",
        action="store_true",
        help="Save the returned extension/function AnnData as an .h5ad file.",
    )


def _add_set_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set-min-genes-count",
        type=int,
        default=10,
        help="Minimum overlap gene count required for a gene-set feature.",
    )
    parser.add_argument(
        "--set-loading-threshold",
        type=float,
        default=0.5,
        help="Loading threshold forwarded to function/extension gene-set evaluation.",
    )
    parser.add_argument(
        "--set-embedding-threshold",
        type=float,
        default=0.1,
        help="Embedding threshold forwarded to function/extension gene-set evaluation.",
    )


def build_parser(prog: str = "crane") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="CRANE command-line interface for the public gene, cell, function, and extension responses.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for command_name, help_text in (
        ("run", "Run CRANE gene-response analysis. Alias of gene-response."),
        ("gene-response", "Run crane.tl.gene_response on an input .h5ad file."),
    ):
        subparser = subparsers.add_parser(
            command_name,
            help=help_text,
            description=(
                "Run the stable crane public gene-response interface on an input "
                "AnnData file and optionally save the result object, result-space AnnData, "
                "and input-space writeback AnnData."
            ),
        )
        _add_core_input_arguments(subparser)
        _add_writeback_arguments(subparser)
        _add_gene_result_save_arguments(subparser, allow_result_space_h5ad=True)
        subparser.set_defaults(handler=_run_gene_response_command)

    cell_parser = subparsers.add_parser(
        "cell-response",
        help="Run crane.tl.cell_response on an input .h5ad file.",
        description=(
            "Run the public cell-response interface on an input AnnData file and optionally "
            "save the returned result object plus an input-space writeback AnnData."
        ),
    )
    _add_core_input_arguments(cell_parser)
    _add_writeback_arguments(cell_parser)
    _add_gene_result_save_arguments(cell_parser, allow_result_space_h5ad=False)
    cell_parser.add_argument(
        "--graph-method",
        default="umap",
        help="Graph method forwarded to crane.tl.cell_response (default: umap).",
    )
    cell_parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="Neighbor count forwarded to crane.tl.cell_response.",
    )
    cell_parser.set_defaults(handler=_run_cell_response_command)

    function_parser = subparsers.add_parser(
        "function-response",
        help="Run crane.tl.function_response from input .h5ad plus a gene-set JSON.",
        description=(
            "Run gene_response first, then evaluate function_response on the resulting CRANE "
            "result space using a gene-set JSON payload."
        ),
    )
    _add_core_input_arguments(function_parser)
    _add_extension_save_arguments(function_parser)
    _add_set_threshold_arguments(function_parser)
    function_parser.add_argument(
        "--gene-set-json",
        required=True,
        help="JSON file containing a mapping of function/gene-set names to gene lists.",
    )
    function_parser.set_defaults(handler=_run_function_response_command)

    extension_parser = subparsers.add_parser(
        "extension-response",
        help="Run crane.tl.extension_response from input .h5ad plus one extension payload.",
        description=(
            "Run gene_response first, then evaluate extension_response on the resulting CRANE "
            "result space using exactly one of gene_set, gene_vector, or cell_vector inputs."
        ),
    )
    _add_core_input_arguments(extension_parser)
    _add_extension_save_arguments(extension_parser)
    _add_set_threshold_arguments(extension_parser)
    extension_parser.add_argument(
        "--vector-min-genes-count",
        type=int,
        default=50,
        help="Minimum overlap gene count required for gene-vector extension evaluation.",
    )
    payload_group = extension_parser.add_mutually_exclusive_group(required=True)
    payload_group.add_argument(
        "--gene-set-json",
        help="JSON file containing a mapping of extension labels to gene lists.",
    )
    payload_group.add_argument(
        "--gene-vector-csv",
        help="CSV/TSV table with genes on the index and one or more vector columns.",
    )
    payload_group.add_argument(
        "--cell-vector-csv",
        help="CSV/TSV table with cells on the index and one or more vector columns.",
    )
    extension_parser.set_defaults(handler=_run_extension_response_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("CRANE CLI interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"CRANE CLI error: {exc}", file=sys.stderr)
        return 1
