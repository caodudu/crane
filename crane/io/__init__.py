"""I/O contracts for CRANE."""

from .schema import (
    CRANEConfig,
    FunctionalInput,
    LogFileConfig,
    LoggerConfig,
    RuntimeOptions,
)

__all__ = [
    "CRANEConfig",
    "CRANEResult",
    "FunctionalInput",
    "LogFileConfig",
    "LoggerConfig",
    "RuntimeOptions",
    "load_result",
]


def __getattr__(name: str):
    if name in {"CRANEResult", "load_result"}:
        from .result import CRANEResult, load_result

        exports = {
            "CRANEResult": CRANEResult,
            "load_result": load_result,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
