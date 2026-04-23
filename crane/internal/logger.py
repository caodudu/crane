"""Structured logging protocol for the new CRANE package.

This module intentionally avoids logging every internal design detail.
At paper stage, default logs should stay concise:
- `user`: minimal run progress
- `reviewer`: paper-core milestones when explicitly enabled
- `debug`: diagnostics and internal branches when explicitly enabled
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Iterable

from ..io.schema import LogFileConfig, LoggerConfig


class _AudienceFilter(logging.Filter):
    def __init__(self, allowed: Iterable[str]) -> None:
        super().__init__()
        self.allowed = set(allowed)

    def filter(self, record: logging.LogRecord) -> bool:
        audience = getattr(record, "audience", "user")
        return audience in self.allowed


class _ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = getattr(record, "context_items", None)
        if not context:
            return base
        suffix = " ".join(f"{key}={value}" for key, value in context)
        return f"{base} | {suffix}"


class _UserConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        scope = "RUN"
        logger_name = getattr(record, "name", "")
        if "gene_response" in logger_name:
            scope = "GENE"
        elif "cell_response" in logger_name:
            scope = "CELL"
        elif "functional" in logger_name:
            scope = "FUNCTION"
        timestamp = self.formatTime(record, self.datefmt)
        return f"[CRANE] [{scope}] [{timestamp}] {record.getMessage()}"


def _resolve_level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def _build_structured_formatter() -> logging.Formatter:
    return _ContextFormatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_user_console_formatter() -> logging.Formatter:
    return _UserConsoleFormatter(datefmt="%Y-%m-%d %H:%M:%S")


def _reset_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _add_console_handler(
    logger: logging.Logger,
    level: int,
    allowed_audiences: Iterable[str],
    formatter: logging.Formatter,
) -> None:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(_AudienceFilter(allowed_audiences))
    logger.addHandler(handler)


def _add_file_handler(
    logger: logging.Logger,
    level: int,
    allowed_audiences: Iterable[str],
    file_config: LogFileConfig,
) -> Path | None:
    if not file_config.enabled:
        return None

    directory = Path(file_config.directory or ".")
    directory.mkdir(parents=True, exist_ok=True)
    filename = file_config.filename or "crane.log"
    path = directory / filename

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_build_structured_formatter())
    handler.addFilter(_AudienceFilter(allowed_audiences))
    logger.addHandler(handler)
    return path


@dataclass
class CRANELogger:
    """Layered logging facade with restrained paper-stage defaults."""

    logger: logging.Logger
    config: LoggerConfig
    sidecar_paths: dict[str, str] = field(default_factory=dict)

    def user(self, message: str, **context: Any) -> None:
        self._emit(logging.INFO, "user", message, **context)

    def reviewer(self, message: str, **context: Any) -> None:
        self._emit(logging.INFO, "reviewer", message, **context)

    def debug(self, message: str, **context: Any) -> None:
        self._emit(logging.DEBUG, "debug", message, **context)

    def step(
        self,
        stage: str,
        message: str,
        audience: str = "user",
        level: str = "INFO",
        **context: Any,
    ) -> None:
        self._emit(
            _resolve_level(level),
            audience,
            message,
            stage=stage,
            **context,
        )

    def event(
        self,
        event: str,
        message: str,
        audience: str = "user",
        level: str = "INFO",
        **context: Any,
    ) -> None:
        self._emit(
            _resolve_level(level),
            audience,
            message,
            event=event,
            **context,
        )

    def bind(self, name: str) -> "CRANELogger":
        child = self.logger.getChild(name)
        return CRANELogger(
            logger=child,
            config=self.config,
            sidecar_paths=dict(self.sidecar_paths),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "logger_name": self.logger.name,
            "console_level": self.config.console_level,
            "reviewer_console": self.config.reviewer_console,
            "debug_console": self.config.debug_console,
            "sidecar_paths": dict(self.sidecar_paths),
        }

    def _emit(
        self,
        level: int,
        audience: str,
        message: str,
        **context: Any,
    ) -> None:
        clean_context = {
            key: value
            for key, value in context.items()
            if value is not None
        }
        context_items = tuple(sorted(clean_context.items()))
        self.logger.log(
            level,
            message,
            extra={
                "audience": audience,
                "context_items": context_items,
            },
        )


def build_logger(config: LoggerConfig) -> CRANELogger:
    logger = logging.getLogger(config.name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = config.propagate
    _reset_handlers(logger)

    _add_console_handler(
        logger=logger,
        level=_resolve_level(config.console_level),
        allowed_audiences=["user"],
        formatter=_build_user_console_formatter(),
    )
    structured_console_audiences: list[str] = []
    if config.reviewer_console:
        structured_console_audiences.append("reviewer")
    if config.debug_console:
        structured_console_audiences.append("debug")
    if structured_console_audiences:
        _add_console_handler(
            logger=logger,
            level=_resolve_level(config.console_level),
            allowed_audiences=structured_console_audiences,
            formatter=_build_structured_formatter(),
        )

    sidecar_paths: dict[str, str] = {}
    user_path = _add_file_handler(
        logger=logger,
        level=logging.INFO,
        allowed_audiences=["user"],
        file_config=config.user_file,
    )
    reviewer_path = _add_file_handler(
        logger=logger,
        level=logging.INFO,
        allowed_audiences=["reviewer"],
        file_config=config.reviewer_file,
    )
    debug_path = _add_file_handler(
        logger=logger,
        level=logging.DEBUG,
        allowed_audiences=["debug"],
        file_config=config.debug_file,
    )
    if user_path is not None:
        sidecar_paths["user"] = str(user_path)
    if reviewer_path is not None:
        sidecar_paths["reviewer"] = str(reviewer_path)
    if debug_path is not None:
        sidecar_paths["debug"] = str(debug_path)

    return CRANELogger(logger=logger, config=config, sidecar_paths=sidecar_paths)
