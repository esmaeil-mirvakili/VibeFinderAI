"""Runtime wrappers for LangGraph-safe tool execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from loguru import logger
from pydantic import BaseModel, ValidationError


@dataclass(frozen=True)
class ToolContext:
    """Runtime resources used by tools but kept out of LangGraph state."""

    songs: pd.DataFrame
    retrieval_prompt_config: dict[str, Any]
    lyric_index: Any | None = None
    lyric_embedder: Any | None = None


class ToolResult(BaseModel):
    """Standard success wrapper for tool execution."""

    ok: Literal[True] = True
    tool_name: str
    output: dict[str, Any]
    warnings: tuple[str, ...] = ()
    trace: dict[str, Any] = {}


class ToolError(BaseModel):
    """Standard error wrapper for tool execution."""

    ok: Literal[False] = False
    tool_name: str
    error_type: str
    message: str
    details: dict[str, Any] = {}
    trace: dict[str, Any] = {}


class ToolRunner:
    """Validate JSON-like inputs, run tools, and return structured results."""

    def __init__(self, context: ToolContext, registry: dict[str, Any]):
        self.context = context
        self.registry = registry

    def run(self, tool_name: str, raw_input: dict[str, Any]) -> ToolResult | ToolError:
        started = time.perf_counter()
        if tool_name not in self.registry:
            return ToolError(
                tool_name=tool_name,
                error_type="unknown_tool",
                message=f"Tool is not registered: {tool_name}",
                trace={"duration_ms": _duration_ms(started)},
            )

        tool = self.registry[tool_name]
        try:
            request = tool.input_schema.model_validate(raw_input)
        except ValidationError as exc:
            logger.warning("tool_validation_failed", tool_name=tool_name, error_count=len(exc.errors()))
            return ToolError(
                tool_name=tool_name,
                error_type="validation_error",
                message="Tool input failed validation.",
                details={"errors": exc.errors(include_context=False)},
                trace={"duration_ms": _duration_ms(started)},
            )

        try:
            output = tool.callable(self.context, request)
        except Exception as exc:
            logger.exception("tool_runtime_failed", tool_name=tool_name)
            return ToolError(
                tool_name=tool_name,
                error_type=exc.__class__.__name__,
                message=str(exc),
                trace={"duration_ms": _duration_ms(started)},
            )

        output_data = output.model_dump(mode="json") if isinstance(output, BaseModel) else output
        warnings = tuple(output_data.get("warnings", ())) if isinstance(output_data, dict) else ()
        result = ToolResult(
            tool_name=tool_name,
            output=output_data,
            warnings=warnings,
            trace={"duration_ms": _duration_ms(started)},
        )
        logger.info(
            "tool_run_finished",
            tool_name=tool_name,
            duration_ms=result.trace["duration_ms"],
            warning_count=len(warnings),
        )
        return result


def _duration_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
