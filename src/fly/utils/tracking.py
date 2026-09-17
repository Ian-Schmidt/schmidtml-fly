"""Experiment tracking: where the MLflow database lives and how to mark the stages of a run with traces."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow
from mlflow.entities import LiveSpan, SpanType

from fly.utils.config import load

_CFG = load("training")
EXPERIMENT: str = _CFG["experiment"]
TRACKING: str = _CFG["tracking_uri"]


@contextmanager
def stage(name: str, kind: str = SpanType.TASK, root: str | None = None, **inputs: Any) -> Iterator[LiveSpan]:
    """Opens an MLflow Tracing span: the stage's duration, its inputs and outputs.

    Args:
        name: Name of the stage in the trace.
        kind: Span type from `SpanType`.
        root: `run_id`; if given, a new trace bound to that run is opened, otherwise the span nests
            into the current one.
        **inputs: Inputs of the stage — any JSON-compatible values.

    Yields:
        The live span; the stage's outputs are recorded with `span.set_outputs(...)`.

    Example:
        >>> with stage("embed", texts=5) as span:  # doctest: +SKIP
        ...     span.set_outputs({"dim": 384})
    """
    with mlflow.start_span(name, span_type=kind, run_id=root) as span:
        span.set_inputs(inputs)
        yield span
