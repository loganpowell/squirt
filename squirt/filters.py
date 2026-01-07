"""
Metric Filtering Utilities

Provides helpers to conditionally include/exclude metrics by namespace,
enabling flexible configuration for different environments or testing scenarios.

Runtime Filtering (Pytest):
    # Skip LLM metrics in CI
    pytest tests/ --skip-metrics-namespaces=llm,vector

    # Only collect core metrics
    pytest tests/ --only-metrics-namespaces=m

Manual Filtering:
    from squirt import m, track
    from squirt.contrib.llm import llm
    from squirt.filters import skip_namespaces, only_namespaces

    # Exclude expensive LLM metrics
    metrics = skip_namespaces([llm], [
        m.runtime_ms.from_output("metadata.runtime_ms"),
        llm.cost.from_output("usage.cost"),  # Skipped
    ])

    @track(metrics=metrics)
    def my_component(text: str) -> dict:
        ...
"""

import contextvars
from typing import Any

from .core.types import Metric
from .plugins.base import MetricNamespace

# Runtime namespace filter configuration
_namespace_filters: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "namespace_filters",
    default={"skip": None, "only": None},
)


def configure_namespace_filters(
    skip: list[str] | None = None,
    only: list[str] | None = None,
) -> None:
    """
    Configure runtime namespace filtering.

    This is called automatically by pytest plugin when using CLI options.
    Can also be called manually for programmatic configuration.

    Args:
        skip: List of namespace names to skip (e.g., ['llm', 'vector'])
        only: List of namespace names to include exclusively (e.g., ['m', 'data'])

    Example:
        from squirt.filters import configure_namespace_filters

        # Skip expensive metrics in CI
        configure_namespace_filters(skip=['llm', 'vector'])
    """
    _namespace_filters.set({"skip": skip, "only": only})


def get_namespace_filters() -> dict:
    """Get currently configured namespace filters."""
    return _namespace_filters.get()


def apply_runtime_filters(metrics: list[Metric]) -> list[Metric]:
    """
    Apply runtime namespace filters to a list of metrics.

    This is called automatically by the @track decorator before collecting metrics.

    Args:
        metrics: List of metrics to filter

    Returns:
        Filtered list based on runtime configuration
    """
    filters = _namespace_filters.get()
    skip_names = filters.get("skip")
    only_names = filters.get("only")

    if not skip_names and not only_names:
        return metrics

    # Import here to get the actual namespace instances
    from . import m as builtin_m

    # Build namespace name to type mapping
    namespace_map = {"m": type(builtin_m)}

    # Try to import contrib namespaces dynamically
    try:
        from .contrib.llm import llm

        namespace_map["llm"] = type(llm)
    except ImportError:
        pass

    try:
        from .contrib.vector import vector

        namespace_map["vector"] = type(vector)
    except ImportError:
        pass

    try:
        from .contrib.chunk import chunk

        namespace_map["chunk"] = type(chunk)
    except ImportError:
        pass

    try:
        from .contrib.data import data

        namespace_map["data"] = type(data)
    except ImportError:
        pass

    # Apply filters
    if only_names:
        allowed_types = [
            namespace_map.get(name) for name in only_names if name in namespace_map
        ]
        return [
            m
            for m in metrics
            if getattr(m, "_namespace", None) is not None
            and type(m._namespace) in allowed_types
        ]
    elif skip_names:
        blocked_types = [
            namespace_map.get(name) for name in skip_names if name in namespace_map
        ]
        return [
            m
            for m in metrics
            if getattr(m, "_namespace", None) is None
            or type(m._namespace) not in blocked_types
        ]

    return metrics


def skip_namespaces(
    namespaces: list[MetricNamespace | Any],
    metrics: list[Metric],
) -> list[Metric]:
    """
    Filter out metrics that belong to specified namespaces.

    Args:
        namespaces: List of namespace objects to exclude (e.g., [llm, vector])
        metrics: List of metrics to filter

    Returns:
        Filtered list of metrics excluding those from specified namespaces

    Example:
        from squirt import m
        from squirt.contrib.llm import llm
        from squirt.filters import skip_namespaces

        # Skip all LLM metrics (cost, tokens, etc.)
        filtered = skip_namespaces([llm], [
            m.runtime_ms.from_output("metadata.runtime_ms"),
            llm.cost.from_output("usage.cost"),  # Excluded
        ])
    """
    # Get namespace types to check against
    namespace_types = [type(ns) for ns in namespaces]

    filtered_metrics = []
    for metric in metrics:
        # Get the namespace of this metric by checking its _namespace attribute
        metric_namespace = getattr(metric, "_namespace", None)

        # Keep metric if its namespace is not in the skip list
        if metric_namespace is None or type(metric_namespace) not in namespace_types:
            filtered_metrics.append(metric)

    return filtered_metrics


def only_namespaces(
    namespaces: list[MetricNamespace | Any],
    metrics: list[Metric],
) -> list[Metric]:
    """
    Filter to only include metrics from specified namespaces.

    Args:
        namespaces: List of namespace objects to include (e.g., [m, data])
        metrics: List of metrics to filter

    Returns:
        Filtered list of metrics including only those from specified namespaces

    Example:
        from squirt import m
        from squirt.contrib.llm import llm
        from squirt.filters import only_namespaces

        # Only keep built-in metrics
        filtered = only_namespaces([m], [
            m.runtime_ms.from_output("metadata.runtime_ms"),  # Kept
            llm.cost.from_output("usage.cost"),  # Excluded
        ])
    """
    # Get namespace types to check against
    namespace_types = [type(ns) for ns in namespaces]

    filtered_metrics = []
    for metric in metrics:
        # Get the namespace of this metric
        metric_namespace = getattr(metric, "_namespace", None)

        # Keep metric if its namespace is in the include list
        if metric_namespace is not None and type(metric_namespace) in namespace_types:
            filtered_metrics.append(metric)

    return filtered_metrics


def when_env(
    var: str,
    value: str = "true",
    metrics: list[Metric] = None,
) -> list[Metric]:
    """
    Conditionally include metrics based on environment variable.

    Args:
        var: Environment variable name to check
        value: Value to match (default: "true")
        metrics: Metrics to include if condition is met

    Returns:
        Metrics list if condition is met, empty list otherwise

    Example:
        import os
        from squirt.filters import when_env

        # Only collect expensive metrics when explicitly enabled
        expensive_metrics = when_env("COLLECT_LLM_METRICS", metrics=[
            llm.cost.from_output("usage.cost"),
            llm.total_tokens.from_output("usage.tokens"),
        ])
    """
    import os

    if metrics is None:
        metrics = []

    return metrics if os.environ.get(var, "").lower() == value.lower() else []


__all__ = [
    "skip_namespaces",
    "only_namespaces",
    "when_env",
    "configure_namespace_filters",
    "get_namespace_filters",
    "apply_runtime_filters",
]
