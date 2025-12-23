"""
Sleuth Extension Registry

Allows registering custom aggregation types and system metrics.

Usage:
    from sleuth.extensions import register_aggregation, register_system_metric

    # Register custom aggregation
    def geometric_mean(values: List[float]) -> float:
        import math
        return math.exp(sum(math.log(v) for v in values) / len(values))

    register_aggregation("geometric_mean", geometric_mean)

    # Register custom system metric
    register_system_metric("quality_score", "Quality Score")

    # Use in metrics
    from sleuth import m

    custom = m.custom("my_metric", aggregation="geometric_mean")
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Registry for custom aggregation functions
_custom_aggregations: contextvars.ContextVar[Dict[str, Callable]] = (
    contextvars.ContextVar(
        "custom_aggregations",
        default={},
    )
)

# Registry for custom system metrics
_custom_system_metrics: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "custom_system_metrics",
    default={},
)


def register_aggregation(
    name: str,
    func: Callable[[Sequence[Union[float, int, bool]]], Union[float, int]],
    description: str = "",
) -> None:
    """
    Register a custom aggregation function.

    Args:
        name: Name of the aggregation (e.g., "geometric_mean")
        func: Function that takes a list of values and returns aggregated result
        description: Optional description

    Example:
        def harmonic_mean(values):
            return len(values) / sum(1/v for v in values if v > 0)

        register_aggregation("harmonic_mean", harmonic_mean)
    """
    registry = _custom_aggregations.get().copy()
    registry[name] = func
    _custom_aggregations.set(registry)


def register_system_metric(name: str, display_name: Optional[str] = None) -> None:
    """
    Register a custom system metric category.

    Args:
        name: Internal name (e.g., "quality_score")
        display_name: Human-readable name (defaults to title-cased name)

    Example:
        register_system_metric("quality_score", "Quality Score")
    """
    registry = _custom_system_metrics.get().copy()
    registry[name] = display_name or name.replace("_", " ").title()
    _custom_system_metrics.set(registry)


def get_aggregation(name: str) -> Optional[Callable]:
    """Get a registered aggregation function by name."""
    return _custom_aggregations.get().get(name)


def get_system_metric(name: str) -> Optional[str]:
    """Get a registered system metric display name."""
    return _custom_system_metrics.get().get(name)


def list_aggregations() -> Dict[str, Callable]:
    """List all registered custom aggregation functions."""
    return _custom_aggregations.get().copy()


def list_system_metrics() -> Dict[str, str]:
    """List all registered custom system metrics."""
    return _custom_system_metrics.get().copy()


def apply_aggregation(
    name: str,
    values: Sequence[Union[float, int, bool]],
) -> Union[float, int]:
    """
    Apply an aggregation (built-in or custom) to values.

    Args:
        name: Aggregation name (enum value or custom name)
        values: Values to aggregate

    Returns:
        Aggregated result
    """
    from .core.types import AggregationType
    from .reporting.aggregation import aggregate_values

    # Try built-in first
    try:
        agg_type = AggregationType(name)
        return aggregate_values(values, agg_type)
    except ValueError:
        pass

    # Try custom
    custom_func = get_aggregation(name)
    if custom_func:
        return custom_func(values)

    raise ValueError(f"Unknown aggregation: {name}")


__all__ = [
    "register_aggregation",
    "register_system_metric",
    "get_aggregation",
    "get_system_metric",
    "list_aggregations",
    "list_system_metrics",
    "apply_aggregation",
]
