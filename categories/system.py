"""
System metric categories for Sleuth.

Canonical system-level metrics that aggregate across the entire pipeline.
"""

from enum import Enum
from typing import Dict, Optional, Set


class SystemMetric(Enum):
    """Canonical system-level metrics."""

    ACCURACY = "accuracy"
    RUNTIME_MS = "runtime_ms"
    MEMORY_MB = "memory_mb"
    CPU_PERCENT = "cpu_percent"
    COST_USD = "cost_usd"
    TOTAL_TOKENS = "total_tokens"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    LATENCY_P95 = "latency_p95"


# Metrics that need inversion (higher is better → lower is better)
INVERTED_METRICS: Set[str] = {
    "error_free",
    "structure_valid",
    "expression_valid",
    "json_schema_valid",
}


def get_aggregation_type(sys_metric: SystemMetric):
    """
    Get the aggregation type for a system metric.

    Args:
        sys_metric: SystemMetric enum value

    Returns:
        AggregationType enum value
    """
    from ..core.types import AggregationType

    SYSTEM_METRIC_AGGREGATION: Dict[SystemMetric, AggregationType] = {
        SystemMetric.ACCURACY: AggregationType.AVERAGE,
        SystemMetric.RUNTIME_MS: AggregationType.SUM,
        SystemMetric.MEMORY_MB: AggregationType.MAX,
        SystemMetric.CPU_PERCENT: AggregationType.AVERAGE,
        SystemMetric.COST_USD: AggregationType.SUM,
        SystemMetric.TOTAL_TOKENS: AggregationType.SUM,
        SystemMetric.THROUGHPUT: AggregationType.AVERAGE,
        SystemMetric.ERROR_RATE: AggregationType.AVERAGE,
        SystemMetric.LATENCY_P95: AggregationType.P95,
    }

    return SYSTEM_METRIC_AGGREGATION.get(sys_metric)


def should_invert(component_metric: str) -> bool:
    """
    Check if a component metric needs inversion for system metric.

    Some metrics are "higher is better" at component level but map to
    "lower is better" system metrics (e.g., error_free → error_rate).

    Args:
        component_metric: Name of the component-level metric

    Returns:
        True if metric should be inverted (1.0 - value), False otherwise
    """
    return component_metric in INVERTED_METRICS


__all__ = [
    "SystemMetric",
    "INVERTED_METRICS",
    "get_aggregation_type",
    "should_invert",
]
