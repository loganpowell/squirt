"""
Core type definitions for Squirt.

This module provides the fundamental types used throughout the library:
- AggregationType: How metrics aggregate across components
- Metric: A single metric definition
- MetricResult: Result of metric collection
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union
import uuid


class AggregationType(Enum):
    """How metrics are aggregated up the dependency tree."""

    AVERAGE = "average"  # Average across all components (accuracy, similarity)
    SUM = "sum"  # Total across all components (cost, runtime, tokens)
    MAX = "max"  # Maximum value (bottleneck detection)
    MIN = "min"  # Minimum value (best case)
    COUNT = "count"  # Count of occurrences
    FAILURE = "failure"  # Count of failures (bool False or below threshold)
    P95 = "p95"  # 95th percentile (latency metrics)
    P99 = "p99"  # 99th percentile (latency metrics)


@dataclass
class Metric:
    """
    Defines a metric to collect for a component.

    Attributes:
        name: Unique identifier for the metric
        transform: Function that takes (inputs, output) -> metric value
        agg: How to aggregate this metric across components
        failure_threshold: Optional threshold below which metric is considered a failure
        is_assertion: If True, raises pytest assertion error on failure (for blocking PRs)
        system_metric: Optional system-level metric this component metric maps to
    """

    name: str
    transform: Callable[[Dict[str, Any], Any], Union[float, int, bool]]
    agg: AggregationType
    failure_threshold: Optional[Union[float, int]] = None
    is_assertion: bool = False
    system_metric: Optional[Any] = (
        None  # SystemMetric enum (Optional to avoid circular import)
    )


@dataclass
class MetricResult:
    """Result of a metric collection for a single component execution."""

    component: str
    test_case_id: str
    metrics: Dict[str, Union[float, int, bool]]
    aggregation_types: Dict[str, str]
    inputs: Dict[str, Any]
    output: Any
    timestamp: float
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_metric_map: Dict[str, str] = field(default_factory=dict)


__all__ = [
    "AggregationType",
    "Metric",
    "MetricResult",
]
