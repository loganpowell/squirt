"""
Squirt Plugin System

Provides base classes for creating custom metric namespaces.

Recommended pattern for IDE support:
    from squirt.plugins import MetricBuilder, AggregationType, SystemMetric

    class MyMetrics:
        '''My custom metrics with full IDE autocomplete.'''

        # System metric: auto-derives AVERAGE aggregation
        custom_accuracy: MetricBuilder = MetricBuilder(
            "custom_accuracy",
            system_metric=SystemMetric.ACCURACY,
            description="Custom accuracy score",
        )

        # Non-system metric: must specify aggregation
        field_count: MetricBuilder = MetricBuilder(
            "field_count",
            aggregation=AggregationType.SUM,
            description="Total field count",
        )

    my = MyMetrics()
    # Usage: my.custom_accuracy.from_output("score")

Legacy pattern (still works, but less IDE support in decorator contexts):
    from squirt.plugins import MetricNamespace, MetricBuilder

    class MyMetrics(MetricNamespace):
        @property
        def custom_accuracy(self) -> MetricBuilder:
            return self._define(...)

    my = MyMetrics()
"""

from ..categories.system import SystemMetric
from ..core.types import AggregationType
from .base import MetricBuilder, MetricNamespace

__all__ = [
    "MetricNamespace",
    "MetricBuilder",
    "AggregationType",
    "SystemMetric",
]
