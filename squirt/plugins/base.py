"""
MetricNamespace Base Class

The foundation for all metric namespaces, both built-in and custom plugins.
Provides the unified pattern for defining metrics with full IDE support.
"""

from __future__ import annotations

from typing import Optional

# Import MetricBuilder from the canonical location
from ..metrics import MetricBuilder
from ..core.types import AggregationType
from ..categories.system import SystemMetric


class MetricNamespace:
    """
    Base class for all metric namespaces.

    Extend this class to create custom metric plugins with full IDE support.
    Each property decorated with @property returns a MetricBuilder.

    Example:
        class TaxMetrics(MetricNamespace):
            @property
            def field_accuracy(self) -> MetricBuilder:
                return self._define(
                    name="field_accuracy",
                    system_metric=SystemMetric.ACCURACY,  # Auto-derives AVERAGE
                    description="Accuracy of field extraction",
                )

            @property
            def field_count(self) -> MetricBuilder:
                return self._define(
                    name="field_count",
                    aggregation=AggregationType.AVERAGE,  # No system metric
                    description="Average number of fields extracted",
                )

        tax = TaxMetrics()
        # Use: tax.field_accuracy.from_output("accuracy")
    """

    def _define(
        self,
        name: str,
        aggregation: Optional[AggregationType] = None,
        system_metric: Optional[SystemMetric] = None,
        inverted: bool = False,
        description: str = "",
    ) -> MetricBuilder:
        """
        Define a metric in this namespace.

        Args:
            name: Unique metric identifier
            aggregation: How to aggregate across components (auto-derived from system_metric if provided)
            system_metric: Optional mapping to canonical system metric
            inverted: If True, lower values are better (e.g., error_rate)
            description: Human-readable description

        Returns:
            MetricBuilder for fluent configuration
        """
        return MetricBuilder(
            name=name,
            aggregation=aggregation,
            system_metric=system_metric,
            inverted=inverted,
            description=description,
        )


__all__ = [
    "MetricNamespace",
    "MetricBuilder",
]
