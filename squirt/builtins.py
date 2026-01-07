"""
Built-in Metrics for Squirt

The 5 most commonly used metrics, available as `m.metric_name`:
- runtime_ms: Execution time (SUM → LATENCY)
- memory_mb: Memory usage (MAX → RESOURCE_USAGE)
- error_free: No errors occurred (AVERAGE → ERROR_RATE)
- structure_valid: Output structure is valid (AVERAGE → ERROR_RATE)
- expected_match: Output matches expected (AVERAGE → ACCURACY)

Usage:
    from squirt import m

    @track(metrics=[
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.memory_mb.from_output("metadata.memory_mb"),
        m.error_free.from_output("metadata.error_free"),
    ])
    def my_component(text: str) -> dict:
        ...
"""

from .categories.system import SystemMetric
from .core.types import AggregationType
from .plugins.base import MetricBuilder, MetricNamespace


class BuiltinMetrics(MetricNamespace):
    """
    Built-in metrics for common use cases.

    These are the most frequently used metrics based on codebase analysis:
    - runtime_ms (12 uses)
    - memory_mb (10 uses)
    - error_free (4 uses)
    - structure_valid (3 uses)
    - expected_match (3 uses)
    """

    @property
    def runtime_ms(self) -> MetricBuilder:
        """Total execution time in milliseconds."""
        return self._define(
            name="runtime_ms",
            aggregation=AggregationType.SUM,
            system_metric=SystemMetric.RUNTIME_MS,
            description="Total execution time across all components",
        )

    @property
    def memory_mb(self) -> MetricBuilder:
        """Peak memory usage in megabytes."""
        return self._define(
            name="memory_mb",
            aggregation=AggregationType.MAX,
            system_metric=SystemMetric.MEMORY_MB,
            description="Peak memory usage",
        )

    @property
    def error_free(self) -> MetricBuilder:
        """Whether execution completed without errors (1.0 = no errors)."""
        return self._define(
            name="error_free",
            aggregation=AggregationType.AVERAGE,
            system_metric=SystemMetric.ERROR_RATE,
            inverted=True,
            description="Error-free execution rate",
        )

    @property
    def structure_valid(self) -> MetricBuilder:
        """Whether output structure is valid (1.0 = valid)."""
        return self._define(
            name="structure_valid",
            aggregation=AggregationType.AVERAGE,
            system_metric=SystemMetric.ERROR_RATE,
            inverted=True,
            description="Output structure validity",
        )

    @property
    def expected_match(self) -> MetricBuilder:
        """How closely output matches expected value (0.0-1.0)."""
        return self._define(
            name="expected_match",
            aggregation=AggregationType.AVERAGE,
            system_metric=SystemMetric.ACCURACY,
            description="Accuracy against expected output",
        )

    @property
    def assert_passes(self) -> MetricBuilder:
        """Assert metric passes threshold, fail test otherwise."""
        return self._define(
            name="assert_passes",
            aggregation=AggregationType.FAILURE,
            system_metric=SystemMetric.ERROR_RATE,
            description="Assert metric passes threshold, fail test otherwise",
        )

    def custom(
        self,
        name: str,
        aggregation: AggregationType = AggregationType.AVERAGE,
        system_metric: SystemMetric | None = None,
        description: str = "",
    ) -> MetricBuilder:
        """
        Create an ad-hoc custom metric.

        Use this for one-off metrics that don't need to be reused
        across multiple components.

        Args:
            name: Unique metric name
            aggregation: How to aggregate across components
            system_metric: Optional system-level metric category
            description: Human-readable description

        Returns:
            MetricBuilder for the custom metric

        Example:
            m.custom("tokens_per_second", AggregationType.AVERAGE).from_output(
                lambda o: o["tokens"] / (o["runtime_ms"] / 1000)
            )
        """
        return self._define(
            name=name,
            aggregation=aggregation,
            system_metric=system_metric,
            description=description or f"Custom metric: {name}",
        )


# Singleton instance - the main entry point
m = BuiltinMetrics()

__all__ = ["m", "BuiltinMetrics"]
