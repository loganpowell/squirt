"""
Squirt Metrics Module

Module-based metrics for robust IDE support. All metrics are defined as
module-level constants with explicit type annotations.

Usage:
    from squirt import metrics

    @track(metrics=[
        metrics.runtime_ms.from_output("metadata.runtime_ms"),
        metrics.memory_mb.from_output("metadata.peak_memory_mb"),
        metrics.custom("my_metric").compute(my_transform),
    ])
    def my_component(text: str) -> dict:
        ...

Creating Custom Metrics:
-----------------------

For project-specific metrics, create your own module:

    # my_project/metrics.py
    from squirt.metrics import MetricBuilder, AggregationType, SystemMetric

    # System metric (auto-derives aggregation)
    field_accuracy: MetricBuilder = MetricBuilder(
        "field_accuracy",
        system_metric=SystemMetric.ACCURACY,  # Auto-derives AVERAGE
    )

    # Non-system metric (explicit aggregation required)
    field_count: MetricBuilder = MetricBuilder(
        "field_count",
        aggregation=AggregationType.AVERAGE,
        description="Average field count per document",
    )

    # Usage:
    from my_project import metrics as project

    project.field_accuracy.compute(my_fn)
    project.field_count.from_output("metadata.field_count")

MetricBuilder Instantiation Patterns:
------------------------------------

See MetricBuilder class docstring for detailed examples of:
- System metrics (auto-derive aggregation)
- Non-system metrics (explicit aggregation)
- Inverted metrics (error_free → error_rate)
- Assertion metrics (fail tests on failure)
- Override patterns
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .categories.system import SystemMetric
from .core.types import AggregationType, Metric

# =============================================================================
# Core Types (re-exported for convenience)
# =============================================================================

# Re-export these so plugins can import from squirt.metrics
__all__ = [
    # Types for plugin authors
    "MetricBuilder",
    "AggregationType",
    "SystemMetric",
    "Metric",
    # Built-in metrics
    "runtime_ms",
    "memory_mb",
    "error_free",
    "structure_valid",
    "expected_match",
    "assert_passes",
    # Factory function
    "custom",
]


# =============================================================================
# MetricBuilder - The core builder class
# =============================================================================


@dataclass(frozen=True)
class _MetricDefinition:
    """Internal metric definition."""

    name: str
    aggregation: AggregationType
    system_metric: SystemMetric | None = None
    inverted: bool = False
    description: str = ""


class MetricBuilder:
    """
    Fluent builder for metrics.

    Provides methods that signal what kind of extraction/comparison is needed:
    - from_output(): Simple value extraction from output dict
    - compute(): Complex transform with access to inputs AND output
    - compare_to_expected(): Compare output to ground truth
    - evaluate(): Use an external evaluator

    Instantiation Patterns:
    ---------------------

    1. System metric (auto-derives aggregation):
        MetricBuilder(
            "accuracy",
            system_metric=SystemMetric.ACCURACY,  # Auto-derives AVERAGE
        )

    2. System metric with custom name (still auto-derives):
        MetricBuilder(
            "field_accuracy",
            system_metric=SystemMetric.ACCURACY,  # Auto-derives AVERAGE
            description="Per-field accuracy score",
        )

    3. Non-system metric (explicit aggregation required):
        MetricBuilder(
            "field_count",
            aggregation=AggregationType.AVERAGE,  # Must specify
            description="Average field count",
        )

    4. System metric with explicit override (not recommended):
        MetricBuilder(
            "custom_runtime",
            aggregation=AggregationType.MAX,      # Overrides auto-derived SUM
            system_metric=SystemMetric.RUNTIME_MS,
        )

    5. Inverted metrics (error_free → error_rate):
        MetricBuilder(
            "error_free",
            system_metric=SystemMetric.ERROR_RATE,
            inverted=True,  # 1.0 - value for system aggregation
        )

    6. Assertion metrics (fail tests on failure):
        MetricBuilder(
            "structure_valid",
            system_metric=SystemMetric.ERROR_RATE,
            inverted=True,
            assertion_mode=True,
            failure_threshold=0.0,  # Fail if value <= 0
        )

    Usage Examples:
        metrics.runtime_ms.from_output("metadata.runtime_ms")
        metrics.custom("accuracy", system_metric=SystemMetric.ACCURACY).compute(my_fn)
        metrics.custom("completeness", aggregation=AggregationType.AVERAGE).compute(my_fn)
    """

    __slots__ = (
        "_name",
        "_aggregation",
        "_system_metric",
        "_inverted",
        "_description",
        "_assertion_mode",
        "_failure_threshold",
        "_namespace",
    )

    def __init__(
        self,
        name: str,
        aggregation: AggregationType | None = None,
        system_metric: SystemMetric | None = None,
        inverted: bool = False,
        description: str = "",
        assertion_mode: bool = False,
        failure_threshold: float | int | None = None,
        namespace: Any | None = None,
    ):
        from .categories.system import get_aggregation_type

        self._name = name
        self._system_metric = system_metric
        # Derive aggregation from system_metric if provided, otherwise use explicit or default
        if system_metric is not None and aggregation is None:
            self._aggregation = get_aggregation_type(system_metric)
        else:
            self._aggregation = aggregation or AggregationType.AVERAGE
        self._inverted = inverted
        self._description = description
        self._assertion_mode = assertion_mode
        self._failure_threshold = failure_threshold
        self._namespace = namespace

    @property
    def name(self) -> str:
        """The metric name."""
        return self._name

    @property
    def aggregation(self) -> AggregationType:
        """How this metric is aggregated."""
        assert (
            self._aggregation is not None
        ), "Aggregation should never be None after __init__"
        return self._aggregation

    @property
    def system_metric(self) -> SystemMetric | None:
        """The system-level metric category."""
        return self._system_metric

    def from_output(self, path_or_fn: str | Callable[[Any], Any]) -> Metric:
        """
        Extract metric value directly from component output.

        Args:
            path_or_fn: Either:
                - A dot-path string like "metadata.runtime_ms"
                - A lambda that takes output and returns the value

        Returns:
            Configured Metric ready for @track

        Examples:
            metrics.runtime_ms.from_output("metadata.runtime_ms")
            metrics.custom("cost").from_output(lambda o: o["usage"]["tokens"] * 0.001)
        """
        if isinstance(path_or_fn, str):
            path = path_or_fn

            def extractor(inputs: dict[str, Any], output: Any) -> Any:
                value = output
                for key in path.split("."):
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        return 0
                return value if value is not None else 0

        else:
            fn = path_or_fn

            def extractor(inputs: dict[str, Any], output: Any) -> Any:
                return fn(output)

        assert self._aggregation is not None, "Aggregation must be set"
        metric = Metric(
            name=self._name,
            transform=extractor,
            agg=self._aggregation,
            failure_threshold=self._failure_threshold if self._assertion_mode else None,
            is_assertion=self._assertion_mode,
            system_metric=self._system_metric,
        )
        metric._namespace = self._namespace
        return metric

    def compute(self, transform_fn: Callable[[dict[str, Any], Any], Any]) -> Metric:
        """
        Compute metric using full transform signature with inputs and output.

        Args:
            transform_fn: Function with signature (inputs, output) -> value

        Returns:
            Configured Metric ready for @track

        Example:
            def my_accuracy(inputs, output):
                expected = inputs.get("data")
                actual = output.get("taxAssistRule")
                return calculate_similarity(expected, actual)

            metrics.custom("accuracy").compute(my_accuracy)
        """
        assert self._aggregation is not None, "Aggregation must be set"
        metric = Metric(
            name=self._name,
            transform=transform_fn,
            agg=self._aggregation,
            failure_threshold=self._failure_threshold if self._assertion_mode else None,
            is_assertion=self._assertion_mode,
            system_metric=self._system_metric,
        )
        metric._namespace = self._namespace
        return metric

    def assert_passes(
        self,
        transform_fn: Callable[[dict[str, Any], Any], Any],
        threshold: float | int = 0.0,
    ) -> Metric:
        """
        Create an assertion metric that fails tests if the transform returns a failure value.

        Use this to prevent PRs from passing when critical metrics fail. The test will
        raise an AssertionError if the metric value is <= threshold (default 0).

        Args:
            transform_fn: Function with signature (inputs, output) -> value
            threshold: Minimum passing value (default 0.0). Values <= this fail the test.

        Returns:
            Configured Metric with assertion enabled

        Example:
            # Fail test if structure validation returns False
            m.structure_valid.assert_passes(
                lambda i, o: validate_structure(o),
                threshold=0.0  # Any value <= 0 fails
            )

            # Fail test if accuracy below 0.8
            m.custom("accuracy").assert_passes(
                accuracy_transform,
                threshold=0.8
            )
        """
        assert self._aggregation is not None, "Aggregation must be set"
        return Metric(
            name=self._name,
            transform=transform_fn,
            agg=self._aggregation,
            failure_threshold=threshold,
            is_assertion=True,
            system_metric=self._system_metric,
        )

    def compare_to_expected(
        self,
        expected_key: str,
        output_key: str | None = None,
        similarity_fn: Callable[[Any, Any], float] | None = None,
    ) -> Metric:
        """
        Compare component output to expected value from expectations.

        Args:
            expected_key: Dot-path to expected value in inputs
            output_key: Dot-path to value in component output to compare
            similarity_fn: Custom similarity function (default: structural comparison)

        Returns:
            Metric that computes similarity between expected and actual
        """

        def extract_by_path(obj: Any, path: str) -> Any:
            if not path:
                return obj
            value = obj
            for key in path.split("."):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return None
            return value

        def default_similarity(expected: Any, actual: Any) -> float:
            import json

            if expected is None or actual is None:
                return 0.0

            # Serialize both to JSON strings for exact comparison
            try:
                expected_str = json.dumps(expected, sort_keys=True, default=str)
                actual_str = json.dumps(actual, sort_keys=True, default=str)
                return 1.0 if expected_str == actual_str else 0.0
            except (TypeError, ValueError):
                # Fallback to direct equality for non-serializable objects
                return 1.0 if expected == actual else 0.0

        compare_fn = similarity_fn or default_similarity

        def transform(inputs: dict[str, Any], output: Any) -> float:
            expected = extract_by_path(inputs, expected_key)
            actual = extract_by_path(output, output_key) if output_key else output
            return compare_fn(expected, actual)

        assert self._aggregation is not None, "Aggregation must be set"
        return Metric(
            name=self._name,
            transform=transform,
            agg=self._aggregation,
            system_metric=self._system_metric,
        )

    def evaluate(self, evaluator: Callable[[dict[str, Any], Any], Any]) -> Metric:
        """
        Use an external evaluator (Azure AI SDK, custom evaluator, etc).

        This is an alias for compute() with a more descriptive name for
        evaluation-style transforms.

        Args:
            evaluator: Function with signature (inputs, output) -> score

        Returns:
            Configured Metric ready for @track
        """
        return self.compute(evaluator)


# =============================================================================
# Built-in Metrics (module-level constants)
# =============================================================================

runtime_ms: MetricBuilder = MetricBuilder(
    name="runtime_ms",
    system_metric=SystemMetric.RUNTIME_MS,
    description="Total execution time in milliseconds",
)

memory_mb: MetricBuilder = MetricBuilder(
    name="memory_mb",
    system_metric=SystemMetric.MEMORY_MB,
    description="Peak memory usage in megabytes",
)

error_free: MetricBuilder = MetricBuilder(
    name="error_free",
    system_metric=SystemMetric.ERROR_RATE,
    inverted=True,
    description="Whether execution completed without errors (1.0 = no errors)",
    assertion_mode=True,  # Automatically fail tests when errors occur
    failure_threshold=0.0,
)

structure_valid: MetricBuilder = MetricBuilder(
    name="structure_valid",
    system_metric=SystemMetric.ERROR_RATE,
    inverted=True,
    description="Whether output structure is valid (1.0 = valid)",
)

expected_match: MetricBuilder = MetricBuilder(
    name="expected_match",
    system_metric=SystemMetric.ACCURACY,
    description="How closely output matches expected value (0.0-1.0)",
)

assert_passes: MetricBuilder = MetricBuilder(
    name="assert_passes",
    aggregation=AggregationType.FAILURE,
    system_metric=SystemMetric.ERROR_RATE,
    description="Assert metric passes threshold, fail test otherwise",
)


# =============================================================================
# Factory Function for Custom Metrics
# =============================================================================


def custom(
    name: str,
    aggregation: AggregationType | None = None,
    system_metric: SystemMetric | None = None,
    description: str = "",
) -> MetricBuilder:
    """
    Create an ad-hoc custom metric.

    Use this for one-off metrics that don't need to be reused across
    multiple components. For reusable metrics, define them as module-level
    constants in your own metrics module.

    Args:
        name: Unique metric name
        aggregation: How to aggregate across components (auto-derived from system_metric if provided)
        system_metric: Optional system-level metric category (determines aggregation if not explicit)
        description: Human-readable description

    Returns:
        MetricBuilder for the custom metric

    Examples:
        # 1. System metric (auto-derive aggregation):
        metrics.custom("my_accuracy", system_metric=SystemMetric.ACCURACY)

        # 2. Non-system metric (explicit aggregation):
        metrics.custom(
            "tokens_per_second",
            aggregation=AggregationType.AVERAGE,
            description="Token processing rate"
        ).from_output(lambda o: o["tokens"] / (o["runtime_ms"] / 1000))

        # 3. System metric with custom name:
        metrics.custom(
            "field_accuracy",
            system_metric=SystemMetric.ACCURACY,  # Auto-derives AVERAGE
            description="Per-field accuracy"
        )

        # 4. Different aggregation types for non-system metrics:
        metrics.custom("api_calls", aggregation=AggregationType.SUM)
        metrics.custom("max_depth", aggregation=AggregationType.MAX)
        metrics.custom("min_latency", aggregation=AggregationType.MIN)
    """
    return MetricBuilder(
        name=name,
        aggregation=aggregation,
        system_metric=system_metric,
        description=description or f"Custom metric: {name}",
    )
