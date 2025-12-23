"""
Built-in Metrics Module

Convenience re-export of metrics from sleuth.metrics with a shorter name.

Usage:
    from sleuth import m

    @track(metrics=[
        m.runtime_ms.from_output("metadata.runtime_ms"),
        m.memory_mb.from_output("metadata.memory_mb"),
    ])
    def my_component(text: str) -> dict:
        ...
"""

from .metrics import (
    runtime_ms,
    memory_mb,
    error_free,
    structure_valid,
    expected_match,
    assert_passes,
    custom,
    MetricBuilder,
)


class _BuiltinMetrics:
    """
    Built-in metrics for common use cases.

    This is a convenience wrapper that re-exports metrics from sleuth.metrics
    with explicit type annotations for IDE autocomplete support.
    """

    runtime_ms: MetricBuilder = runtime_ms
    memory_mb: MetricBuilder = memory_mb
    error_free: MetricBuilder = error_free
    structure_valid: MetricBuilder = structure_valid
    expected_match: MetricBuilder = expected_match
    assert_passes: MetricBuilder = assert_passes
    custom = staticmethod(custom)


# Singleton instance - the main entry point
m: _BuiltinMetrics = _BuiltinMetrics()

__all__ = ["m"]
