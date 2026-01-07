"""Squirt - Metrics Library for Component Testing

A framework for collecting, aggregating, and analyzing metrics from instrumented components.
Uses a unified base class pattern for both built-in and custom metrics.

Usage:
    from squirt import m, track, configure

    # Configure squirt (optional)
    configure(
        results_dir="./tests/results",
        expectations_file="./tests/data/expectations.json",
    )

    @track(
        expects="description",
        metrics=[
            m.runtime_ms.from_output("metadata.runtime_ms"),
            m.memory_mb.from_output("metadata.memory_mb"),
        ],
    )
    def my_component(text: str) -> dict:
        return {"result": process(text), "metadata": {...}}

Plugin Usage:
    from squirt.contrib.tax import tax

    @track(metrics=[
        m.runtime_ms.from_output("metadata.runtime_ms"),
        tax.field_accuracy.compute(my_fn),
    ])
    def extract_tax_rules(text: str) -> dict:
        ...
"""

import contextvars
from pathlib import Path
from typing import Optional

from .categories.system import SystemMetric
from .core.decorator import (
    configure_expectations,
    get_expectations,
    get_test_context,
    set_test_context,
    track,
)
from .core.types import (
    AggregationType,
    Metric,
    MetricResult,
)
from .filters import (
    only_namespaces,
    skip_namespaces,
    when_env,
)

# Import m from the typed module for IDE support
# The _M class has explicit type annotations that Pylance can follow
from .m import m

__version__ = "0.1.0"

# Global configuration
_config: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "squirt_config",
    default={
        "results_dir": "./tests/results",
        "history_dir": "./tests/history",
        "expectations_file": None,
    },
)


def configure(
    results_dir: str | Path | None = None,
    history_dir: str | Path | None = None,
    expectations_file: str | Path | None = None,
) -> None:
    """
    Configure squirt settings.

    Args:
        results_dir: Directory to store metric results
        history_dir: Directory to store historical reports
        expectations_file: Path to expectations.json file

    Example:
        from squirt import configure

        configure(
            results_dir="./tests/metrics/results",
            history_dir="./tests/metrics/history",
            expectations_file="./tests/data/expectations.json",
        )
    """
    config = _config.get().copy()

    if results_dir is not None:
        config["results_dir"] = str(results_dir)
    if history_dir is not None:
        config["history_dir"] = str(history_dir)
    if expectations_file is not None:
        config["expectations_file"] = str(expectations_file)
        # Only load expectations if file exists
        expectations_path = Path(expectations_file)
        if expectations_path.exists():
            configure_expectations(path=expectations_file)

    _config.set(config)


def get_config() -> dict:
    """Get current squirt configuration."""
    return _config.get().copy()


# Global metrics client instance
_metrics_client: contextvars.ContextVar[Optional["MetricsClient"]] = (
    contextvars.ContextVar("squirt_metrics_client", default=None)
)


def configure_metrics(
    results_dir: str = "tests/results",
    history_dir: str | None = None,
    expectations_path: str | None = None,
    persist: bool = True,
) -> "MetricsClient":
    """
    Initialize the metrics system for a test session.

    This is a convenience function that creates a MetricsClient and stores it globally.

    Args:
        results_dir: Directory to store metric results
        history_dir: Directory for historical reports
        expectations_path: Optional path to expectations.json file
        persist: Whether to persist results to files (default: True)

    Returns:
        MetricsClient instance
    """
    from .client import MetricsClient as MC

    # Also update global config
    configure(
        results_dir=results_dir,
        history_dir=history_dir or f"{results_dir}/../history",
        expectations_file=expectations_path,
    )

    client = MC(results_dir=results_dir, persist=persist)
    _metrics_client.set(client)
    return client


def get_metrics_client() -> "MetricsClient":
    """
    Get the current metrics client, creating one if necessary.

    Returns:
        MetricsClient instance
    """
    from .client import MetricsClient as MC

    client = _metrics_client.get()
    if client is None:
        client = MC(persist=True)
        _metrics_client.set(client)
    return client


# Import optional components for export
# Import analysis components
from .analysis import (
    DependencyGraphBuilder,
    analyze_codebase,
    visualize_graph,
)

# Import category helpers
from .categories import (
    INVERTED_METRICS,
    get_aggregation_type,
    should_invert,
)
from .client import MetricsClient
from .extensions import register_aggregation, register_system_metric

# Import reporting components
from .reporting import (
    ComponentReport,
    Insight,
    InsightGenerator,
    Severity,
    SystemHeartbeat,
    aggregate_by_system_metrics,
    aggregate_metrics_from_graph,
    aggregate_results,
    aggregate_values,
    find_bottlenecks,
    find_underperforming_components,
    generate_heartbeat,
    generate_heartbeat_from_graph,
    generate_insight_report,
    save_hierarchical_reports,
)

__all__ = [
    # Core API
    "m",
    "track",
    "configure",
    "get_config",
    "configure_expectations",
    "get_expectations",
    "configure_metrics",
    "get_metrics_client",
    "set_test_context",
    "get_test_context",
    # Filters
    "skip_namespaces",
    "only_namespaces",
    "when_env",
    # Client
    "MetricsClient",
    # Extensions
    "register_aggregation",
    "register_system_metric",
    # Types
    "AggregationType",
    "Metric",
    "MetricResult",
    "SystemMetric",
    # Categories
    "INVERTED_METRICS",
    "get_aggregation_type",
    "should_invert",
    # Reporting - Core
    "aggregate_values",
    "aggregate_results",
    "generate_heartbeat",
    # Reporting - Graph-based
    "aggregate_metrics_from_graph",
    "generate_heartbeat_from_graph",
    "save_hierarchical_reports",
    "aggregate_by_system_metrics",
    "find_bottlenecks",
    "find_underperforming_components",
    # Reporting - Data classes
    "ComponentReport",
    "SystemHeartbeat",
    # Insights
    "Severity",
    "Insight",
    "InsightGenerator",
    "generate_insight_report",
    # Analysis
    "DependencyGraphBuilder",
    "analyze_codebase",
    "visualize_graph",
]
